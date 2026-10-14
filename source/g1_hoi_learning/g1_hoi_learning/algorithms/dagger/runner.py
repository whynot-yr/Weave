"""Four-stage action DAgger with same-state labels and persistent CPU replay."""

from __future__ import annotations

import copy
import hashlib
import json
import random
import re
import resource
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .algorithm import DaggerAlgorithm
from .config import DaggerTrainConfig
from .networks import StudentWeaveActor, TeacherWeaveActor, export_student
from .storage import ReservoirBuffer, capture_samples, concatenate_samples

SCHEMA = "weave-depth-dagger-v1"
STAGE_IDS = {"visual_warmup": 0, "joint_warmup": 1, "dagger_decay": 2, "student_only": 3}


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_actions(teacher_actions, student_actions, beta: float):
    """Select a whole action per env, never interpolate joint commands."""
    selected = torch.rand(teacher_actions.shape[0], device=teacher_actions.device) < beta
    return torch.where(selected[:, None], teacher_actions, student_actions), selected


class DaggerRunner:
    def __init__(
        self,
        env,
        teacher: TeacherWeaveActor,
        student: StudentWeaveActor,
        cfg: DaggerTrainConfig,
        run_dir: str,
        teacher_info: dict,
        environment_contract: dict,
        resolved_env: dict,
    ):
        self.env, self.teacher, self.student, self.cfg = env, teacher, student, cfg
        self.device = str(next(student.parameters()).device)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.teacher_info = copy.deepcopy(teacher_info)
        self.environment_contract = copy.deepcopy(environment_contract)
        self.resolved_env = copy.deepcopy(resolved_env)
        self.algorithm = DaggerAlgorithm(student, cfg)
        self.buffer = ReservoirBuffer(cfg.buffer_capacity, cfg.seed)
        self.iteration = 0  # NEXT iteration; snapshots are only taken at completed rollout/update boundaries.
        self.total_transitions = 0
        self.episode_ids = torch.zeros(env.num_envs, dtype=torch.long, device=self.device)
        self.episode_steps = torch.zeros_like(self.episode_ids)
        self.writer = None

    def learn(self, initial_obs, is_running=lambda: True, max_iterations: int | None = None):
        self.writer = SummaryWriter(str(self.run_dir / "tensorboard"), purge_step=self.iteration)
        obs = initial_obs
        end = self.cfg.total_iterations
        if max_iterations is not None:
            end = min(end, self.iteration + max_iterations)
        if end <= self.iteration:
            raise ValueError("No iterations remaining")
        try:
            while self.iteration < end and is_running():
                start = time.perf_counter()
                if self.device.startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats(self.device)
                stage, beta = self.cfg.stage(self.iteration)
                self.algorithm.set_stage(stage)
                chunks = []
                reward_sum, done_count, teacher_count, rollout_mse = 0.0, 0, 0, 0.0
                for _ in range(self.cfg.rollout_steps):
                    if not is_running():
                        return  # Do not save a partially collected/updated iteration.
                    with torch.no_grad():
                        a_teacher = self.teacher(obs)
                        a_student = self.student(obs)
                        actions, selected = choose_actions(a_teacher, a_student, beta)
                        command = self.env.unwrapped.command_manager.get_term("motion")
                        metadata = {
                            "clip_id": command.env_clip,
                            "ref_step": command.time_steps,
                            "env_id": torch.arange(self.env.num_envs, device=self.device),
                            "episode_id": self.episode_ids,
                            "episode_step": self.episode_steps,
                            "stage": torch.full_like(self.episode_ids, STAGE_IDS[stage]),
                            "executed_teacher": selected,
                        }
                        chunks.append(capture_samples(obs, a_teacher, metadata))
                        rollout_mse += (a_student - a_teacher).square().mean().item()
                        teacher_count += selected.sum().item()
                        obs, rewards, dones, _ = self.env.step(actions)
                        reward_sum += rewards.sum().item()
                        done_count += dones.sum().item()
                        self.episode_steps += 1
                        self.episode_ids += dones.long()
                        self.episode_steps[dones.bool()] = 0
                current = concatenate_samples(chunks)
                losses = self.algorithm.update(current, self.buffer, self.device)
                self.buffer.add(current)
                self.total_transitions += len(current["teacher_action"])
                metrics = {
                    **losses,
                    "beta": beta,
                    "stage_id": STAGE_IDS[stage],
                    "teacher_fraction": teacher_count / len(current["teacher_action"]),
                    "rollout_action_mse": rollout_mse / self.cfg.rollout_steps,
                    "reward_per_step": reward_sum / len(current["teacher_action"]),
                    "done_count": done_count,
                    "buffer_size": self.buffer.size,
                    "transitions": self.total_transitions,
                    "seconds": time.perf_counter() - start,
                    "process_ram_peak_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
                }
                if self.device.startswith("cuda"):
                    metrics["torch_cuda_peak_allocated_gib"] = torch.cuda.max_memory_allocated(self.device) / (1024**3)
                    metrics["torch_cuda_peak_reserved_gib"] = torch.cuda.max_memory_reserved(self.device) / (1024**3)
                for key, value in metrics.items():
                    self.writer.add_scalar(f"dagger/{key}", value, self.iteration)
                with (self.run_dir / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps({"iteration": self.iteration, "stage": stage, **metrics}) + "\n")
                self.iteration += 1
                if self.iteration == self.cfg.total_iterations or self.cfg.stage(self.iteration)[0] != stage:
                    self.save_student(self.run_dir / f"student_{stage}.pt", completed_stage=stage)
                if self.iteration % self.cfg.log_interval == 0 or self.iteration == 1:
                    print(
                        f"[DAgger] {self.iteration}/{end} {stage} beta={beta:.3f} "
                        f"mse={losses['action_mse']:.6f} replay={self.buffer.size} "
                        f"seconds={metrics['seconds']:.2f}",
                        flush=True,
                    )
                if self.iteration % self.cfg.save_interval == 0:
                    self.save(self.run_dir / f"model_{self.iteration}.pt")
            self.save(self.run_dir / f"model_{self.iteration}.pt")
            export_student(self.student, str(self.run_dir / "student_policy.pt"))
        finally:
            self.writer.close()

    def _student_snapshot(self) -> dict:
        return {
            "schema": SCHEMA,
            "actor_spec": self.student.spec,
            "student_state_dict": self.student.state_dict(),
            "next_iteration": self.iteration,
            "teacher_info": self.teacher_info,
            "environment_contract": self.environment_contract,
            "resolved_env": self.resolved_env,
        }

    def save_student(self, path: Path, completed_stage: str):
        """Evaluation-compatible weights without optimizer or replay; not a resume checkpoint."""
        state = self._student_snapshot()
        state.update(kind="student_weights", completed_stage=completed_stage)
        temporary = path.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(path)

    def save(self, path: Path):
        state = self._student_snapshot()
        state.update(
            {
                "kind": "training",
                "optimizer_state_dict": self.algorithm.optimizer.state_dict(),
                "joint_enabled": self.algorithm.joint_enabled,
                "train_cfg": asdict(self.cfg),
                "stage": self.cfg.stage(self.iteration)[0],
                "beta": self.cfg.stage(self.iteration)[1],
                "total_transitions": self.total_transitions,
                "replay": self.buffer.state_dict(),
                "episode_ids": self.episode_ids.cpu(),
                "rng": {
                    "torch": torch.get_rng_state(),
                    "numpy": np.random.get_state(),
                    "python": random.getstate(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                },
            }
        )
        temporary = path.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(path)
        # Only prune this run's numbered full snapshots, after the new one is committed.
        if path.parent.resolve() == self.run_dir.resolve() and re.fullmatch(r"model_\d+\.pt", path.name):
            checkpoints = [
                entry for entry in self.run_dir.glob("model_*.pt") if re.fullmatch(r"model_\d+\.pt", entry.name)
            ]
            checkpoints.sort(key=lambda entry: int(entry.stem.removeprefix("model_")), reverse=True)
            for old in checkpoints[self.cfg.keep_last_checkpoints :]:
                old.unlink()

    def load(self, path: str):
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("kind", "training") != "training":
            raise ValueError("Resume requires a full model_<iteration>.pt checkpoint, not student weights")
        if state["schema"] != SCHEMA or state["actor_spec"] != self.student.spec:
            raise ValueError("Incompatible DAgger checkpoint")
        saved_cfg = asdict(DaggerTrainConfig(**state["train_cfg"]))
        current_cfg = asdict(self.cfg)
        # Memory/save controls may change on resume; the effective batch and learning schedule may not.
        for key in ("micro_batch_size", "keep_last_checkpoints"):
            saved_cfg.pop(key)
            current_cfg.pop(key)
        if saved_cfg != current_cfg or state["environment_contract"] != self.environment_contract:
            raise ValueError("Resume requires the original training and environment contracts")
        if state["teacher_info"]["sha256"] != self.teacher_info["sha256"]:
            raise ValueError("Resume teacher checkpoint differs")
        self.student.load_state_dict(state["student_state_dict"], strict=True)
        if state["joint_enabled"]:
            self.algorithm.set_stage("joint_warmup")  # build identical parameter groups before loading AdamW
        self.algorithm.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.buffer.load_state_dict(state["replay"])
        self.iteration = state["next_iteration"]
        self.total_transitions = state["total_transitions"]
        self.episode_ids.copy_(state["episode_ids"].to(self.device) + 1)
        self.episode_steps.zero_()  # Simulator physics is restarted, not serialized.
        self.algorithm.set_stage(self.cfg.stage(self.iteration)[0])
        torch.set_rng_state(state["rng"]["torch"])
        np.random.set_state(state["rng"]["numpy"])
        random.setstate(state["rng"]["python"])
        if state["rng"]["cuda"]:
            torch.cuda.set_rng_state_all(state["rng"]["cuda"])
