"""Synchronous student rollouts followed by current-window action distillation."""

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
from .storage import RolloutBuffer

SCHEMA = "weave-depth-dagger-v2"
STUDENT_SCHEMAS = {SCHEMA, "weave-depth-dagger-v1"}


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        if env.num_envs != cfg.num_envs:
            raise ValueError("Environment count differs from training config")
        self.env, self.teacher, self.student, self.cfg = env, teacher, student, cfg
        self.device = str(next(student.parameters()).device)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.teacher_info = copy.deepcopy(teacher_info)
        self.environment_contract = copy.deepcopy(environment_contract)
        self.resolved_env = copy.deepcopy(resolved_env)
        self.algorithm = DaggerAlgorithm(student, cfg)
        self.rollout = RolloutBuffer(env.num_envs, cfg.rollout_steps)
        self.iteration = 0  # NEXT iteration; save only at completed rollout/update boundaries.
        self.total_transitions = 0
        self.total_optimizer_updates = 0
        self.episode_ids = torch.zeros(env.num_envs, dtype=torch.long, device=self.device)
        self.episode_steps = torch.zeros_like(self.episode_ids)
        self.writer = None

    def learn(self, initial_obs, is_running=lambda: True, max_iterations: int | None = None):
        if max_iterations is not None and max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        end = self.cfg.num_iterations
        if max_iterations is not None:
            end = min(end, self.iteration + max_iterations)
        if end <= self.iteration:
            raise ValueError("No iterations remaining")
        self.writer = SummaryWriter(str(self.run_dir / "tensorboard"), purge_step=self.iteration)
        obs = initial_obs
        env_ids = torch.arange(self.env.num_envs, device=self.device)
        try:
            while self.iteration < end and is_running():
                start = time.perf_counter()
                if self.device.startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats(self.device)
                self.student.eval()
                self.rollout.clear()
                reward_sum, done_count, rollout_mse = 0.0, 0, 0.0
                for _ in range(self.cfg.rollout_steps):
                    if not is_running():
                        return  # Never save a partially collected/updated iteration.
                    with torch.no_grad():
                        a_teacher = self.teacher(obs)
                        a_student = self.student(obs)
                        if not torch.isfinite(a_student).all():
                            raise FloatingPointError("Non-finite student rollout action")
                        command = self.env.unwrapped.command_manager.get_term("motion")
                        metadata = {
                            "clip_id": command.env_clip,
                            "ref_step": command.time_steps,
                            "env_id": env_ids,
                            "episode_id": self.episode_ids,
                            "episode_step": self.episode_steps,
                        }
                        self.rollout.add(obs, a_teacher, metadata)
                        rollout_mse += (a_student - a_teacher).square().mean().item()
                        obs, rewards, dones, _ = self.env.step(a_student)
                        reward_sum += rewards.sum().item()
                        done_count += dones.sum().item()
                        self.episode_steps += 1
                        self.episode_ids += dones.long()
                        self.episode_steps[dones.bool()] = 0
                rollout_seconds = time.perf_counter() - start
                if not is_running():
                    return
                training_start = time.perf_counter()
                losses = self.algorithm.update(self.rollout, self.device)
                training_seconds = time.perf_counter() - training_start
                self.total_transitions += self.rollout.size
                self.total_optimizer_updates += losses["optimizer_updates"]
                metrics = {
                    **losses,
                    "rollout_action_mse": rollout_mse / self.cfg.rollout_steps,
                    "reward_per_step": reward_sum / self.rollout.size,
                    "done_count": done_count,
                    "rollout_samples": self.rollout.size,
                    "transitions": self.total_transitions,
                    "total_optimizer_updates": self.total_optimizer_updates,
                    "rollout_seconds": rollout_seconds,
                    "training_seconds": training_seconds,
                    "seconds": time.perf_counter() - start,
                    "process_ram_peak_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
                }
                if self.device.startswith("cuda"):
                    metrics["torch_cuda_peak_allocated_gib"] = torch.cuda.max_memory_allocated(self.device) / (1024**3)
                    metrics["torch_cuda_peak_reserved_gib"] = torch.cuda.max_memory_reserved(self.device) / (1024**3)
                for key, value in metrics.items():
                    self.writer.add_scalar(f"dagger/{key}", value, self.iteration)
                with (self.run_dir / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps({"iteration": self.iteration, **metrics}) + "\n")
                self.iteration += 1
                if self.iteration % self.cfg.log_interval == 0 or self.iteration == 1:
                    print(
                        f"[DAgger] {self.iteration}/{end} "
                        f"mse={losses['train_action_mse']:.6f} samples={self.rollout.size} "
                        f"updates={losses['optimizer_updates']} seconds={metrics['seconds']:.2f}",
                        flush=True,
                    )
                if self.iteration % self.cfg.save_interval == 0:
                    self.save(self.run_dir / f"model_{self.iteration}.pt")
            self.save(self.run_dir / f"model_{self.iteration}.pt")
            self.save_student(self.run_dir / "student_weights.pt")
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

    def save_student(self, path: Path):
        """Evaluation/warm-start weights; not an optimizer resume checkpoint."""
        state = self._student_snapshot()
        state.update(kind="student_weights")
        temporary = path.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(path)

    def load_student(self, path: str):
        """Warm start v1/v2 weights with a fresh optimizer and training counters."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("schema") not in STUDENT_SCHEMAS or state.get("actor_spec") != self.student.spec:
            raise ValueError("Incompatible student weights; use a DAgger checkpoint, not a TorchScript export")
        if state["teacher_info"]["sha256"] != self.teacher_info["sha256"]:
            raise ValueError("Student weights use a different teacher checkpoint")
        self.student.load_state_dict(state["student_state_dict"], strict=True)

    def save(self, path: Path):
        state = self._student_snapshot()
        state.update(
            {
                "kind": "training",
                "optimizer_state_dict": self.algorithm.optimizer.state_dict(),
                "train_cfg": asdict(self.cfg),
                "total_transitions": self.total_transitions,
                "total_optimizer_updates": self.total_optimizer_updates,
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
        # Prune only this run's numbered snapshots, after committing the new one.
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
        if state.get("schema") != SCHEMA:
            raise ValueError(
                "Legacy DAgger training cannot resume this schedule; use --student_checkpoint to warm start"
            )
        if state["actor_spec"] != self.student.spec:
            raise ValueError("Incompatible DAgger actor")
        saved_cfg = asdict(DaggerTrainConfig(**state["train_cfg"]))
        current_cfg = asdict(self.cfg)
        # Memory/save controls may change; effective batches and the schedule may not.
        for key in ("micro_batch_size", "keep_last_checkpoints"):
            saved_cfg.pop(key)
            current_cfg.pop(key)
        if saved_cfg != current_cfg or state["environment_contract"] != self.environment_contract:
            raise ValueError("Resume requires the original training and environment contracts")
        if state["teacher_info"]["sha256"] != self.teacher_info["sha256"]:
            raise ValueError("Resume teacher checkpoint differs")
        self.student.load_state_dict(state["student_state_dict"], strict=True)
        self.algorithm.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.iteration = state["next_iteration"]
        self.total_transitions = state["total_transitions"]
        self.total_optimizer_updates = state["total_optimizer_updates"]
        self.episode_ids.copy_(state["episode_ids"].to(self.device) + 1)
        self.episode_steps.zero_()  # Physics restarts on resume.
        self.rollout.clear()
        torch.set_rng_state(state["rng"]["torch"])
        np.random.set_state(state["rng"]["numpy"])
        random.setstate(state["rng"]["python"])
        if state["rng"]["cuda"]:
            torch.cuda.set_rng_state_all(state["rng"]["cuda"])
