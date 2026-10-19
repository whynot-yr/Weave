"""Student Gaussian rollouts and the restored 3904770 PPO+BC training loop."""

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
from .distributed import DistributedContext
from .networks import StudentWeaveActor, TeacherWeaveActor, export_student

SCHEMA = "weave-depth-dagger-v4"
STUDENT_SCHEMAS = {SCHEMA, "weave-depth-dagger-v3", "weave-depth-dagger-v2", "weave-depth-dagger-v1"}


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
        context: DistributedContext | None = None,
        *,
        policy=None,
    ):
        self.env, self.teacher, self.student, self.cfg = env, teacher, student, cfg
        self.device = str(next(student.parameters()).device)
        self.run_dir = Path(run_dir)
        self.context = context or DistributedContext.single(self.device)
        self.context.require_same((env.num_envs, cfg.num_envs), "per-rank environment counts")
        if env.num_envs != cfg.num_envs:
            raise ValueError("Environment count differs from training config")
        self.context.require_same(
            (str(self.run_dir), teacher_info, environment_contract), "run directory, teacher and environment contract"
        )
        self.context.rank_zero_call(lambda: self.run_dir.mkdir(parents=True, exist_ok=True))
        self.teacher_info = copy.deepcopy(teacher_info)
        self.environment_contract = copy.deepcopy(environment_contract)
        self.resolved_env = copy.deepcopy(resolved_env)
        if policy is None or policy.actor is not student or policy.teacher is not teacher:
            raise ValueError("Supply the teacher-initialized Actor-Critic wrapping this student and teacher")
        self.policy = policy
        self.algorithm = DaggerAlgorithm(policy, cfg, self.context)
        self.rollout = None
        self.iteration = 0  # NEXT iteration; save only at completed rollout/update boundaries.
        self.total_transitions = 0
        self.total_optimizer_updates = 0
        self.episode_ids = torch.zeros(env.num_envs, dtype=torch.long, device=self.device)
        self.episode_steps = torch.zeros_like(self.episode_ids)
        self.writer = None

    def learn(self, initial_obs, is_running=lambda: True, max_iterations: int | None = None):
        self.context.require_same((self.iteration, self.cfg.num_iterations, max_iterations), "iteration budget")
        if max_iterations is not None and max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        end = self.cfg.num_iterations
        if max_iterations is not None:
            end = min(end, self.iteration + max_iterations)
        if end <= self.iteration:
            raise ValueError("No iterations remaining")

        def create_writer():
            self.writer = SummaryWriter(str(self.run_dir / "tensorboard"), purge_step=self.iteration)

        self.context.rank_zero_call(create_writer)
        obs = initial_obs
        if self.algorithm.storage is None:
            self.context.all_call(
                lambda: self.algorithm.init_storage(
                    "rl", self.env.num_envs, self.cfg.rollout_steps, obs, [self.student.spec["num_actions"]]
                )
            )
            self.rollout = self.algorithm.storage
        env_ids = torch.arange(self.env.num_envs, device=self.device) + self.context.rank * self.env.num_envs
        try:
            while self.iteration < end and self.context.all_running(is_running()):
                start = time.perf_counter()
                if self.device.startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats(self.device)
                self.policy.eval()
                self.rollout.clear()
                reward_sum, done_count, rollout_mse = 0.0, 0, 0.0
                for _ in range(self.cfg.rollout_steps):
                    if not self.context.all_running(is_running()):
                        return  # All ranks discard a partial iteration together.

                    def collect_step():
                        with torch.no_grad():
                            a_teacher = self.teacher(obs)
                            command = self.env.unwrapped.command_manager.get_term("motion")
                            metadata = {
                                "clip_id": command.env_clip,
                                "ref_step": command.time_steps,
                                "env_id": env_ids,
                                "episode_id": self.episode_ids,
                                "episode_step": self.episode_steps,
                            }
                            a_student = self.algorithm.act(obs, metadata)
                            error = (self.policy.action_mean - a_teacher).square().mean().item()
                            next_obs, rewards, dones, extras = self.env.step(a_student)
                            self.algorithm.process_env_step(next_obs, rewards, dones, extras)
                            self.episode_steps += 1
                            self.episode_ids += dones.long()
                            self.episode_steps[dones.bool()] = 0
                            return next_obs, rewards.sum().item(), dones.sum().item(), error

                    obs, reward, done, error = self.context.all_call(collect_step)
                    reward_sum += reward
                    done_count += done
                    rollout_mse += error
                rollout_seconds = time.perf_counter() - start
                if not self.context.all_running(is_running()):
                    return
                training_start = time.perf_counter()
                with torch.no_grad():
                    self.algorithm.compute_returns(obs)
                losses = self.algorithm.update()
                training_seconds = time.perf_counter() - training_start
                totals = self.context.sum(
                    torch.tensor(
                        [reward_sum, done_count, rollout_mse * self.env.num_envs, self.rollout.size],
                        dtype=torch.float64,
                    )
                )
                global_samples = int(totals[3].item())
                self.total_transitions += global_samples
                self.total_optimizer_updates += losses["optimizer_updates"]  # Synchronized updates, not rank sum.
                resources = {
                    "rank": self.context.rank,
                    "rollout_seconds": rollout_seconds,
                    "training_seconds": training_seconds,
                    "seconds": time.perf_counter() - start,
                    "process_ram_peak_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
                }
                if self.device.startswith("cuda"):
                    resources["torch_cuda_peak_allocated_gib"] = torch.cuda.max_memory_allocated(self.device) / (
                        1024**3
                    )
                    resources["torch_cuda_peak_reserved_gib"] = torch.cuda.max_memory_reserved(self.device) / (1024**3)
                rank_resources = self.context.gather_objects(resources)
                metrics = {
                    **losses,
                    "rollout_action_mse": (totals[2] / totals[3]).item(),
                    "reward_per_step": (totals[0] / totals[3]).item(),
                    "done_count": int(totals[1].item()),
                    "rollout_samples": global_samples,
                    "rollout_samples_per_rank": self.rollout.size,
                    "world_size": self.context.world_size,
                    "transitions": self.total_transitions,
                    "total_optimizer_updates": self.total_optimizer_updates,
                    "rank_resources": rank_resources,
                }
                metrics.update({key: max(item[key] for item in rank_resources) for key in resources if key != "rank"})

                def write_metrics():
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            self.writer.add_scalar(f"dagger/{key}", value, self.iteration)
                    with (self.run_dir / "metrics.jsonl").open("a") as stream:
                        stream.write(json.dumps({"iteration": self.iteration, **metrics}) + "\n")

                self.context.rank_zero_call(write_metrics)
                self.iteration += 1
                if self.context.rank == 0 and (self.iteration % self.cfg.log_interval == 0 or self.iteration == 1):
                    print(
                        f"[DAgger] {self.iteration}/{end} "
                        f"bc={losses['train_action_mse']:.6f} ppo={losses['surrogate']:.6f} "
                        f"value={losses['value_function']:.6f} alpha={losses['bc_coef']:.4f} "
                        f"global_samples={global_samples} "
                        f"updates={losses['optimizer_updates']} seconds={metrics['seconds']:.2f}",
                        flush=True,
                    )
                if self.iteration % self.cfg.save_interval == 0:
                    self.save(self.run_dir / f"model_{self.iteration}.pt")
            self.save(self.run_dir / f"model_{self.iteration}.pt")
            self.save_student(self.run_dir / "student_weights.pt")
            self.context.rank_zero_call(lambda: export_student(self.student, str(self.run_dir / "student_policy.pt")))
        finally:
            if self.writer is not None:
                self.writer.close()

    def _student_snapshot(self) -> dict:
        return {
            "schema": SCHEMA,
            "actor_spec": self.student.spec,
            "teacher_actor_spec": self.teacher.spec,
            "student_state_dict": self.student.state_dict(),
            "next_iteration": self.iteration,
            "teacher_info": self.teacher_info,
            "environment_contract": self.environment_contract,
            "resolved_env": self.resolved_env,
        }

    def save_student(self, path: Path):
        """Evaluation/warm-start weights; not an optimizer resume checkpoint."""

        def write():
            state = self._student_snapshot()
            state.update(kind="student_weights")
            temporary = path.with_suffix(".tmp")
            torch.save(state, temporary)
            temporary.replace(path)

        self.context.rank_zero_call(write)

    def load_student(self, path: str):
        """Explicit optional warm start; default training initializes from the teacher."""
        self.context.all_call(lambda: self._load_student(path))

    def _load_student(self, path: str):
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("schema") not in STUDENT_SCHEMAS or state.get("actor_spec") != self.student.spec:
            raise ValueError("Incompatible student weights; use a DAgger checkpoint, not a TorchScript export")
        if state.get("teacher_actor_spec", state["actor_spec"]) != self.teacher.spec:
            raise ValueError("Student weights use a different teacher observation contract")
        if state["teacher_info"]["sha256"] != self.teacher_info["sha256"]:
            raise ValueError("Student weights use a different teacher checkpoint")
        self.student.load_state_dict(state["student_state_dict"], strict=True)

    def _rank_state(self):
        return {
            "episode_ids": self.episode_ids.cpu(),
            "rng": {
                "torch": torch.get_rng_state(),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
                "cuda": torch.cuda.get_rng_state(self.device) if self.context.device.type == "cuda" else None,
            },
        }

    def save(self, path: Path):
        self.context.require_same(
            (str(path), self.iteration, self.total_transitions, self.total_optimizer_updates), "checkpoint boundary"
        )
        rank_states = self.context.gather_objects(self.context.all_call(self._rank_state))
        self.context.rank_zero_call(lambda: self._write_checkpoint(path, rank_states))

    def _write_checkpoint(self, path: Path, rank_states):
        state = self._student_snapshot()
        state.update(
            {
                "kind": "training",
                "critic_spec": self.policy.critic_spec,
                "critic_state_dict": self.policy.critic.state_dict(),
                "critic_normalizer_state_dict": self.policy.critic_obs_normalizer.state_dict(),
                "action_std": self.policy.std.detach().cpu(),
                "algorithm_state": {
                    "iteration": self.algorithm._it,
                    "bc_coef": self.algorithm.bc_coef,
                    "learning_rate": self.algorithm.learning_rate,
                },
                "optimizer_state_dict": self.algorithm.optimizer.state_dict(),
                "train_cfg": asdict(self.cfg),
                "total_transitions": self.total_transitions,
                "total_optimizer_updates": self.total_optimizer_updates,
                "world_size": self.context.world_size,
                "distributed_state_version": 1,
                "rank_states": rank_states,
            }
        )
        temporary = path.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(path)
        # Prune only this run's numbered snapshots, after committing the new one.
        if (
            self.cfg.keep_last_checkpoints is not None
            and path.parent.resolve() == self.run_dir.resolve()
            and re.fullmatch(r"model_\d+\.pt", path.name)
        ):
            checkpoints = [
                entry for entry in self.run_dir.glob("model_*.pt") if re.fullmatch(r"model_\d+\.pt", entry.name)
            ]
            checkpoints.sort(key=lambda entry: int(entry.stem.removeprefix("model_")), reverse=True)
            for old in checkpoints[self.cfg.keep_last_checkpoints :]:
                old.unlink()

    def load(self, path: str):
        self.context.all_call(lambda: self._load(path))

    def _load(self, path: str):
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("kind", "training") != "training":
            raise ValueError("Resume requires a full model_<iteration>.pt checkpoint, not student weights")
        if state.get("schema") != SCHEMA:
            raise ValueError(
                "Legacy DAgger training cannot resume the goal contract; start a fresh run from the teacher"
            )
        if state.get("world_size", 1) != self.context.world_size:
            raise ValueError("Resume requires the same world size; use --student_checkpoint to start a new run")
        if state["actor_spec"] != self.student.spec or state["critic_spec"] != self.policy.critic_spec:
            raise ValueError("Incompatible DAgger actor or critic")
        if state["teacher_actor_spec"] != self.teacher.spec:
            raise ValueError("Incompatible DAgger teacher observation contract")
        saved_cfg = asdict(DaggerTrainConfig(**state["train_cfg"]))
        current_cfg = asdict(self.cfg)
        # Memory/save controls may change; effective batches and the schedule may not.
        for key in ("micro_batch_size", "save_interval", "keep_last_checkpoints"):
            saved_cfg.pop(key)
            current_cfg.pop(key)
        if saved_cfg != current_cfg or state["environment_contract"] != self.environment_contract:
            raise ValueError("Resume requires the original training and environment contracts")
        if state["teacher_info"]["sha256"] != self.teacher_info["sha256"]:
            raise ValueError("Resume teacher checkpoint differs")
        self.student.load_state_dict(state["student_state_dict"], strict=True)
        self.policy.critic.load_state_dict(state["critic_state_dict"], strict=True)
        self.policy.critic_obs_normalizer.load_state_dict(state["critic_normalizer_state_dict"], strict=True)
        with torch.no_grad():
            self.policy.std.copy_(state["action_std"].to(self.device))
        # _load is already inside all_call: keep validation local until it returns.
        if not torch.isfinite(self.policy.std).all() or (self.policy.std <= 0).any():
            raise ValueError("Invalid restored action standard deviation")
        self.algorithm.optimizer.load_state_dict(state["optimizer_state_dict"])
        algorithm_state = state["algorithm_state"]
        if algorithm_state["iteration"] != state["next_iteration"]:
            raise ValueError("Checkpoint BC schedule does not match its completed iteration")
        self.algorithm.set_iteration(algorithm_state["iteration"])
        if abs(self.algorithm.bc_coef - algorithm_state["bc_coef"]) > 1e-12:
            raise ValueError("Checkpoint BC coefficient differs from its saved schedule")
        self.algorithm.learning_rate = algorithm_state["learning_rate"]
        for group in self.algorithm.optimizer.param_groups:
            group["lr"] = self.algorithm.learning_rate
        self.iteration = state["next_iteration"]
        self.total_transitions = state["total_transitions"]
        self.total_optimizer_updates = state["total_optimizer_updates"]
        rank_state = state["rank_states"][self.context.rank] if "rank_states" in state else state
        self.episode_ids.copy_(rank_state["episode_ids"].to(self.device) + 1)
        self.episode_steps.zero_()  # Physics restarts on resume.
        if self.rollout is not None:
            self.rollout.clear()
        self.algorithm.transition.clear()
        rng = rank_state["rng"]
        torch.set_rng_state(rng["torch"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
        if self.context.device.type == "cuda" and rng["cuda"] is not None:
            cuda_rng = rng["cuda"]
            if isinstance(cuda_rng, list):  # Earlier single-rank v2 snapshots stored all visible devices.
                cuda_rng = cuda_rng[self.context.device.index or 0] if cuda_rng else None
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, self.device)
