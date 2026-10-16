"""First-episode clip metrics using the same command metrics and success definition as PPO eval."""

from __future__ import annotations

import math

import numpy as np
import torch

PERTURBATION_PROFILES = {
    "none": (0.0, 0.0),
    "initial_small": (0.002, 0.5),
    "initial_medium": (0.005, 1.0),
    "initial_large": (0.010, 2.0),
}


def build_eval_trials(
    clip_names: list[str], clip_ids: list[int] | None, num_trials: int, seed: int, perturbation: str
) -> list[dict]:
    """Sample conditions by (seed, clip, trial), independent of batching/selection order."""
    if num_trials < 1:
        raise ValueError("num_trials must be positive")
    if not 0 <= seed < 2**32:
        raise ValueError("Evaluation seed must be in [0, 2**32 - 1]")
    if perturbation not in PERTURBATION_PROFILES:
        raise ValueError(f"Unknown perturbation profile: {perturbation}")
    selected = list(range(len(clip_names))) if clip_ids is None else list(clip_ids)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("clip_ids must be nonempty and contain no duplicates")
    if any(i < 0 or i >= len(clip_names) for i in selected):
        raise ValueError(f"clip_ids must be in [0, {len(clip_names) - 1}]")
    xy_m, yaw_deg = PERTURBATION_PROFILES[perturbation]
    bounds = np.array([xy_m, xy_m, math.radians(yaw_deg)])
    trials = []
    for clip_id in selected:
        for trial_id in range(num_trials):
            trial_seed = int(np.random.SeedSequence([seed, clip_id, trial_id]).generate_state(1, dtype=np.uint64)[0])
            offsets = (np.random.default_rng(trial_seed).uniform(-1, 1, 3) * bounds).astype(np.float32)
            trials.append(
                {
                    "clip_id": clip_id,
                    "name": clip_names[clip_id],
                    "trial_id": trial_id,
                    "trial_seed": trial_seed,
                    "perturbation": perturbation,
                    "object_pose_offset": dict(zip(("x_m", "y_m", "yaw_rad"), offsets.tolist())),
                }
            )
    return trials


def assign_eval_clips(command, clip_ids: list[int]):
    """Reuse a single-object scene; pad the final batch and ignore its duplicate report rows."""
    if not command.cfg.eval_mode or command.motion.num_objects != 1:
        raise ValueError("Clip assignment requires single-object eval_mode")
    if not clip_ids or len(clip_ids) > command.num_envs:
        raise ValueError("Evaluation clip batch must fit the existing environment")
    ids = torch.tensor(clip_ids, device=command._eval_env_clip.device, dtype=torch.long)
    if (ids < 0).any() or (ids >= command.motion.num_clips).any():
        raise ValueError("Evaluation clip id out of range")
    padded = torch.cat((ids, ids[-1:].expand(command.num_envs - len(clip_ids))))
    command._eval_env_clip.copy_(padded)
    command.cfg.eval_clip_ids = padded.tolist()


def assign_eval_trials(command, trials: list[dict]):
    """Assign clips and fixed per-trial offsets before reset; pad unused environments."""
    assign_eval_clips(command, [trial["clip_id"] for trial in trials])
    offsets = torch.tensor(
        [[trial["object_pose_offset"][key] for key in ("x_m", "y_m", "yaw_rad")] for trial in trials],
        device=command._eval_object_pose_offsets.device,
        dtype=command._eval_object_pose_offsets.dtype,
    )
    padded = torch.cat((offsets, offsets[-1:].expand(command.num_envs - len(trials), -1)))
    command._eval_object_pose_offsets.copy_(padded)


class ClipEvaluator:
    def __init__(self, env, clip_names: list[str]):
        self.env = env.unwrapped
        self.command = self.env.command_manager.get_term("motion")
        self.term_mgr = self.env.termination_manager
        self.fail_names = [n for n in self.term_mgr.active_terms if not self.term_mgr.get_term_cfg(n).time_out]
        self.clip_ids = self.command.env_clip.clone()
        self.clip_len = self.command.motion.clip_lengths[self.clip_ids].clone()
        self.clip_names = [clip_names[int(i)] for i in self.clip_ids]
        self.finished = torch.zeros(env.num_envs, dtype=torch.bool, device=self.env.device)
        self.success = torch.zeros_like(self.finished)
        self.steps = torch.zeros_like(self.clip_ids)
        self.causes = ["incomplete"] * env.num_envs
        self.sums = {key: torch.zeros(env.num_envs, device=self.env.device) for key in self.command.metrics}
        self.contact_lost = torch.zeros(env.num_envs, device=self.env.device)
        self.action_change_sum = torch.zeros(env.num_envs, device=self.env.device)
        self.action_change_steps = torch.zeros_like(self.steps)
        self.previous_actions = None

    @property
    def max_steps(self):
        return int(self.clip_len.max()) + 5

    @property
    def done(self):
        return bool(self.finished.all())

    def collect(self, actions):
        self.command._update_metrics()
        active = ~self.finished
        for key in self.sums:
            self.sums[key] += self.command.metrics[key] * active
        forces = self.env.scene["contact_sensor"].data.force_matrix_w[:, 0].norm(dim=-1)
        desired = self.command.ref_contact_label > 0
        self.contact_lost += (desired & (forces <= 1.0)).sum(dim=-1) * active
        if self.previous_actions is not None:
            self.action_change_sum += (actions - self.previous_actions).square().mean(-1) * active
            self.action_change_steps += active
        self.previous_actions = actions.clone()
        self.steps += active

    def update(self, dones):
        newly = dones.bool() & ~self.finished
        flags = {name: self.term_mgr.get_term(name) for name in self.term_mgr.active_terms}
        terminated = self.env.reset_terminated
        self.success |= newly & flags["clip_end"] & ~terminated
        for i in newly.nonzero(as_tuple=False).flatten().tolist():
            if terminated[i]:
                self.causes[i] = next((name for name in self.fail_names if bool(flags[name][i])), "unknown")
            else:
                self.causes[i] = "clip_end" if bool(flags["clip_end"][i]) else "time_out"
        self.finished |= newly

    def rows(self):
        rows = []
        for i in range(len(self.clip_ids)):
            steps = max(int(self.steps[i]), 1)
            rows.append(
                {
                    "clip_id": int(self.clip_ids[i]),
                    "name": self.clip_names[i],
                    "length": int(self.clip_len[i]),
                    "steps": int(self.steps[i]),
                    "progress": min(int(self.steps[i]) / max(int(self.clip_len[i]), 1), 1.0),
                    "finished": bool(self.finished[i]),
                    "success": bool(self.success[i]),
                    "cause": self.causes[i],
                    "contact_lost_bodies_per_step": float(self.contact_lost[i]) / steps,
                    "action_change_steps": int(self.action_change_steps[i]),
                    "action_change_mse": float(self.action_change_sum[i]) / max(int(self.action_change_steps[i]), 1),
                    **{key: float(values[i]) / steps for key, values in self.sums.items()},
                }
            )
        return rows


def summarize(rows: list[dict], metric_keys: list[str]) -> dict:
    def aggregate(selected):
        result = {}
        for key in metric_keys:
            weight = "action_change_steps" if key == "action_change_mse" else "steps"
            count = sum(row[weight] for row in selected)
            result[key] = sum(row[key] * row[weight] for row in selected) / count if count else None
        return result

    def success_stats(selected):
        n = len(selected)
        successes = sum(row["success"] for row in selected)
        p, z = successes / n, 1.959963984540054
        denominator = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denominator
        radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
        return {
            "num_trials": n,
            "num_finished_trials": sum(row["finished"] for row in selected),
            "num_successes": successes,
            "success_rate": p,
            "success_rate_ci95": [max(0.0, center - radius), min(1.0, center + radius)],
        }

    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(row.get("clip_id", i), []).append(row)

    return {
        "num_clips": len(groups),
        **success_stats(rows),
        "progress_rate": sum(r["progress"] for r in rows) / len(rows),
        "complete": all(r["finished"] for r in rows),
        "metric_keys": metric_keys,
        "metrics_all": aggregate(rows),
        "metrics_success": aggregate([r for r in rows if r["success"]]),
        "failures": {
            cause: sum(r["cause"] == cause for r in rows)
            for cause in sorted({r["cause"] for r in rows if not r["success"]})
        },
        "clips": rows,
        "clip_summary": [
            {
                "clip_id": clip_id,
                "name": selected[0].get("name"),
                **success_stats(selected),
                "progress_rate": sum(row["progress"] for row in selected) / len(selected),
                "metrics_all": aggregate(selected),
                "failures": {
                    cause: sum(row["cause"] == cause for row in selected)
                    for cause in sorted({row["cause"] for row in selected if not row["success"]})
                },
            }
            for clip_id, selected in groups.items()
        ],
    }
