"""First-episode clip metrics using the same command metrics and success definition as PPO eval."""

from __future__ import annotations

import torch


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

    return {
        "num_clips": len(rows),
        "success_rate": sum(r["success"] for r in rows) / len(rows),
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
    }
