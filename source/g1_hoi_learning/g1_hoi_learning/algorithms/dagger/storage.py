"""Action-label vector storage; active PPO uses distillation.storage.VectorRolloutStorage."""

from __future__ import annotations

import numpy as np
import torch

from .networks import STUDENT_GROUPS


class RolloutBuffer:
    """Own observations before env.step; reuse the allocation for the next window."""

    def __init__(self, num_envs: int, rollout_steps: int):
        if num_envs < 1 or rollout_steps < 1:
            raise ValueError("Rollout dimensions must be positive")
        self.num_envs = num_envs
        self.capacity = num_envs * rollout_steps
        self.size = 0
        self.arrays: dict[str, np.ndarray] = {}

    def clear(self):
        self.size = 0

    @torch.no_grad()
    def add(self, obs, teacher_actions, metadata: dict[str, torch.Tensor]):
        stop = self.size + self.num_envs
        if stop > self.capacity:
            raise ValueError("Rollout is full; clear it before collecting another window")
        values = {g: obs[g].detach() for g in STUDENT_GROUPS}
        values["teacher_action"] = teacher_actions.detach()
        values.update({key: value.detach() for key, value in metadata.items()})
        for key, value in values.items():
            if value.shape[0] != self.num_envs:
                raise ValueError(f"Rollout batch dimension changed for {key}")
            if not torch.isfinite(value).all():
                raise ValueError("Non-finite DAgger observation/action")
        samples = {key: value.cpu().numpy() for key, value in values.items()}
        if not self.arrays:
            self.arrays = {
                key: np.empty((self.capacity, *value.shape[1:]), dtype=value.dtype) for key, value in samples.items()
            }
        if samples.keys() != self.arrays.keys():
            raise ValueError("Rollout schema changed")
        for key, value in samples.items():
            target = self.arrays[key]
            if target.shape[1:] != value.shape[1:] or target.dtype != value.dtype:
                raise ValueError(f"Rollout shape/dtype changed for {key}")
        for key, value in samples.items():
            self.arrays[key][self.size : stop] = value
        self.size = stop

    def minibatches(self, num_mini_batches: int):
        """A new permutation per epoch; balanced batches include every sample once."""
        if self.size != self.capacity:
            raise ValueError("Cannot train on an incomplete rollout")
        if not 1 <= num_mini_batches <= self.size:
            raise ValueError("Invalid num_mini_batches")
        indices = torch.randperm(self.size).numpy()
        yield from np.array_split(indices, num_mini_batches)

    def batch(self, indices: np.ndarray, device: str):
        """Transfer only a microbatch; metadata never goes to the learner GPU."""
        if not len(indices) or np.any(indices < 0) or np.any(indices >= self.size):
            raise ValueError("Invalid rollout sample indices")
        obs = {group: torch.from_numpy(self.arrays[group][indices]).to(device) for group in STUDENT_GROUPS}
        target = torch.from_numpy(self.arrays["teacher_action"][indices]).to(device)
        return obs, target
