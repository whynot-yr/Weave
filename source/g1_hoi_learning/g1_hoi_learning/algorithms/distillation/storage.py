"""RSL-RL PPO storage adapter: CPU uint16 depth and pre-step owned snapshots."""

from __future__ import annotations

import numpy as np
import torch
from rsl_rl.storage import RolloutStorage

from g1_hoi_learning.algorithms.dagger.distributed import DistributedContext
from g1_hoi_learning.algorithms.vision import DEPTH_FAR, depth_from_mm


class DepthRolloutStorage(RolloutStorage):
    """Reuse RSL-RL's GAE; replace its full-float allocation and batch transfer."""

    def __init__(self, num_envs, rollout_steps, obs, context: DistributedContext):
        # Do not call the full-float observation allocator in RolloutStorage.__init__.
        self.num_envs = num_envs
        self.num_transitions_per_env = rollout_steps
        self.capacity = num_envs * rollout_steps
        self.device = "cpu"
        self.context = context
        self.training_type = "rl"
        self.step = 0
        self.staged = False
        self.obs_keys = tuple(obs)
        self.arrays = {
            key: np.empty(
                (self.capacity, *value.shape[1:]),
                dtype=np.uint16 if key == "head_depth" else value.detach().cpu().numpy().dtype,
            )
            for key, value in obs.items()
        }
        self.actions = self.values = self.actions_log_prob = self.mu = self.sigma = None
        shape = (rollout_steps, num_envs, 1)
        self.rewards = torch.zeros(shape)
        self.dones = torch.zeros(shape, dtype=torch.uint8)
        self.returns = torch.zeros(shape)
        self.advantages = torch.zeros(shape)

    @property
    def size(self):
        return self.step * self.num_envs

    def clear(self):
        super().clear()
        self.staged = False

    @torch.no_grad()
    def stage(self, transition, metadata=None):
        """Copy state, sampled action and old distribution BEFORE env.step."""
        if self.step >= self.num_transitions_per_env or self.staged:
            raise ValueError("Rollout is full or a staged transition has not been completed")
        values = {key: transition.observations[key].detach() for key in self.obs_keys}
        values.update({key: value.detach() for key, value in (metadata or {}).items()})
        for key, value in values.items():
            if value.shape[0] != self.num_envs or not torch.isfinite(value).all():
                raise ValueError(f"Non-finite or invalid rollout observation: {key}")
        depth = values["head_depth"]
        if (depth < 0).any() or (depth > 1).any():
            raise ValueError("Expected preprocessed depth in [0, 1]")
        values["head_depth"] = (depth * (1000 * DEPTH_FAR)).round().to(torch.uint16)
        start, stop = self.size, self.size + self.num_envs
        for key, value in values.items():
            array = value.cpu().numpy()
            if key not in self.arrays:
                self.arrays[key] = np.empty((self.capacity, *array.shape[1:]), dtype=array.dtype)
            target = self.arrays[key]
            if target.shape[1:] != array.shape[1:] or target.dtype != array.dtype:
                raise ValueError(f"Rollout shape/dtype changed for {key}")
            target[start:stop] = array

        for name, value in (
            ("actions", transition.actions),
            ("values", transition.values),
            ("actions_log_prob", transition.actions_log_prob.reshape(-1, 1)),
            ("mu", transition.action_mean),
            ("sigma", transition.action_sigma),
        ):
            if not torch.isfinite(value).all():
                raise ValueError(f"Non-finite rollout {name}")
            if getattr(self, name) is None:
                setattr(self, name, torch.empty(self.num_transitions_per_env, *value.shape))
            getattr(self, name)[self.step].copy_(value)
        self.staged = True

    @torch.no_grad()
    def add_transitions(self, transition):
        """RSL-RL process_env_step has already applied timeout reward bootstrap."""
        if not self.staged:
            raise ValueError("Stage the pre-action state before completing the transition")
        if not torch.isfinite(transition.rewards).all():
            raise ValueError("Non-finite rollout rewards")
        self.rewards[self.step].copy_(transition.rewards.reshape(-1, 1))
        self.dones[self.step].copy_(transition.dones.reshape(-1, 1))
        self.step += 1
        self.staged = False

    @torch.no_grad()
    def compute_returns(self, last_values, gamma, lam, normalize_advantage=True):
        if self.size != self.capacity or self.staged:
            raise ValueError("Cannot compute returns on an incomplete rollout")
        # Keep the historical GAE recursion and done masks without duplicating it.
        super().compute_returns(last_values.detach().cpu(), gamma, lam, normalize_advantage=False)
        if normalize_advantage:
            self.normalize_advantages()

    @torch.no_grad()
    def normalize_advantages(self):
        adv = self.advantages.double()
        sums = self.context.sum(torch.tensor([adv.sum(), adv.square().sum(), adv.numel()], dtype=torch.float64))
        mean = sums[0] / sums[2]
        variance = ((sums[1] - sums[0].square() / sums[2]) / (sums[2] - 1).clamp_min(1)).clamp_min(0)
        self.advantages.sub_(mean.item()).div_(variance.sqrt().item() + 1e-8)

    def minibatches(self, num_mini_batches):
        if self.size != self.capacity or self.staged:
            raise ValueError("Cannot train on an incomplete rollout")
        if not 1 <= num_mini_batches <= self.size:
            raise ValueError("Invalid minibatch count")
        yield from np.array_split(torch.randperm(self.size).numpy(), num_mini_batches)

    def batch(self, indices, device):
        """Decode/transfer only this microbatch; metadata stays on the CPU."""
        if not len(indices) or np.any(indices < 0) or np.any(indices >= self.size):
            raise ValueError("Invalid rollout indices")
        obs = {key: torch.from_numpy(self.arrays[key][indices]).to(device) for key in self.obs_keys}
        obs["head_depth"] = depth_from_mm(obs["head_depth"])

        def take(name):
            return getattr(self, name).flatten(0, 1)[indices].to(device)

        return (
            obs,
            take("actions"),
            take("values"),
            take("advantages"),
            take("returns"),
            take("actions_log_prob"),
            take("mu"),
            take("sigma"),
            (None, None),
            None,
        )
