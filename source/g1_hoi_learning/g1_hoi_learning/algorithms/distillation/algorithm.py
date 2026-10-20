"""3904770 Muon PPO+BC, adapted for CPU vector storage, microbatches and DDP.

The PPO/value/BC objectives, KL learning-rate rule and cosine BC schedule come
from 3904770889ee67b34113df01f3f5533add0a468b. RSL-RL still owns action recording,
timeout reward correction and GAE. Only the storage/update transport is adapted.
"""

from __future__ import annotations

import math
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from g1_hoi_learning.algorithms.dagger.config import DaggerTrainConfig
from g1_hoi_learning.algorithms.dagger.distributed import DistributedContext
from g1_hoi_learning.algorithms.dagger.networks import TEACHER_GROUPS
from g1_hoi_learning.algorithms.ppo.algorithm import MuonPPO

from .storage import VectorRolloutStorage


class MuonPPODistill(MuonPPO):
    def __init__(self, policy, cfg: DaggerTrainConfig, context: DistributedContext | None = None):
        self.cfg = cfg
        self.context = context or DistributedContext.single(next(policy.parameters()).device)
        super().__init__(
            policy,
            weight_decay=cfg.weight_decay,
            device=str(self.context.device),
            num_learning_epochs=cfg.num_learning_epochs,
            num_mini_batches=cfg.num_mini_batches,
            clip_param=cfg.clip_param,
            gamma=cfg.gamma,
            lam=cfg.gae_lambda,
            value_loss_coef=cfg.value_loss_coef,
            entropy_coef=cfg.entropy_coef,
            learning_rate=cfg.learning_rate,
            max_grad_norm=cfg.max_grad_norm,
            use_clipped_value_loss=cfg.use_clipped_value_loss,
            schedule=cfg.schedule,
            desired_kl=cfg.desired_kl,
        )
        self.bc_coef_init = cfg.bc_coef
        self.bc_coef_anneal_iters = cfg.bc_coef_anneal_iters
        self._it = 0
        self.bc_coef = cfg.bc_coef
        self.context.require_same((policy.actor.spec, policy.critic_spec, cfg), "policy and training configuration")
        self.model = (
            DDP(
                policy,
                device_ids=[self.context.device.index] if self.context.device.type == "cuda" else None,
                broadcast_buffers=False,  # All normalization statistics stay fixed.
            )
            if self.context.distributed
            else policy
        )
        self.trainable_parameters = [p for p in policy.parameters() if p.requires_grad]

    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape=None):
        if training_type != "rl":
            raise ValueError("PPO+BC uses RL storage, not RSL-RL's separate distillation runner")
        groups = tuple(
            dict.fromkeys(
                (
                    *self.policy.actor.observation_groups,
                    *TEACHER_GROUPS,
                    *self.policy.critic_spec["groups"],
                )
            )
        )
        self.storage = VectorRolloutStorage(
            num_envs, num_transitions_per_env, {g: obs[g] for g in groups}, self.context
        )

    def act(self, obs, metadata=None):
        action = super().act(obs)
        # RSL-RL retains an obs reference until process_env_step. The simulator may
        # overwrite it in-place, so own the state and old distribution right now.
        self.storage.stage(self.transition, metadata)
        return action

    @torch.no_grad()
    def compute_returns(self, obs):
        # Propagate local errors before entering global advantage collectives.
        last_values = self.context.all_call(lambda: self.policy.evaluate(obs).detach())
        self.context.require_finite(last_values, "bootstrap value")
        self.context.all_call(
            lambda: self.storage.compute_returns(last_values, self.gamma, self.lam, normalize_advantage=False)
        )
        self.context.require_finite(self.storage.advantages, "GAE advantages")
        self.storage.normalize_advantages()

    def _loss_terms(self, output, batch):
        """The historical clipped PPO, clipped value and mean-only teacher MSE."""
        obs, _, old_values, advantages, returns, old_log_prob, _, _, _, _ = batch
        mu, _, value, log_prob, entropy = output
        ratio = torch.exp(log_prob - old_log_prob.squeeze(-1))
        surrogate = -advantages.squeeze(-1) * ratio
        clipped = -advantages.squeeze(-1) * ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param)
        surrogate_loss = torch.max(surrogate, clipped).mean()
        if self.use_clipped_value_loss:
            value_clipped = old_values + (value - old_values).clamp(-self.clip_param, self.clip_param)
            value_loss = torch.max((value - returns).square(), (value_clipped - returns).square()).mean()
        else:
            value_loss = (value - returns).square().mean()
        entropy_mean = entropy.mean()
        rl_loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_mean
        bc_loss = nn.functional.mse_loss(mu, self.policy.teacher_act(obs))
        loss = self.bc_coef * bc_loss + (1.0 - self.bc_coef) * rl_loss
        clip_fraction = ((ratio - 1).abs() > self.clip_param).float().mean()
        return loss, bc_loss, surrogate_loss, value_loss, entropy_mean, clip_fraction

    def _adapt_learning_rate(self, kl):
        # Same thresholds/bounds as 3904770, applied once per effective minibatch.
        if self.schedule == "adaptive":
            if self.context.rank == 0:
                if kl > self.desired_kl * 2:
                    self.learning_rate = max(1e-6, self.learning_rate / 1.5)
                elif 0 < kl < self.desired_kl / 2:
                    self.learning_rate = min(1e-2, self.learning_rate * 1.5)
            self.learning_rate = self.context.rank_zero_call(lambda: self.learning_rate)
            for group in self.optimizer.param_groups:
                group["lr"] = self.learning_rate

    def set_iteration(self, iteration):
        self._it = iteration
        self.bc_coef = (
            self.bc_coef_init * 0.5 * (1 + math.cos(math.pi * iteration / self.bc_coef_anneal_iters))
            if iteration < self.bc_coef_anneal_iters
            else 0.0
        )

    def update(self):
        rollout = self.storage
        self.context.require_same((rollout.size, rollout.capacity), "rollout sizes")
        if rollout.size != rollout.capacity:
            raise ValueError("Cannot train on an incomplete rollout")
        self.model.train()
        # loss, BC, surrogate, value, entropy, clip fraction, KL, sample count
        totals = torch.zeros(8, dtype=torch.float64)
        grad_sum, updates = 0.0, 0
        coefficient = self.bc_coef
        for _ in range(self.num_learning_epochs):
            for indices in rollout.minibatches(self.num_mini_batches):
                self.optimizer.zero_grad(set_to_none=True)
                batch_size = len(indices)
                micro_size = self.cfg.micro_batch_size or batch_size
                kl_sum = 0.0
                for start in range(0, batch_size, micro_size):
                    ids = indices[start : start + micro_size]
                    batch = self.context.all_call(lambda: rollout.batch(ids, self.device))
                    obs, actions, _, _, _, _, old_mu, old_sigma, _, _ = batch
                    final = start + micro_size >= batch_size
                    sync = self.model.no_sync() if self.context.distributed and not final else nullcontext()
                    with sync:
                        output = self.model(obs, actions)
                        terms = self._loss_terms(output, batch)
                        self.context.require_finite(torch.stack(terms), "PPO+BC loss")
                        (terms[0] * (len(ids) / batch_size)).backward()
                    with torch.no_grad():
                        mu, sigma = output[:2]
                        kl = (
                            (
                                torch.log(sigma / old_sigma + 1e-5)
                                + (old_sigma.square() + (old_mu - mu).square()) / (2 * sigma.square())
                                - 0.5
                            )
                            .sum(-1)
                            .mean()
                        )
                        self.context.require_finite(kl, "policy KL")
                        kl_sum += kl.item() * len(ids)
                    totals[:6] += torch.tensor([term.detach().item() for term in terms], dtype=torch.float64) * len(ids)
                    totals[6] += kl.item() * len(ids)
                    totals[7] += len(ids)
                global_kl = self.context.sum(torch.tensor([kl_sum, batch_size], dtype=torch.float64))
                self._adapt_learning_rate((global_kl[0] / global_kl[1]).item())
                grad = nn.utils.clip_grad_norm_(self.trainable_parameters, self.max_grad_norm, error_if_nonfinite=False)
                self.context.require_finite(grad, "gradient norm")
                self.optimizer.step()
                self.policy.protect_std()
                self.context.require_finite(self.policy.std, "action standard deviation")
                grad_sum += grad.item()
                updates += 1

        totals = self.context.sum(totals)
        names = ("total_loss", "train_action_mse", "surrogate", "value_function", "entropy", "clip_fraction", "kl")
        metrics = {name: (totals[i] / totals[7]).item() for i, name in enumerate(names)}
        returns, errors = rollout.returns.double(), (rollout.returns - rollout.values).double()
        moments = self.context.sum(
            torch.tensor([returns.sum(), returns.square().sum(), errors.sum(), errors.square().sum(), returns.numel()])
        )
        n = moments[4]
        return_var = (moments[1] / n - (moments[0] / n).square()).clamp_min(0)
        error_var = (moments[3] / n - (moments[2] / n).square()).clamp_min(0)
        metrics.update(
            explained_variance=(1 - error_var / (return_var + 1e-8)).item(),
            bc_coef=coefficient,
            ppo_coef=1 - coefficient,
            learning_rate=self.learning_rate,
            action_std=self.policy.std.detach().mean().item(),
            grad_norm=grad_sum / updates,
            optimizer_updates=updates,
            samples_trained=int(totals[7].item()),
            minibatches_per_epoch=self.num_mini_batches,
        )
        self.set_iteration(self._it + 1)
        # Runner clears at the next collection boundary; no historical replay is kept.
        return metrics
