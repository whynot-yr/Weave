from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.networks import EmpiricalNormalization

from g1_hoi_learning.algorithms.ppo.networks import SimBaActorCritic, build_group_backbone


class SimBaActorCriticTeacher(SimBaActorCritic):
    """SimBa actor-critic + a frozen SimBa teacher for the distillation + RL stage.

    Initialized from a *tracker* checkpoint (``actor.*`` / ``critic.*``) via
    :meth:`load_state_dict`:

      - ``teacher``        <- tracker frozen actor
      - student ``critic`` <- tracker critic
      - student ``actor``  <- tracker actor SimBa backbone + the per-group encoders of
                              the groups shared with the teacher.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        teacher_hidden_dim: int = 2048,
        teacher_num_blocks: int = 2,
        teacher_obs_normalization: bool = True,
        latent_dim: int = 256,
        encoder_hidden_dims: dict[str, list[int]] | None = None,
        expansion: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(
            obs,
            obs_groups,
            num_actions,
            latent_dim=latent_dim,
            encoder_hidden_dims=encoder_hidden_dims,
            expansion=expansion,
            **kwargs,
        )  # student actor + critic + std

        self.teacher = build_group_backbone(
            obs,
            obs_groups["teacher"],
            encoder_hidden_dims,
            latent_dim,
            output_dim=num_actions,
            hidden_dim=teacher_hidden_dim,
            num_blocks=teacher_num_blocks,
            expansion=expansion,
        )
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        num_teacher_obs = sum(obs[g].shape[-1] for g in obs_groups["teacher"])
        self.teacher_obs_normalization = teacher_obs_normalization
        self.teacher_obs_normalizer = (
            EmpiricalNormalization(num_teacher_obs) if teacher_obs_normalization else nn.Identity()
        )
        self.loaded_teacher = False

    def teacher_act(self, obs: TensorDict) -> torch.Tensor:
        x = self.teacher_obs_normalizer(torch.cat([obs[g] for g in self.obs_groups["teacher"]], dim=-1))
        with torch.no_grad():
            return self.teacher(x)

    # ------------------------------------------------------------------ loading

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        # Resume: a distillation checkpoint already carries teacher.* -> load everything.
        if any(k.startswith("teacher.") for k in state_dict):
            nn.Module.load_state_dict(self, state_dict, strict=strict)
            self.loaded_teacher = True
            return True
        # Fresh init from a TRACKER checkpoint (actor.* / critic.* / *_obs_normalizer.* / std).
        self._load_from_tracker(state_dict)
        self.loaded_teacher = True
        return False

    def _load_from_tracker(self, sd: dict) -> None:
        def sub(prefix: str) -> dict:
            return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

        # 1. teacher <- tracker actor
        self.teacher.load_state_dict(sub("actor."), strict=True)
        self.teacher.eval()
        if self.teacher_obs_normalization:
            self.teacher_obs_normalizer.load_state_dict(sub("actor_obs_normalizer."), strict=True)

        # 2. student critic <- tracker critic
        self.critic.load_state_dict(sub("critic."), strict=True)
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.load_state_dict(sub("critic_obs_normalizer."), strict=True)

        # 3. student actor <- tracker actor
        # SimBa backbone + encoders of the groups shared with the teacher
        self.actor[1].load_state_dict(sub("actor.1."), strict=True)  # SimBa backbone
        teacher_groups = self.obs_groups["teacher"]
        for i, g in enumerate(self.obs_groups["policy"]):
            if i < len(teacher_groups) and g == teacher_groups[i]:  # shared slot
                self.actor[0].encoders[i].load_state_dict(sub(f"actor.0.encoders.{i}."), strict=True)
