"""Four independent WEAVE value networks with the existing actor API."""
import torch
from torch import nn
from g1_hoi_learning.algorithms.networks import build_group_backbone
from .networks import SimBaActorCritic


class CriticEnsemble(nn.ModuleList):
    def forward(self, x):
        return torch.cat([critic(x) for critic in self], dim=-1)


class MultiCriticSimBaActorCritic(SimBaActorCritic):
    def update_normalization(self, obs):
        # RSL-RL collects under inference_mode; its normalizer replaces _std.
        # Keep persistent buffers normal tensors so checkpoint reload is legal.
        with torch.inference_mode(False), torch.no_grad():
            super().update_normalization(obs)

    def __init__(self, obs, obs_groups, num_actions, compile=True, **kwargs):
        super().__init__(obs, obs_groups, num_actions, compile=False, **kwargs)
        critics = [self.critic]
        for _ in range(3):
            critics.append(build_group_backbone(
                obs, obs_groups['critic'], kwargs['encoder_hidden_dims'],
                kwargs.get('latent_dim', 256), 1, kwargs.get('critic_hidden_dim', 512),
                kwargs.get('critic_num_blocks', 2), kwargs.get('expansion', 4),
            ))
        self.critic = CriticEnsemble(critics)
        if compile:
            self.actor.forward = torch.compile(self.actor.forward, fullgraph=True, dynamic=False)
            self.critic.forward = torch.compile(self.critic.forward, fullgraph=True, dynamic=False)
