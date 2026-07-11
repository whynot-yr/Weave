from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.modules import ActorCritic
from rsl_rl.networks import EmpiricalNormalization

from g1_hoi_learning.algorithms.networks import SimBa


class GroupEncoder(nn.Module):
    """Per-observation-group encoder bank.
    ``Linear -> SiLU -> ... -> Linear(latent_dim)``.
    """

    def __init__(self, group_dims: list[int], hidden_dims: list[list[int]], latent_dim: int) -> None:
        super().__init__()
        if len(group_dims) != len(hidden_dims):
            raise ValueError(f"group_dims ({len(group_dims)}) and hidden_dims ({len(hidden_dims)}) must align.")
        self.group_dims: list[int] = list(group_dims)
        self.in_features = sum(group_dims)
        self.out_features = latent_dim * len(group_dims)
        self.encoders = nn.ModuleList()
        for group_dim, hidden in zip(group_dims, hidden_dims):
            layers: list[nn.Module] = []
            for in_dim, out_dim in zip([group_dim] + list(hidden), hidden):
                layers += [nn.Linear(in_dim, out_dim), nn.SiLU()]
            layers.append(nn.Linear(hidden[-1] if hidden else group_dim, latent_dim))
            self.encoders.append(nn.Sequential(*layers))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = torch.split(x, self.group_dims, dim=-1)
        latents: list[torch.Tensor] = []
        i = 0
        for encoder in self.encoders:
            latents.append(encoder(chunks[i]))
            i += 1
        return torch.cat(latents, dim=-1)


class SimBaActorCritic(ActorCritic):
    """Actor-critic with SimBa backbones.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = True,
        critic_obs_normalization: bool = True,
        actor_hidden_dim: int = 512,
        critic_hidden_dim: int = 512,
        actor_num_blocks: int = 2,
        critic_num_blocks: int = 2,
        expansion: int = 4,
        latent_dim: int = 256,
        encoder_hidden_dims: dict[str, list[int]] | None = None,
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        **kwargs: Any,
    ) -> None:
        if state_dependent_std:
            raise NotImplementedError("SimBaActorCritic currently supports state-independent std only.")
        if kwargs:
            print(
                "SimBaActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )

        nn.Module.__init__(self)

        # Observation dimensions
        self.obs_groups = obs_groups
        for g in obs_groups["policy"] + obs_groups["critic"]:
            assert len(obs[g].shape) == 2, "SimBaActorCritic only supports 1D observations."
        num_actor_obs = sum(obs[g].shape[-1] for g in obs_groups["policy"])
        num_critic_obs = sum(obs[g].shape[-1] for g in obs_groups["critic"])

        self.state_dependent_std = False

        def build(groups: list[str], output_dim: int, hidden_dim: int, num_blocks: int) -> nn.Sequential:
            group_dims = [obs[g].shape[-1] for g in groups]
            hidden_dims = [encoder_hidden_dims[g] for g in groups]
            encoder = GroupEncoder(group_dims, hidden_dims, latent_dim)
            backbone = SimBa(encoder.out_features, output_dim, hidden_dim, num_blocks, expansion)
            return nn.Sequential(encoder, backbone)

        # Actor
        self.actor = build(obs_groups["policy"], num_actions, actor_hidden_dim, actor_num_blocks)
        print(f"Actor SimBa: {self.actor}")

        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        # Critic
        self.critic = build(obs_groups["critic"], 1, critic_hidden_dim, critic_num_blocks)
        print(f"Critic SimBa: {self.critic}")

        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        # Action noise (state-independent)
        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown noise_std_type: {noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated by ActorCritic._update_distribution)
        self.distribution: Normal | None = None

        # Disable args validation for speedup (matches parent)
        Normal.set_default_validate_args(False)
