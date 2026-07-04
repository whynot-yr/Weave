from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.modules import ActorCritic
from rsl_rl.networks import EmpiricalNormalization

from g1_hoi_learning.algorithms.networks import SimBa

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
        num_actor_obs = 0
        for g in obs_groups["policy"]:
            assert len(obs[g].shape) == 2, "SimBaActorCritic only supports 1D observations."
            num_actor_obs += obs[g].shape[-1]
        num_critic_obs = 0
        for g in obs_groups["critic"]:
            assert len(obs[g].shape) == 2, "SimBaActorCritic only supports 1D observations."
            num_critic_obs += obs[g].shape[-1]

        self.state_dependent_std = False

        # Actor
        self.actor = SimBa(
            input_dim=num_actor_obs,
            output_dim=num_actions,
            hidden_dim=actor_hidden_dim,
            num_blocks=actor_num_blocks,
            expansion=expansion,
        )
        print(f"Actor SimBa: {self.actor}")

        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        # Critic
        self.critic = SimBa(
            input_dim=num_critic_obs,
            output_dim=1,
            hidden_dim=critic_hidden_dim,
            num_blocks=critic_num_blocks,
            expansion=expansion,
        )
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
