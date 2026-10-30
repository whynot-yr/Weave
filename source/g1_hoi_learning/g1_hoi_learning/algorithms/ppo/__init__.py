"""PPO (on-policy) — networks + algorithm + runner.
"""

import rsl_rl.runners.on_policy_runner as _opr

from .algorithm import MuonPPO
from .networks import SimBaActorCritic
from .runner import PPORunner
from .multi_critic_networks import MultiCriticSimBaActorCritic
from .multi_critic_algorithm import MultiCriticMuonPPO

# Register custom classes so RSL-RL can resolve `class_name` strings.
_opr.SimBaActorCritic = SimBaActorCritic
_opr.MuonPPO = MuonPPO
_opr.MultiCriticSimBaActorCritic = MultiCriticSimBaActorCritic
_opr.MultiCriticMuonPPO = MultiCriticMuonPPO

__all__ = [
    "MuonPPO",
    "PPORunner",
    "SimBaActorCritic",
]
