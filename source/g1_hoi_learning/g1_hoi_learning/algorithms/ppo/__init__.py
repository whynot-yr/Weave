"""PPO (on-policy) — networks + algorithm + runner.
"""

import rsl_rl.runners.on_policy_runner as _opr

from .algorithm import MuonPPO
from .networks import SimBaActorCritic
from .runner import PPORunner

# Register custom classes so RSL-RL can resolve `class_name` strings.
_opr.SimBaActorCritic = SimBaActorCritic
_opr.MuonPPO = MuonPPO

__all__ = [
    "MuonPPO",
    "PPORunner",
    "SimBaActorCritic",
]
