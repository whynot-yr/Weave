"""PPO (on-policy) — networks + algorithm + runner.
"""

import rsl_rl.runners.on_policy_runner as _opr

from .algorithm import MuonAdamWWrapper, MuonPPO, OptimizerGroup
from .networks import SimBa, SimBaActorCritic, SimBaBlock
from .runner import PPORunner

# Register custom classes so RSL-RL can resolve `class_name` strings.
_opr.SimBaActorCritic = SimBaActorCritic
_opr.MuonPPO = MuonPPO

__all__ = [
    "MuonAdamWWrapper",
    "MuonPPO",
    "OptimizerGroup",
    "PPORunner",
    "SimBa",
    "SimBaActorCritic",
    "SimBaBlock",
]
