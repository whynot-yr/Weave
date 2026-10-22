"""Distillation + RL — SimBa actor-critic with a frozen SimBa teacher, Muon.

Registers the distillation+RL pair into ``on_policy_runner`` so ``eval(class_name)``
in the runner can resolve them.
"""

import rsl_rl.runners.on_policy_runner as _opr

from .algorithm import MuonPPODistill
from .networks import SimBaActorCriticTeacher

_opr.SimBaActorCriticTeacher = SimBaActorCriticTeacher
_opr.MuonPPODistill = MuonPPODistill

__all__ = [
    "MuonPPODistill",
    "SimBaActorCriticTeacher",
]
