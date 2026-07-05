"""Distillation — SimBa student-teacher (DAgger) + SimBa actor-critic-teacher (distillation+RL), Muon.

Registers the DAgger pair into ``distillation_runner`` and the distillation+RL pair into
``on_policy_runner`` so ``eval(class_name)`` in each runner can resolve them.
"""

import rsl_rl.runners.distillation_runner as _dr
import rsl_rl.runners.on_policy_runner as _opr

from .algorithm import MuonDistillation, MuonPPODistill
from .networks import SimBaActorCriticTeacher, SimBaStudentTeacher

# DAgger (DistillationRunner)
_dr.SimBaStudentTeacher = SimBaStudentTeacher
_dr.MuonDistillation = MuonDistillation

# distillation + RL (OnPolicyRunner)
_opr.SimBaActorCriticTeacher = SimBaActorCriticTeacher
_opr.MuonPPODistill = MuonPPODistill

__all__ = [
    "MuonDistillation",
    "MuonPPODistill",
    "SimBaActorCriticTeacher",
    "SimBaStudentTeacher",
]
