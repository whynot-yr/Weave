"""Distillation (DAgger) — SimBa student-teacher + Muon.
"""

import rsl_rl.runners.distillation_runner as _dr

from .algorithm import MuonDistillation
from .networks import SimBaStudentTeacher

_dr.SimBaStudentTeacher = SimBaStudentTeacher
_dr.MuonDistillation = MuonDistillation

__all__ = [
    "MuonDistillation",
    "SimBaStudentTeacher",
]
