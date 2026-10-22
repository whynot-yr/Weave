"""Per-algorithm subpackages — networks, algorithm, runner under one namespace."""

from . import ppo  # noqa: F401  (registers SimBaActorCritic, MuonPPO into rsl_rl's namespace)
from . import distillation  # noqa: F401  (registers SimBaActorCriticTeacher, MuonPPODistill)
