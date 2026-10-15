"""3904770 PPO+BC restored for the current student and standalone DaggerRunner.

The adapted constructors accept existing student/config objects, so they are
constructed directly rather than registered in RSL-RL's class-name factory.
"""

from .algorithm import MuonPPODistill
from .networks import SimBaActorCriticTeacher

__all__ = [
    "MuonPPODistill",
    "SimBaActorCriticTeacher",
]
