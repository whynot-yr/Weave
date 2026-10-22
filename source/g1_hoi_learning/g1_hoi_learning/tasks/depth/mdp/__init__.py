"""MDP for the depth-perception student."""

from g1_hoi_learning.tasks.hoi.mdp import *  # noqa: F401, F403

from .depth import object_depth_b  # noqa: F401
from .events import randomize_camera_extrinsics, randomize_camera_intrinsics  # noqa: F401
