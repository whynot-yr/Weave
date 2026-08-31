"""Depth-specific motion command state."""

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import configclass

from g1_hoi_learning.tasks.hoi.mdp.commands import MotionCommand, MotionCommandCfg


class DepthMotionCommand(MotionCommand):
    """Motion command extended with actor-only odometry bias state."""

    def __init__(self, cfg: "DepthMotionCommandCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.odom_pos_error_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.odom_quat_error = torch.zeros(self.num_envs, 4, device=self.device)
        self.odom_quat_error[:, 0] = 1.0


@configclass
class DepthMotionCommandCfg(MotionCommandCfg):
    """Configuration for the depth-only motion command extension."""

    class_type: type = DepthMotionCommand
