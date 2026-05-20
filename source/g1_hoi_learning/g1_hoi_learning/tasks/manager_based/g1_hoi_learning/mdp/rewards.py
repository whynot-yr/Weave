from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, RewardTermCfg
from isaaclab.utils.math import quat_apply_inverse, quat_error_magnitude

from .commands import MotionCommand


def motion_anchor_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_anchor_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(
        torch.square(
            command.body_pos_w[:, command.body_indices] - command.robot_body_pos_w[:, command.body_indices]
        ),
        dim=-1,
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = (
        quat_error_magnitude(
            command.body_quat_w[:, command.body_indices],
            command.robot_body_quat_w[:, command.body_indices],
        )
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(
        torch.square(
            command.body_lin_vel_w[:, command.body_indices] - command.robot_body_lin_vel_w[:, command.body_indices]
        ),
        dim=-1,
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(
        torch.square(
            command.body_ang_vel_w[:, command.body_indices] - command.robot_body_ang_vel_w[:, command.body_indices]
        ),
        dim=-1,
    )
    return torch.exp(-error.mean(-1) / std**2)


# -- Object tracking rewards --
def object_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.ref_obj_pos_w - command.obj_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def object_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.ref_obj_quat_w, command.obj_quat_w) ** 2
    return torch.exp(-error / std**2)


def object_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.ref_obj_lin_vel_w - command.obj_lin_vel_w), dim=-1)
    return torch.exp(-error / std**2)


def object_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.ref_obj_ang_vel_w - command.obj_ang_vel_w), dim=-1)
    return torch.exp(-error / std**2)


# -- Hand-object relative pose tracking --
class motion_hand_obj_relative_pos_error_exp(ManagerTermBase):
    """Tracking error for hand body positions expressed in the object's local frame.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        robot: Articulation = env.scene["robot"]
        self.hand_idx = robot.find_bodies(cfg.params["hand_body_names"])[0]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        hand_body_names: list[str],
        std: float,
    ) -> torch.Tensor:
        command: MotionCommand = env.command_manager.get_term(command_name)
        n = len(self.hand_idx)

        # ref: hand positions in ref object frame
        ref_diff_w = command.body_pos_w[:, self.hand_idx] - command.ref_obj_pos_w[:, None, :]
        ref_pos_in_obj = quat_apply_inverse(
            command.ref_obj_quat_w[:, None, :].expand(-1, n, -1),
            ref_diff_w,
        )

        # sim: hand positions in sim object frame
        sim_diff_w = command.robot_body_pos_w[:, self.hand_idx] - command.obj_pos_w[:, None, :]
        sim_pos_in_obj = quat_apply_inverse(
            command.obj_quat_w[:, None, :].expand(-1, n, -1),
            sim_diff_w,
        )

        error = torch.sum((ref_pos_in_obj - sim_pos_in_obj) ** 2, dim=-1).mean(-1)
        return torch.exp(-error / std**2)


# -- Contact reward --
class contact_reward(ManagerTermBase):
    """Contact reward for hand bodies based on reference labels. Range [0, 1].

    Per-body scoring:
      ref=+1, sim=contact    → 1.0 (correct contact)
      ref=+1, sim=no contact → 0.0 (missed contact)
      ref=-1, sim=no contact → 1.0 (correct no contact)
      ref=-1, sim=contact    → 0.0 (wrong contact)
      ref=0                  → ignored (neutral)
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        robot: Articulation = env.scene["robot"]
        self.hand_idx = robot.find_bodies(cfg.params["hand_body_names"])[0]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        sensor_name: str,
        hand_body_names: list[str],
        saturate_force: float = 5.0,
    ) -> torch.Tensor:
        command: MotionCommand = env.command_manager.get_term(command_name)
        sensor = env.scene[sensor_name]

        ref_label = command.ref_contact_label[:, self.hand_idx]              # (num_envs, num_hand)
        forces = sensor.data.force_matrix_w[:, 0, self.hand_idx, :]           # (num_envs, num_hand, 3)
        sim_strength = (forces.norm(dim=-1) / saturate_force).clamp(0, 1)     # (num_envs, num_hand)

        target = (ref_label > 0).float()
        mask = (ref_label != 0).float()

        err = (target - sim_strength).abs() * mask
        score = 1.0 - err
        return (score * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
