from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, RewardTermCfg
from isaaclab.utils.math import quat_error_magnitude

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
        threshold: float = 1.0,
    ) -> torch.Tensor:
        command: MotionCommand = env.command_manager.get_term(command_name)
        sensor = env.scene[sensor_name]

        ref_label = command.ref_contact_label[:, self.hand_idx]              # (num_envs, num_hand)
        forces = sensor.data.force_matrix_w[:, 0, self.hand_idx, :]           # (num_envs, num_hand, 3)
        sim_contact = (forces.norm(dim=-1) > threshold).float()                # (num_envs, num_hand)

        # score per body: ref * (2*sim - 1) maps to [-1, 1], then shift to [0, 1]
        score = (ref_label * (2.0 * sim_contact - 1.0) + 1.0) / 2.0
        mask = (ref_label != 0).float()
        return (score * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
