"""Episode-constant odometry noise for actor observations."""

from collections.abc import Sequence

import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils.math import (
    matrix_from_quat,
    quat_apply,
    quat_from_euler_xyz,
    quat_mul,
    subtract_frame_transforms,
)

from .commands import MotionCommand


def randomize_actor_odometry_bias(
    env: ManagerBasedEnv,
    env_ids: Sequence[int] | torch.Tensor | None,
    command_name: str,
    odom_pos_range: tuple[float, float, float],
    odom_rot_range: tuple[float, float, float],
) -> None:
    """Sample an episode-constant uniform odometry pose bias."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    count = len(env_ids)
    if count == 0:
        return

    def sample_uniform(range_values: tuple[float, float, float]) -> torch.Tensor:
        half_range = torch.tensor(range_values, device=env.device, dtype=torch.float32)
        return (2.0 * torch.rand(count, 3, device=env.device) - 1.0) * half_range

    odom_pos_error = sample_uniform(odom_pos_range)
    odom_rot_error = sample_uniform(odom_rot_range)
    command.odom_pos_error_b[env_ids] = odom_pos_error
    command.odom_quat_error[env_ids] = quat_from_euler_xyz(
        odom_rot_error[:, 0], odom_rot_error[:, 1], odom_rot_error[:, 2]
    )


def estimated_anchor_pose_w(command: MotionCommand) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the live robot anchor pose perturbed by the sampled odometry bias."""
    anchor_pos_w = command.robot_anchor_pos_w + quat_apply(command.robot_anchor_quat_w, command.odom_pos_error_b)
    anchor_quat_w = quat_mul(command.robot_anchor_quat_w, command.odom_quat_error)
    return anchor_pos_w, anchor_quat_w


def future_pose_in_estimated_frame(
    command: MotionCommand, target_pos_w: torch.Tensor, target_quat_w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Express a future target pose in the noisy odometry frame."""
    num_offsets = len(command.cfg.future_offsets)
    anchor_pos_w, anchor_quat_w = estimated_anchor_pose_w(command)
    return subtract_frame_transforms(
        anchor_pos_w[:, None, :].expand(-1, num_offsets, -1),
        anchor_quat_w[:, None, :].expand(-1, num_offsets, -1),
        target_pos_w,
        target_quat_w,
    )


def motion_future_anchor_pos_noisy_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    anchor_pos_b = future_pose_in_estimated_frame(
        command, command.future_anchor_pos_w, command.future_anchor_quat_w
    )[0]
    return anchor_pos_b.reshape(env.num_envs, -1)


def motion_future_anchor_ori_noisy_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    anchor_quat_b = future_pose_in_estimated_frame(
        command, command.future_anchor_pos_w, command.future_anchor_quat_w
    )[1]
    return matrix_from_quat(anchor_quat_b)[..., :2].reshape(env.num_envs, -1)


def motion_future_obj_pos_noisy_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    object_pos_b = future_pose_in_estimated_frame(
        command, command.future_obj_pos_w, command.future_obj_quat_w
    )[0]
    return object_pos_b.reshape(env.num_envs, -1)


def motion_future_obj_ori_noisy_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    object_quat_b = future_pose_in_estimated_frame(
        command, command.future_obj_pos_w, command.future_obj_quat_w
    )[1]
    return matrix_from_quat(object_quat_b)[..., :2].reshape(env.num_envs, -1)
