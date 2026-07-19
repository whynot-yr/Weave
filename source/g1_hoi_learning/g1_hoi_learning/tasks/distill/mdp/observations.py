"""Goal observations for the DAgger distillation.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse, quat_conjugate, quat_mul, subtract_frame_transforms

from g1_hoi_learning.tasks.hoi.mdp.observations import (
    motion_future_body_pos_b,
    motion_future_obj_ori_b,
)

from .commands import GoalMotionCommand
from .utils import goal_channel


@goal_channel
def goal_root_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Goal anchor position in the anchor frame."""
    command: GoalMotionCommand = env.command_manager.get_term(command_name)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w, command.robot_anchor_quat_w, command.goal_root_pos_w, command.goal_root_quat_w
    )
    return pos_b.view(env.num_envs, -1)


@goal_channel
def goal_root_rot_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Goal anchor orientation in the anchor frame."""
    command: GoalMotionCommand = env.command_manager.get_term(command_name)
    _, goal_b = subtract_frame_transforms(
        command.robot_anchor_pos_w, command.robot_anchor_quat_w, command.goal_root_pos_w, command.goal_root_quat_w
    )
    return matrix_from_quat(goal_b)[..., :2].reshape(env.num_envs, -1)


@goal_channel
def goal_root_waypoint_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Near-future root 2D path: xy of the future anchor at future_offsets, in the anchor frame."""
    command: GoalMotionCommand = env.command_manager.get_term(command_name)
    n = len(command.cfg.future_offsets)
    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].expand(-1, n, -1),
        command.robot_anchor_quat_w[:, None, :].expand(-1, n, -1),
        command.future_anchor_pos_w,
        command.future_anchor_quat_w,
    )
    return pos[..., :2].reshape(env.num_envs, -1)


@goal_channel
def goal_end_effector_pose_b(env: ManagerBasedEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Goal-frame end-effector pose (pos + 6D rot) in the current anchor frame; bodies from ``asset_cfg``."""
    command: GoalMotionCommand = env.command_manager.get_term(command_name)
    ids = asset_cfg.body_ids
    n = len(ids)
    pos_b, quat_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].expand(-1, n, -1),
        command.robot_anchor_quat_w[:, None, :].expand(-1, n, -1),
        command.goal_body_pos_w[:, ids],
        command.goal_body_quat_w[:, ids],
    )
    rot6 = matrix_from_quat(quat_b)[..., :2].reshape(env.num_envs, n, -1)
    return torch.cat([pos_b, rot6], dim=-1).reshape(env.num_envs, -1)


@goal_channel
def goal_keypoint_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Goal-frame keypoint (tracked-body) positions in the current anchor frame."""
    command: GoalMotionCommand = env.command_manager.get_term(command_name)
    n = len(command.body_indices)
    delta_w = command.goal_body_pos_w[:, command.body_indices] - command.robot_anchor_pos_w[:, None, :]
    pos_b = quat_apply_inverse(command.robot_anchor_quat_w[:, None, :].expand(-1, n, -1), delta_w)
    return pos_b.reshape(env.num_envs, -1)


@goal_channel
def goal_object_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Object position residual (current -> goal) in the anchor frame."""
    command: GoalMotionCommand = env.command_manager.get_term(command_name)
    dpos_b = quat_apply_inverse(command.robot_anchor_quat_w, command.goal_obj_pos_w - command.obj_pos_w)
    return dpos_b.view(env.num_envs, -1)


@goal_channel
def goal_object_rot_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Object orientation residual (current -> goal) in the anchor frame."""
    command: GoalMotionCommand = env.command_manager.get_term(command_name)
    _, obj_b = subtract_frame_transforms(
        command.robot_anchor_pos_w, command.robot_anchor_quat_w, command.obj_pos_w, command.obj_quat_w
    )
    _, goal_b = subtract_frame_transforms(
        command.robot_anchor_pos_w, command.robot_anchor_quat_w, command.goal_obj_pos_w, command.goal_obj_quat_w
    )
    dquat = quat_mul(goal_b, quat_conjugate(obj_b))
    return matrix_from_quat(dquat)[..., :2].reshape(env.num_envs, -1)


@goal_channel
def goal_object_waypoint_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Near-future object 3D path: object positions at future_offsets, in the current anchor frame."""
    command: GoalMotionCommand = env.command_manager.get_term(command_name)
    n = len(command.cfg.future_offsets)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].expand(-1, n, -1),
        command.robot_anchor_quat_w[:, None, :].expand(-1, n, -1),
        command.future_obj_pos_w,
        command.future_obj_quat_w,
    )
    return pos_b.reshape(env.num_envs, -1)


@goal_channel
def goal_contact_b(env: ManagerBasedEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Goal-frame contact labels (+1 touch / -1 no-touch / 0 neutral) for the given hand bodies."""
    command: GoalMotionCommand = env.command_manager.get_term(command_name)
    return command.goal_contact_label[:, asset_cfg.body_ids]


# -- Short-horizon reference trajectories (dense motion-future, low reveal prob) --
@goal_channel(prob=0.1)
def goal_body_pos_traj_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Short-horizon body-position trajectory (reference future tracked-body positions)."""
    return motion_future_body_pos_b(env, command_name)


@goal_channel(prob=0.1)
def goal_obj_ori_traj_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Short-horizon object-orientation trajectory (reference future object orientations)."""
    return motion_future_obj_ori_b(env, command_name)
