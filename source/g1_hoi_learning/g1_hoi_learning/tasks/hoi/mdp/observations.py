from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse, subtract_frame_transforms, transform_points

from g1_hoi_learning.models.object_encoder import get_object_encoder

from .commands import MotionCommand


def motion_joint_pos(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Reference joint positions from motion data."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.joint_pos


def motion_joint_vel(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Reference joint velocities from motion data."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.joint_vel


def motion_future_joint_pos(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Future reference joint positions for configured offsets."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.future_joint_pos.reshape(env.num_envs, -1)


def motion_future_joint_vel(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Future reference joint velocities for configured offsets."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.future_joint_vel.reshape(env.num_envs, -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Goal anchor position in the robot's anchor (body) frame."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Goal anchor orientation in the robot's anchor frame (first 2 columns of rotation matrix)."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(env.num_envs, -1)


def motion_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Goal tracked-body positions in the robot's anchor frame."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indices = command.body_indices
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].expand(-1, len(body_indices), -1),
        command.robot_anchor_quat_w[:, None, :].expand(-1, len(body_indices), -1),
        command.body_pos_w[:, body_indices],
        command.body_quat_w[:, body_indices],
    )
    return pos_b.view(env.num_envs, -1)


def motion_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Goal tracked-body orientations in the robot's anchor frame."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indices = command.body_indices
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].expand(-1, len(body_indices), -1),
        command.robot_anchor_quat_w[:, None, :].expand(-1, len(body_indices), -1),
        command.body_pos_w[:, body_indices],
        command.body_quat_w[:, body_indices],
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(env.num_envs, -1)


def motion_future_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Future anchor positions in current robot anchor frame. (num_envs, num_offsets * 3)"""
    command: MotionCommand = env.command_manager.get_term(command_name)
    # future_anchor_pos_w: (num_envs, num_offsets, 3)
    n_offsets = len(command.cfg.future_offsets)
    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].expand(-1, n_offsets, -1),
        command.robot_anchor_quat_w[:, None, :].expand(-1, n_offsets, -1),
        command.future_anchor_pos_w,
        command.future_anchor_quat_w,
    )
    return pos.reshape(env.num_envs, -1)


def motion_future_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Future anchor orientations in current robot anchor frame. (num_envs, num_offsets * 6)"""
    command: MotionCommand = env.command_manager.get_term(command_name)
    n_offsets = len(command.cfg.future_offsets)
    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].expand(-1, n_offsets, -1),
        command.robot_anchor_quat_w[:, None, :].expand(-1, n_offsets, -1),
        command.future_anchor_pos_w,
        command.future_anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(env.num_envs, -1)


def motion_future_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Future tracked-body positions in current robot anchor frame. (num_envs, num_offsets * num_bodies * 3)"""
    command: MotionCommand = env.command_manager.get_term(command_name)
    bi = command.body_indices
    n_offsets = len(command.cfg.future_offsets)
    n_bodies = len(bi)
    # future_body_pos_w: (num_envs, num_offsets, num_all_bodies, 3) -> select tracked bodies
    future_pos = command.future_body_pos_w[:, :, bi]  # (num_envs, num_offsets, num_bodies, 3)
    future_quat = command.future_body_quat_w[:, :, bi]
    anchor_pos = command.robot_anchor_pos_w[:, None, None, :].expand(-1, n_offsets, n_bodies, -1)
    anchor_quat = command.robot_anchor_quat_w[:, None, None, :].expand(-1, n_offsets, n_bodies, -1)
    pos_b, _ = subtract_frame_transforms(anchor_pos, anchor_quat, future_pos, future_quat)
    return pos_b.reshape(env.num_envs, -1)


def motion_future_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Future tracked-body orientations in current robot anchor frame. (num_envs, num_offsets * num_bodies * 6)"""
    command: MotionCommand = env.command_manager.get_term(command_name)
    bi = command.body_indices
    n_offsets = len(command.cfg.future_offsets)
    n_bodies = len(bi)
    future_pos = command.future_body_pos_w[:, :, bi]
    future_quat = command.future_body_quat_w[:, :, bi]
    anchor_pos = command.robot_anchor_pos_w[:, None, None, :].expand(-1, n_offsets, n_bodies, -1)
    anchor_quat = command.robot_anchor_quat_w[:, None, None, :].expand(-1, n_offsets, n_bodies, -1)
    _, ori_b = subtract_frame_transforms(anchor_pos, anchor_quat, future_pos, future_quat)
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Robot tracked-body positions in the robot's anchor frame."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indices = command.body_indices
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].expand(-1, len(body_indices), -1),
        command.robot_anchor_quat_w[:, None, :].expand(-1, len(body_indices), -1),
        command.robot_body_pos_w[:, body_indices],
        command.robot_body_quat_w[:, body_indices],
    )
    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Robot tracked-body orientations in the robot's anchor frame."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indices = command.body_indices
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].expand(-1, len(body_indices), -1),
        command.robot_anchor_quat_w[:, None, :].expand(-1, len(body_indices), -1),
        command.robot_body_pos_w[:, body_indices],
        command.robot_body_quat_w[:, body_indices],
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(env.num_envs, -1)

# -- Object observations --


def object_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Current object position relative to robot anchor in anchor frame. (3 dims)"""
    command: MotionCommand = env.command_manager.get_term(command_name)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.obj_pos_w,
        command.obj_quat_w,
    )
    return pos_b.view(env.num_envs, -1)


def object_rot_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Current object orientation in anchor frame (6D tangent-normal). (6 dims)"""
    command: MotionCommand = env.command_manager.get_term(command_name)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.obj_pos_w,
        command.obj_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(env.num_envs, -1)


def motion_future_obj_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Future reference object positions in current robot anchor frame. (num_envs, num_offsets * 3)"""
    command: MotionCommand = env.command_manager.get_term(command_name)
    n_offsets = len(command.cfg.future_offsets)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].expand(-1, n_offsets, -1),
        command.robot_anchor_quat_w[:, None, :].expand(-1, n_offsets, -1),
        command.future_obj_pos_w,
        command.future_obj_quat_w,
    )
    return pos_b.reshape(env.num_envs, -1)


def object_point_cloud_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Frozen PointNet++ embedding of the object surface point cloud.
    Returns (num_envs, encoder.output_dim).
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    # per-env surface points: pick this env's object's surface from (N, P, 3)
    pts_local = command.motion.surface[command.env_object]    # (num_envs, P, 3) object frame
    # surface points: object frame -> world
    pts_w = transform_points(pts_local, pos=command.obj_pos_w, quat=command.obj_quat_w)  # (num_envs, P, 3)
    P = pts_w.shape[1]
    
    pts_b = quat_apply_inverse(
        command.robot_anchor_quat_w[:, None, :].expand(-1, P, -1),
        pts_w - command.robot_anchor_pos_w[:, None, :],
    )
    return get_object_encoder(pts_b.device)(pts_b)   # (num_envs, output_dim)

def object_nearest_point_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Per-body vector from each robot body to its nearest point on the object surface,
    expressed in the robot anchor frame. (num_envs, num_bodies * 3)
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    # per-env surface points: pick this env's object's surface from (N, P, 3)
    pts_local = command.motion.surface[command.env_object]    # (num_envs, P, 3) object frame
    # surface points: object frame -> world
    pts_w = transform_points(pts_local, pos=command.obj_pos_w, quat=command.obj_quat_w)  # (num_envs, P, 3)
    body_pos_w = command.robot_body_pos_w   # (num_envs, B, 3)
    B = body_pos_w.shape[1]
    # nearest point per body
    dist = torch.cdist(body_pos_w, pts_w)             # (num_envs, B, P)
    idx = dist.argmin(dim=-1)                          # (num_envs, B)
    nearest_w = pts_w.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))  # (num_envs, B, 3)
    # vector body -> nearest point, in world
    diff_w = nearest_w - body_pos_w
    # rotate into anchor frame
    diff_b = quat_apply_inverse(
        command.robot_anchor_quat_w[:, None, :].expand(-1, B, -1),
        diff_w,
    )
    return diff_b.reshape(env.num_envs, -1)


def motion_future_obj_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Future reference object orientations in current robot anchor frame. (num_envs, num_offsets * 6)"""
    command: MotionCommand = env.command_manager.get_term(command_name)
    n_offsets = len(command.cfg.future_offsets)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].expand(-1, n_offsets, -1),
        command.robot_anchor_quat_w[:, None, :].expand(-1, n_offsets, -1),
        command.future_obj_pos_w,
        command.future_obj_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(env.num_envs, -1)


# -- Contact observations --


def motion_contact_label(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Reference per-body contact labels for current timestep. (num_envs, num_bodies)"""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.ref_contact_label


def motion_future_contact_label(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Future reference contact labels. (num_envs, num_offsets * num_bodies)"""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.future_contact_label.reshape(env.num_envs, -1)


def contact(env: ManagerBasedEnv, sensor_name: str, threshold: float = 1.0) -> torch.Tensor:
    """Per-robot-body binary contact flag with object from force_matrix_w. (num_envs, 54)"""
    sensor = env.scene[sensor_name]
    forces = sensor.data.force_matrix_w[:, 0, :, :]  # (num_envs, 54, 3)
    return (forces.norm(dim=-1) > threshold).float()