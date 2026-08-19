from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import (
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_conjugate,
    quat_mul,
    subtract_frame_transforms,
)

from g1_hoi_learning.objects import ASSET_DIR

from .commands import MotionCommand
from .geometry import SampledSurfaceQuery, load_bps_sdf_assets, sample_sdf_trilinear


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


class ObjectBpsSdf(ManagerTermBase):
    """Signed distances from a fixed 128-point BPS basis to the normalized object mesh."""

    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.command: MotionCommand = env.command_manager.get_term(cfg.params["command_name"])
        self.basis, self.sdf_grids, self.grid_bound = load_bps_sdf_assets(
            ASSET_DIR,
            self.command.motion.object_names,
            self.command.motion.device,
        )

    def __call__(self, env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
        del env, command_name
        command = self.command
        num_basis = self.basis.shape[0]
        object_from_anchor = quat_mul(quat_conjugate(command.obj_quat_w), command.robot_anchor_quat_w)
        query_points_o = quat_apply(
            object_from_anchor[:, None, :].expand(-1, num_basis, -1),
            self.basis[None, :, :].expand(command.num_envs, -1, -1),
        )
        return sample_sdf_trilinear(
            self.sdf_grids,
            command.env_object,
            query_points_o,
            self.grid_bound,
        )


class object_nearest_point_b(ManagerTermBase):
    """Per-body vector from a robot body to its nearest point on the object surface, expressed in
    the robot anchor frame. (num_envs, len(body_names) * 3)
    """

    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.command: MotionCommand = env.command_manager.get_term(cfg.params["command_name"])
        self.body_ids, self.body_names = env.scene["robot"].find_bodies(cfg.params["body_names"])
        self.surface_query = SampledSurfaceQuery(self.command.motion.surface, self.command.env_object)
        self.nearest_o = torch.empty(
            (env.num_envs, len(self.body_ids), 3),
            device=env.device,
            dtype=self.command.motion.surface.dtype,
        )

    def __call__(self, env: ManagerBasedEnv, command_name: str, body_names: list[str]) -> torch.Tensor:
        del command_name, body_names
        command = self.command
        body_pos_w = command.robot_body_pos_w[:, self.body_ids]                # (num_envs, B, 3)
        num_bodies = body_pos_w.shape[1]

        body_pos_o = quat_apply_inverse(
            command.obj_quat_w[:, None, :].expand(-1, num_bodies, -1),
            body_pos_w - command.obj_pos_w[:, None, :],
        )
        nearest_o = self.surface_query(body_pos_o, self.nearest_o)
        diff_o = nearest_o - body_pos_o

        diff_w = quat_apply(command.obj_quat_w[:, None, :].expand(-1, num_bodies, -1), diff_o)
        diff_b = quat_apply_inverse(
            command.robot_anchor_quat_w[:, None, :].expand(-1, num_bodies, -1),
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
