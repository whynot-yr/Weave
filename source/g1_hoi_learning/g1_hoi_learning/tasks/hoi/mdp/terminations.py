from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv as _ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg
from isaaclab.sensors import ContactSensor

from .commands import MotionCommand


def bad_anchor_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold


def bad_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    command: MotionCommand = env.command_manager.get_term(command_name)

    motion_projected_gravity_b = math_utils.quat_apply_inverse(
        command.anchor_quat_w, asset.data.GRAVITY_VEC_W
    )
    robot_projected_gravity_b = math_utils.quat_apply_inverse(
        command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W
    )

    return (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold


def bad_motion_body_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indices = command.robot.find_bodies(body_names, preserve_order=True)[0]
    error = torch.norm(
        command.body_pos_w[:, body_indices] - command.robot_body_pos_w[:, body_indices],
        dim=-1,
    )
    return torch.any(error > threshold, dim=-1)


def bad_object_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(command.ref_obj_pos_w - command.obj_pos_w, dim=-1) > threshold


def bad_object_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    command: MotionCommand = env.command_manager.get_term(command_name)

    ref_projected_gravity_b = math_utils.quat_apply_inverse(
        command.ref_obj_quat_w, asset.data.GRAVITY_VEC_W
    )
    obj_projected_gravity_b = math_utils.quat_apply_inverse(
        command.obj_quat_w, asset.data.GRAVITY_VEC_W
    )

    return (ref_projected_gravity_b[:, 2] - obj_projected_gravity_b[:, 2]).abs() > threshold


def motion_clip_end(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Reference clip exhausted. Register with ``time_out=True`` -- reaching the last frame is a success.
    Not part of TerminationsCfg: the eval script adds it, training rolls over inside the command instead."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.time_steps >= command.motion.clip_lengths[command.env_clip] - 1


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indices = command.robot.find_bodies(body_names, preserve_order=True)[0]
    error = torch.abs(
        command.body_pos_w[:, body_indices, -1] - command.robot_body_pos_w[:, body_indices, -1]
    )
    return torch.any(error > threshold, dim=-1)


class bad_contact(ManagerTermBase):
    """Terminate when all hand bodies lose object contact for N consecutive frames.

    Uses force_matrix_w from object contact sensor (on Object, filtered against robot bodies).
    The filter dimension is in robot.body_names order.
    """

    def __init__(self, cfg: TerminationTermCfg, env: _ManagerBasedRLEnv):
        super().__init__(cfg, env)
        robot: Articulation = env.scene["robot"]
        self.hand_idx = robot.find_bodies(cfg.params["hand_body_names"])[0]
        self.lost_counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            self.lost_counter[:] = 0
        else:
            self.lost_counter[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        sensor_name: str,
        hand_body_names: list[str],
        max_lost_frames: int,
        threshold: float = 1.0,
    ) -> torch.Tensor:
        command: MotionCommand = env.command_manager.get_term(command_name)
        sensor: ContactSensor = env.scene[sensor_name]

        # Reference: should any hand body have contact?
        ref_label = command.ref_contact_label[:, self.hand_idx]  # (num_envs, num_hand)
        any_expected = (ref_label > 0).any(dim=-1)  # (num_envs,)

        # Sim: does any hand body have object contact?
        forces = sensor.data.force_matrix_w[:, 0, self.hand_idx, :]  # (num_envs, num_hand, 3)
        any_contact = (forces.norm(dim=-1) > threshold).any(dim=-1)  # (num_envs,)

        # Increment if expected but no contact, reset otherwise
        lost = any_expected & ~any_contact
        self.lost_counter[lost] += 1
        self.lost_counter[~lost] = 0

        return self.lost_counter >= max_lost_frames


class goal_success(ManagerTermBase):
    """Check object goals once per post-physics step, before automatic reset.

    Snapshots deliberately survive reset so the evaluator can read terminal states.
    """

    def __init__(self, cfg: TerminationTermCfg, env: _ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.hold = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.elapsed = torch.zeros_like(self.hold)
        self.history = [[] for _ in range(env.num_envs)]
        self.snapshot = {}
        self.snapshot_history = list(self.history)

    def reset(self, env_ids: Sequence[int] | None = None):
        ids = list(range(len(self.history))) if env_ids is None else env_ids
        if isinstance(ids, slice):
            ids = list(range(len(self.history)))[ids]
        for i in ids:
            self.history[int(i)] = []
        self.hold[ids] = 0
        self.elapsed[ids] = 0

    def __call__(self, env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
        import math

        command: MotionCommand = env.command_manager.get_term(command_name)
        cfg = command.cfg
        self.elapsed += 1
        pos_error = torch.linalg.vector_norm(command.obj_pos_w - command.goal_obj_pos_w, dim=-1)
        rot_error = math_utils.quat_error_magnitude(command.obj_quat_w, command.goal_obj_quat_w)
        inside = (pos_error < cfg.goal_position_threshold) & (rot_error < math.radians(cfg.goal_rotation_threshold_deg))
        self.hold = torch.where(inside, self.hold + 1, 0)
        reached = self.hold >= math.ceil(cfg.goal_hold_seconds / env.step_dt)
        command.goal_reached.copy_(reached)
        last = command.motion.clip_lengths[command.env_clip] - 1
        success = reached & (command.goal_time_steps == last)
        for i in reached.nonzero(as_tuple=False).flatten().tolist():
            self.history[i] = [*self.history[i], {
                "frame": int(command.goal_time_steps[i]),
                "elapsed_seconds": float(self.elapsed[i]) * env.step_dt,
            }]
        self.hold[reached] = 0
        final = command.motion.frames(command.env_clip, last)
        self.snapshot = {
            "elapsed_seconds": self.elapsed.clone().float() * env.step_dt,
            "current_goal_frame": command.goal_time_steps.clone(),
            "current_goal_reached": reached.clone(),
            "current_goal_position_error": pos_error.clone(),
            "current_goal_rotation_error": rot_error.clone(),
            "final_object_position_error": torch.linalg.vector_norm(
                command.obj_pos_w - (final.object_pos_w + env.scene.env_origins), dim=-1
            ),
            "final_object_rotation_error": math_utils.quat_error_magnitude(
                command.obj_quat_w, final.object_quat_w
            ),
            "final_goal_reached": success.clone(),
        }
        self.snapshot_history = list(self.history)
        return success


def robot_fallen(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Absolute root height and tilt checks, independent of the reference motion."""
    import math

    command: MotionCommand = env.command_manager.get_term(command_name)
    gravity = torch.zeros_like(command.robot_anchor_pos_w)
    gravity[:, 2] = -1
    projected = math_utils.quat_apply_inverse(command.robot_anchor_quat_w, gravity)
    height = command.robot_anchor_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return (height < command.cfg.fallen_height_threshold) | (
        -projected[:, 2] < math.cos(math.radians(command.cfg.fallen_tilt_threshold_deg))
    )
