from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, quat_error_magnitude, transform_points

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
    """Tracking error for hand body positions expressed in the object's local frame."""

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

        # ref: hand positions in the reference object frame
        ref_diff_w = command.body_pos_w[:, self.hand_idx] - command.ref_obj_pos_w[:, None, :]
        ref_pos_in_obj = quat_apply_inverse(
            command.ref_obj_quat_w[:, None, :].expand(-1, n, -1),
            ref_diff_w,
        )

        # sim: hand positions in the sim object frame
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


# -- Reference-free grasp shape --
class hand_opposition_reward(ManagerTermBase):
    """Reward the thumb for opposing the other fingers across the object surface. Range [0, 1].
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        robot: Articulation = env.scene["robot"]
        thumb, fingers = [], []
        for side in ("L", "R"):
            thumb.append(robot.body_names.index(f"{side}_{cfg.params['thumb_body_name']}"))
            fingers.append([robot.body_names.index(f"{side}_{n}") for n in cfg.params["finger_body_names"]])
        self.thumb_idx = torch.tensor(thumb, device=env.device)                # (H,)
        self.finger_idx = torch.tensor(fingers, device=env.device)             # (H, F)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        thumb_body_name: str,
        finger_body_names: list[str],
        num_nearest: int = 1,
    ) -> torch.Tensor:
        command: MotionCommand = env.command_manager.get_term(command_name)

        # object surface points of each env's own object, in world
        pts_local = command.motion.surface[command.env_object]                 # (N, P, 3)
        pts_w = transform_points(pts_local, pos=command.obj_pos_w, quat=command.obj_quat_w)

        H, F = self.finger_idx.shape
        tips = torch.cat([self.thumb_idx[:, None], self.finger_idx], dim=-1).reshape(-1)   # (H*(F+1),)
        tip_pos_w = command.robot_body_pos_w[:, tips]                          # (N, H*(F+1), 3)

        # nearest surface point per fingertip; averaging the k nearest smooths the argmin's jump
        # between neighbouring samples once a fingertip is close to the surface
        dist = torch.cdist(tip_pos_w, pts_w)                                   # (N, H*(F+1), P)
        k = max(1, min(num_nearest, pts_w.shape[1]))
        idx = dist.topk(k, dim=-1, largest=False).indices                      # (N, H*(F+1), k)
        nearest_w = pts_w[torch.arange(pts_w.shape[0], device=idx.device)[:, None, None], idx].mean(-2)

        # unit bearing vectors: object surface -> fingertip
        bearing = tip_pos_w - nearest_w
        u = bearing / bearing.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        u = u.reshape(-1, H, F + 1, 3)
        u_thumb, u_fingers = u[:, :, :1], u[:, :, 1:]                          # (N, H, 1, 3), (N, H, F, 3)

        # 1 when the thumb opposes the finger across the object, 0 when both sit on the same side
        opposition = (1.0 - (u_thumb * u_fingers).sum(-1)) / 2.0               # (N, H, F)

        # score a hand only while the reference has any of its fingertips in contact
        tips_by_hand = torch.cat([self.thumb_idx[:, None], self.finger_idx], dim=-1)   # (H, F+1)
        gate = (command.ref_contact_label[:, tips_by_hand] > 0).any(-1).float()        # (N, H)
        return (opposition.mean(-1) * gate).sum(-1) / gate.sum(-1).clamp(min=1.0)      # (N,)


# -- Feet slip penalty --
def feet_slide(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize horizontal foot velocity while the foot is in contact with the ground.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]     # (N, H, F, 3)
    in_contact = forces.norm(dim=-1).max(dim=1)[0] > force_threshold                  # (N, F)

    asset: Articulation = env.scene[asset_cfg.name]
    foot_vel_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]                # (N, F, 2)

    return (foot_vel_xy.norm(dim=-1) * in_contact).sum(dim=1)                          # (N,)
