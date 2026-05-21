"""
Multi-motion command term: holds a buffer of N motions and tracks
which (motion_id, local_t) each env is currently at.

Memory layout::

    buffer (TensorClass, batch_size=[sum_T])
      motion 0 frames | motion 1 frames | ... | motion N-1 frames
      [0..T0-1]       | [T0..T0+T1-1]   |     | [...sum_T-1]

    motion_starts: (N+1,)  cumulative start offsets
    motion_lengths: (N,)   per-motion length

Each env carries (env_object_ids[i], time_steps[i]). Frame access is via
``buffer[motion_starts[env_object_ids] + time_steps]``.
"""

import os
from collections.abc import Sequence

import numpy as np
import torch
from tensordict import TensorClass

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_error_magnitude

from g1_hoi_learning.objects import ASSET_DIR


# ----------------------------------------------------------------------- TensorClass

class MotionData(TensorClass):
    """
    Per-frame motion state. 
    Used both as the flat buffer (batch_size=[sum_T])
    and as a per-env / per-env-per-offset gather (batch_size=[num_envs] or
    [num_envs, num_offsets]).
    """

    motion_id:        torch.Tensor   # (..., ) long  — which motion this frame belongs to
    step:             torch.Tensor   # (..., ) long  — local step within its motion
    joint_pos:        torch.Tensor   # (..., 53)
    joint_vel:        torch.Tensor   # (..., 53)
    body_pos_w:       torch.Tensor   # (..., 54, 3)
    body_quat_w:      torch.Tensor   # (..., 54, 4)
    body_lin_vel_w:   torch.Tensor   # (..., 54, 3)
    body_ang_vel_w:   torch.Tensor   # (..., 54, 3)
    object_pos_w:     torch.Tensor   # (..., 3)
    object_quat_w:    torch.Tensor   # (..., 4)
    object_lin_vel_w: torch.Tensor   # (..., 3)
    object_ang_vel_w: torch.Tensor   # (..., 3)
    contact_label:    torch.Tensor   # (..., 54)


# ----------------------------------------------------------------------- MotionLoader

class MotionLoader:
    def __init__(self, motion_files: list[str], device: str | torch.device):
        assert len(motion_files) > 0, "motion_files must be non-empty"
        for f in motion_files:
            assert os.path.isfile(f), f"motion file not found: {f}"

        per_object = [np.load(f, allow_pickle=True) for f in motion_files]
        self.num_objects = len(per_object)
        self.fps = int(per_object[0]["fps"][0])

        # ----- per-motion length and cumulative starts -----
        lengths = torch.tensor(
            [int(d["joint_pos"].shape[0]) for d in per_object],
            dtype=torch.long, device=device,
        )                                                       # (N,)
        starts = torch.cat([
            torch.zeros(1, dtype=torch.long, device=device),
            lengths.cumsum(0),
        ])                                                      # (N+1,)
        sum_T = int(starts[-1].item())
        self.motion_lengths = lengths
        self.motion_starts  = starts

        # ----- per-frame motion_id and step -----
        motion_id = torch.cat([
            torch.full((int(L.item()),), i, dtype=torch.long, device=device)
            for i, L in enumerate(lengths)
        ])
        step = torch.cat([
            torch.arange(int(L.item()), dtype=torch.long, device=device)
            for L in lengths
        ])

        # ----- flat-concat each per-frame field -----
        def _flat(key, dtype=torch.float32):
            return torch.cat(
                [torch.tensor(d[key], dtype=dtype, device=device) for d in per_object],
                dim=0,
            )

        self.buffer = MotionData(
            motion_id=motion_id,
            step=step,
            joint_pos        =_flat("joint_pos"),
            joint_vel        =_flat("joint_vel"),
            body_pos_w       =_flat("body_pos_w"),
            body_quat_w      =_flat("body_quat_w"),
            body_lin_vel_w   =_flat("body_lin_vel_w"),
            body_ang_vel_w   =_flat("body_ang_vel_w"),
            object_pos_w     =_flat("object_pos_w"),
            object_quat_w    =_flat("object_quat_w"),
            object_lin_vel_w =_flat("object_lin_vel_w"),
            object_ang_vel_w =_flat("object_ang_vel_w"),
            contact_label    =_flat("contact_label"),
            batch_size=[sum_T],
        )

        # ----- per-object surface points + names -----
        self.object_names = [str(d["object_name"]) for d in per_object]
        self.surface = torch.stack([
            torch.tensor(
                np.load(os.path.join(ASSET_DIR, name, "surface.npy")),
                dtype=torch.float32, device=device,
            )
            for name in self.object_names
        ])                                                      # (N, P, 3)

    def get_frames(self, motion_ids: torch.Tensor, local_t: torch.Tensor) -> MotionData:
        """Gather frames for arbitrary (motion_ids, local_t)."""
        global_idx = self.motion_starts[motion_ids] + local_t
        return self.buffer[global_idx]


# ----------------------------------------------------------------------- MotionCommand

class MotionCommand(CommandTerm):
    cfg: "MotionCommandCfg"

    def __init__(self, cfg: "MotionCommandCfg", env: ManagerBasedRLEnv):
        # Set up robot references before super().__init__ because it calls _set_debug_vis_impl
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.object: RigidObject = env.scene[cfg.object_name]
        self.anchor_index = self.robot.body_names.index(cfg.anchor_body_name)
        self.body_indices, self.body_names = self.robot.find_bodies(cfg.body_names, preserve_order=True)

        super().__init__(cfg, env)

        self.motion = MotionLoader(self.cfg.motion_files, device=self.device)

        # per-env state
        self.time_steps     = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.env_object_ids = (
            torch.arange(self.num_envs, device=self.device) % self.motion.num_objects
        )                                                       # round-robin assignment
        self.future_offsets = torch.tensor(cfg.future_offsets, dtype=torch.long, device=self.device)

        # caches refreshed in _update_command
        self._current_frame: MotionData | None = None
        self._future_frame: MotionData | None = None

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"]   = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"]   = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"]  = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"]  = torch.zeros(self.num_envs, device=self.device)

        # initial frame so properties are valid before the first _update_command
        self._refresh_caches()

    # ------------------------------------------------------------------- caches

    def _refresh_caches(self) -> None:
        self._current_frame = self.motion.get_frames(self.env_object_ids, self.time_steps)
        # future: per-env clamp at each env's own motion length
        future_t = self.time_steps[:, None] + self.future_offsets[None, :]    # (E, K)
        seq_end = self.motion.motion_lengths[self.env_object_ids][:, None]    # (E, 1)
        future_t = future_t.minimum(seq_end - 1)
        motion_ids = self.env_object_ids[:, None].expand_as(future_t)
        self._future_frame = self.motion.get_frames(motion_ids, future_t)

    # ------------------------------------------------------------------- command

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    # ------------------------------------------------------------------- motion-data properties (current frame)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._current_frame.joint_pos

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._current_frame.joint_vel

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._current_frame.body_pos_w + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._current_frame.body_quat_w

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._current_frame.body_lin_vel_w

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._current_frame.body_ang_vel_w

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self._current_frame.body_pos_w[:, self.anchor_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self._current_frame.body_quat_w[:, self.anchor_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self._current_frame.body_lin_vel_w[:, self.anchor_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self._current_frame.body_ang_vel_w[:, self.anchor_index]

    # ------------------------------------------------------------------- motion-data properties (future frames)

    @property
    def future_joint_pos(self) -> torch.Tensor:
        # (E, K, 53) — caller flattens last two dims if needed
        return self._future_frame.joint_pos

    @property
    def future_joint_vel(self) -> torch.Tensor:
        return self._future_frame.joint_vel

    @property
    def future_body_pos_w(self) -> torch.Tensor:
        return self._future_frame.body_pos_w + self._env.scene.env_origins[:, None, None, :]

    @property
    def future_body_quat_w(self) -> torch.Tensor:
        return self._future_frame.body_quat_w

    @property
    def future_anchor_pos_w(self) -> torch.Tensor:
        return self._future_frame.body_pos_w[:, :, self.anchor_index] + self._env.scene.env_origins[:, None, :]

    @property
    def future_anchor_quat_w(self) -> torch.Tensor:
        return self._future_frame.body_quat_w[:, :, self.anchor_index]

    # ------------------------------------------------------------------- reference object data (current + future)

    @property
    def ref_obj_pos_w(self) -> torch.Tensor:
        return self._current_frame.object_pos_w + self._env.scene.env_origins

    @property
    def ref_obj_quat_w(self) -> torch.Tensor:
        return self._current_frame.object_quat_w

    @property
    def ref_obj_lin_vel_w(self) -> torch.Tensor:
        return self._current_frame.object_lin_vel_w

    @property
    def ref_obj_ang_vel_w(self) -> torch.Tensor:
        return self._current_frame.object_ang_vel_w

    @property
    def future_obj_pos_w(self) -> torch.Tensor:
        return self._future_frame.object_pos_w + self._env.scene.env_origins[:, None, :]

    @property
    def future_obj_quat_w(self) -> torch.Tensor:
        return self._future_frame.object_quat_w

    # ------------------------------------------------------------------- contact labels

    @property
    def ref_contact_label(self) -> torch.Tensor:
        return self._current_frame.contact_label

    @property
    def future_contact_label(self) -> torch.Tensor:
        return self._future_frame.contact_label

    # ------------------------------------------------------------------- robot data passthrough

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.anchor_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.anchor_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.anchor_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.anchor_index]

    # ------------------------------------------------------------------- sim object passthrough

    @property
    def obj_pos_w(self) -> torch.Tensor:
        return self.object.data.root_pos_w

    @property
    def obj_quat_w(self) -> torch.Tensor:
        return self.object.data.root_quat_w

    @property
    def obj_lin_vel_w(self) -> torch.Tensor:
        return self.object.data.root_lin_vel_w

    @property
    def obj_ang_vel_w(self) -> torch.Tensor:
        return self.object.data.root_ang_vel_w

    # ------------------------------------------------------------------- CommandTerm interface

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_body_pos"] = torch.norm(
            self.body_pos_w[:, self.body_indices] - self.robot_body_pos_w[:, self.body_indices], dim=-1
        ).mean(dim=-1)
        self.metrics["error_body_rot"] = quat_error_magnitude(
            self.body_quat_w[:, self.body_indices], self.robot_body_quat_w[:, self.body_indices]
        ).mean(dim=-1)
        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        n = len(env_ids)

        # 1. sample new time_steps per env (per-env motion length)
        if self.cfg.rsi:
            T_per_env = self.motion.motion_lengths[self.env_object_ids[env_ids]]   # (n,)
            rand = torch.rand(n, device=self.device)
            self.time_steps[env_ids] = (rand * T_per_env.float()).long().clamp(min=0)
        else:
            self.time_steps[env_ids] = 0

        # 2. fetch fresh motion data for the reset envs (don't rely on cache yet)
        new_frames = self.motion.get_frames(self.env_object_ids[env_ids], self.time_steps[env_ids])

        # 3. root pose from motion's anchor pose
        root_pos      = new_frames.body_pos_w[:, self.anchor_index] + self._env.scene.env_origins[env_ids]
        root_ori      = new_frames.body_quat_w[:, self.anchor_index]
        root_lin_vel  = new_frames.body_lin_vel_w[:, self.anchor_index]
        root_ang_vel  = new_frames.body_ang_vel_w[:, self.anchor_index]

        # 4. joint positions from motion
        joint_pos = new_frames.joint_pos
        joint_vel = new_frames.joint_vel

        # 5. write to sim
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1),
            env_ids=env_ids,
        )
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # 6. reset object to its motion's reference (with env_origins offset)
        obj_pos = new_frames.object_pos_w + self._env.scene.env_origins[env_ids]
        obj_state = torch.cat([
            obj_pos,
            new_frames.object_quat_w,
            new_frames.object_lin_vel_w,
            new_frames.object_ang_vel_w,
        ], dim=-1)
        self.object.write_root_state_to_sim(obj_state, env_ids=env_ids)

    def _update_command(self):
        self.time_steps += 1
        seq_end = self.motion.motion_lengths[self.env_object_ids]    # (num_envs,)
        env_ids_to_reset = torch.where(self.time_steps >= seq_end)[0]
        self._resample_command(env_ids_to_reset)
        # refresh caches once for the rest of the step
        self._refresh_caches()

    # ------------------------------------------------------------------- debug viz

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_anchor_vis"):
                self.goal_anchor_vis = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/World/Visuals/Command/goal/anchor")
                )
                self.current_anchor_vis = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/World/Visuals/Command/current/anchor")
                )
                self.goal_body_vis = VisualizationMarkers(
                    self.cfg.body_visualizer_cfg.replace(prim_path="/World/Visuals/Command/goal/bodies")
                )
                self.current_body_vis = VisualizationMarkers(
                    self.cfg.body_visualizer_cfg.replace(prim_path="/World/Visuals/Command/current/bodies")
                )

            for vis in (self.goal_anchor_vis, self.current_anchor_vis, self.goal_body_vis, self.current_body_vis):
                vis.set_visibility(True)
        else:
            if hasattr(self, "goal_anchor_vis"):
                for vis in (self.goal_anchor_vis, self.current_anchor_vis, self.goal_body_vis, self.current_body_vis):
                    vis.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        self.goal_anchor_vis.visualize(self.anchor_pos_w, self.anchor_quat_w)
        self.current_anchor_vis.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        bi = self.body_indices
        self.goal_body_vis.visualize(
            self.body_pos_w[:, bi].reshape(-1, 3),
            self.body_quat_w[:, bi].reshape(-1, 4),
        )
        self.current_body_vis.visualize(
            self.robot_body_pos_w[:, bi].reshape(-1, 3),
            self.robot_body_quat_w[:, bi].reshape(-1, 4),
        )


# ----------------------------------------------------------------------- cfg

@configclass
class MotionCommandCfg(CommandTermCfg):
    class_type: type = MotionCommand
    asset_name: str = "robot"
    object_name: str = "object"

    anchor_body_name: str = "pelvis"
    body_names: list[str] = [
        # legs
        "left_hip_pitch_link", "left_knee_link", "left_ankle_roll_link",
        "right_hip_pitch_link", "right_knee_link", "right_ankle_roll_link",
        # torso
        "torso_link",
        # arms
        "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
        "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
    ]

    future_offsets: list[int] = [0, 1, 2, 4, 8]
    """Frame offsets for observation (0 = current frame)."""

    rsi: bool = True
    """Random State Initialization: start from random frame (training) or frame 0 (evaluation)."""

    motion_files: list[str] = []
    """List of motion npz files. N=1 is single-object training; N>1 is multi-object."""

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}
    joint_position_range: tuple[float, float] = (-0.1, 0.1)

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/World/Visuals/Command/anchor")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/World/Visuals/Command/bodies")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
