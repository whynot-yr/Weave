"""
Multi-clip motion command: holds a buffer of M reference CLIPS (trajectories) grouped by the
OBJECT each clip manipulates, and tracks which (clip, local_t) each env is currently at.

Memory layout::

    buffer (TensorClass, batch_size=[sum_T])
      clip 0 frames | clip 1 frames | ... | clip M-1 frames
      [0..T0-1]     | [T0..T0+T1-1] |     | [...sum_T-1]

    clip_starts:  (M+1,)  cumulative start offsets
    clip_lengths: (M,)    per-clip length
    clip_object:  (M,)    object index each clip manipulates

Each env owns a fixed object (``env_object``) and plays one clip of that object at a time
(``env_clip``, re-sampled every episode). 
Frame access is via ``buffer[clip_starts[env_clip] + time_steps]``.
"""

import os
from collections.abc import Sequence

import numpy as np
import torch
from tensordict import TensorClass

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_error_magnitude,
)

from g1_hoi_learning.objects import ASSET_DIR


# ----------------------------------------------------------------------- TensorClass

class MotionData(TensorClass):
    """
    Per-frame motion state. 
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

def read_motion_object_names(path: str) -> list[str]:
    """Per-clip object names for one packed motion file (data_replay_multiple.py).
    """
    return [str(x) for x in np.asarray(np.load(path, allow_pickle=True)["object_names"])]


class MotionLoader:
    """A library of reference motion CLIPS grouped by the OBJECT each clip manipulates.

    clip  : one reference trajectory (M total); all clips' frames live in one concatenated buffer.
    object: a distinct manipulable asset (O total); each clip belongs to one object, many clips may
            share an object. Surface points are stored once per object.
    """

    def __init__(self, motion_files: list[str], device: str | torch.device):
        assert len(motion_files) > 0, "motion_files must be non-empty"
        for f in motion_files:
            assert os.path.isfile(f), f"motion file not found: {f}"
        per_file = [np.load(f, allow_pickle=True) for f in motion_files]
        self.device = torch.device(device)
        self.fps = int(np.asarray(per_file[0]["fps"]).reshape(-1)[0])

        # ----- clips: per-clip length + object it manipulates -----
        clip_lengths: list[int] = []
        clip_obj_names: list[str] = []
        for d in per_file:
            clip_lengths.extend(int(x) for x in np.asarray(d["motion_lengths"]))
            clip_obj_names.extend(str(x) for x in np.asarray(d["object_names"]))
        self.num_clips = len(clip_lengths)
        self.clip_lengths = torch.tensor(clip_lengths, dtype=torch.long, device=device)         # (M,)
        self.clip_starts = torch.cat([
            torch.zeros(1, dtype=torch.long, device=device),
            self.clip_lengths.cumsum(0),
        ])                                                                                      # (M+1,)
        sum_T = int(self.clip_starts[-1].item())

        # ----- objects: unique names (stable order) + clip->object map + per-object clip table -----
        self.object_names = list(dict.fromkeys(clip_obj_names))                                 # (O,) first-seen
        self.num_objects = len(self.object_names)
        oid = {name: i for i, name in enumerate(self.object_names)}
        self.clip_object = torch.tensor([oid[n] for n in clip_obj_names], dtype=torch.long, device=device)  # (M,)
        self.object_nclips = torch.bincount(self.clip_object, minlength=self.num_objects)       # (O,)
        c_max = int(self.object_nclips.max())
        self._clips_by_object = torch.zeros((self.num_objects, c_max), dtype=torch.long, device=device)
        for o in range(self.num_objects):                       # O is small (#objects); runs once at init
            self._clips_by_object[o, : self.object_nclips[o]] = torch.where(self.clip_object == o)[0]

        # ----- frame buffer -----
        motion_id = torch.cat([
            torch.full((L,), i, dtype=torch.long, device=device) for i, L in enumerate(clip_lengths)
        ])
        step = torch.cat([torch.arange(L, dtype=torch.long, device=device) for L in clip_lengths])

        def _flat(key, dtype=torch.float32):
            return torch.cat(
                [torch.tensor(np.asarray(d[key]), dtype=dtype, device=device) for d in per_file],
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

        # ----- surface points, one set per object -----
        self.surface = torch.stack([
            torch.tensor(
                np.load(os.path.join(ASSET_DIR, name, "surface.npy")),
                dtype=torch.float32, device=device,
            )
            for name in self.object_names
        ])                                                                                      # (O, P, 3)

    @staticmethod
    def object_names_of(motion_files: list[str]) -> list[str]:
        """Unique object names across the files, in stable (file, then in-file) order."""
        names: list[str] = []
        for f in motion_files:
            for n in read_motion_object_names(f):
                if n not in names:
                    names.append(n)
        return names

    def frames(self, clip_ids: torch.Tensor, t: torch.Tensor) -> MotionData:
        """Gather frames at (clip, local time)."""
        return self.buffer[self.clip_starts[clip_ids] + t]

    def sample_clip(self, object_ids: torch.Tensor) -> torch.Tensor:
        """A uniformly random clip of each given object. (vectorized)"""
        counts = self.object_nclips[object_ids]
        j = (torch.rand(object_ids.shape[0], device=self.device) * counts.float()).long().minimum(counts - 1)
        return self._clips_by_object[object_ids, j]

    def first_clip(self, object_ids: torch.Tensor) -> torch.Tensor:
        """The first clip of each given object (deterministic; used for eval)."""
        return self._clips_by_object[object_ids, 0]

    def clip_at(self, object_ids: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """The (pos mod nclips)-th clip of each object — used to walk through all clips in eval."""
        return self._clips_by_object[object_ids, pos % self.object_nclips[object_ids]]

# ----------------------------------------------------------------------- MotionCommand

class MotionCommand(CommandTerm):
    cfg: "MotionCommandCfg"

    def __init__(self, cfg: "MotionCommandCfg", env: ManagerBasedRLEnv):
        if cfg.goal_sampling_mode not in ("terminal", "random_future"):
            raise ValueError(f"Invalid goal_sampling_mode: {cfg.goal_sampling_mode}")
        if cfg.eval_start_mode not in ("zero", "random"):
            raise ValueError(f"Invalid eval_start_mode: {cfg.eval_start_mode}")

        # Set up robot references before super().__init__ because it calls _set_debug_vis_impl
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.object: RigidObject = env.scene[cfg.object_name]
        self.anchor_index = self.robot.body_names.index(cfg.anchor_body_name)
        self.body_indices, self.body_names = self.robot.find_bodies(cfg.body_names, preserve_order=True)

        super().__init__(cfg, env)

        self.motion = MotionLoader(self.cfg.motion_files, device=self.device)

        # per-env state
        self.time_steps     = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.goal_time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # env -> object (fixed for the whole run; drives object spawn + surface)
        self.env_object     = torch.arange(self.num_envs, device=self.device) % self.motion.num_objects
        # env -> current clip within that object (re-sampled every episode); init to the object's first clip
        self.env_clip       = self.motion.first_clip(self.env_object)
        self._eval_clip_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.future_offsets = torch.tensor(cfg.future_offsets, dtype=torch.long, device=self.device)

        # Episode-constant odometry bias used only by noisy actor observations.
        self.odom_pos_error_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.odom_quat_error = torch.zeros(self.num_envs, 4, device=self.device)
        self.odom_quat_error[:, 0] = 1.0

        # clip-wise evaluation
        if self.cfg.eval_mode:
            if self.motion.num_objects != 1:
                raise ValueError(
                    f"eval_mode requires a single-object motion set, got {self.motion.num_objects} objects "
                )
            if self.num_envs != self.motion.num_clips:
                print(
                    f"[WARN] eval_mode: num_envs={self.num_envs} != num_clips={self.motion.num_clips};"
                )
            self._eval_env_clip = torch.arange(self.num_envs, device=self.device) % self.motion.num_clips
            self.env_clip = self._eval_env_clip.clone()

        # caches refreshed in _update_command
        self._current_frame: MotionData | None = None
        self._future_joint_pos: torch.Tensor
        self._future_anchor_pos: torch.Tensor
        self._future_anchor_quat: torch.Tensor
        self._future_obj_pos: torch.Tensor
        self._future_obj_quat: torch.Tensor
        self._future_contact_label: torch.Tensor

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"]   = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"]   = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"]  = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"]  = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_object_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_object_rot"] = torch.zeros(self.num_envs, device=self.device)

        # initial frame so properties are valid before the first _update_command
        self._resample_goal()
        self._refresh_caches()

    # ------------------------------------------------------------------- caches

    def _resample_goal(self, env_ids: Sequence[int] | None = None) -> None:
        """Select terminal or strictly future goal frames, then cache their poses."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        last = self.motion.clip_lengths[self.env_clip[env_ids]] - 1
        if self.cfg.goal_sampling_mode == "terminal":
            self.goal_time_steps[env_ids] = last
        else:
            # Uniform integer in [t + 1, last]; at the final frame, hold last.
            low = torch.minimum(self.time_steps[env_ids] + 1, last)
            count = last - low + 1
            offset = (torch.rand(env_ids.numel(), device=self.device) * count).long()
            self.goal_time_steps[env_ids] = low + offset

        self._refresh_goal(env_ids)

    def _refresh_goal(self, env_ids: Sequence[int] | None = None) -> None:
        """Cache the selected goal poses independently of the current reference frame."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        clips = self.env_clip[env_ids]
        indices = self.motion.clip_starts[clips] + self.goal_time_steps[env_ids]
        buffer = self.motion.buffer
        values = {
            "_goal_joint_pos": buffer.joint_pos[indices],
            "_goal_anchor_pos": buffer.body_pos_w[indices, self.anchor_index],
            "_goal_anchor_quat": buffer.body_quat_w[indices, self.anchor_index],
            "_goal_obj_pos": buffer.object_pos_w[indices],
            "_goal_obj_quat": buffer.object_quat_w[indices],
        }
        for name, value in values.items():
            if not hasattr(self, name):
                cache = torch.empty((self.num_envs, *value.shape[1:]), device=value.device, dtype=value.dtype)
                setattr(self, name, cache)
            getattr(self, name)[env_ids] = value

    def _refresh_caches(self) -> None:
        self._current_frame = self.motion.frames(self.env_clip, self.time_steps)
        # future: per-env clamp at each env's own clip length
        future_t = self.time_steps[:, None] + self.future_offsets[None, :]    # (E, K)
        clip_end = self.motion.clip_lengths[self.env_clip][:, None]           # (E, 1)
        future_t = future_t.minimum(clip_end - 1)
        frame_indices = self.motion.clip_starts[self.env_clip][:, None] + future_t
        buffer = self.motion.buffer
        self._future_joint_pos = buffer.joint_pos[frame_indices]
        self._future_anchor_pos = buffer.body_pos_w[frame_indices, self.anchor_index]
        self._future_anchor_quat = buffer.body_quat_w[frame_indices, self.anchor_index]
        self._future_obj_pos = buffer.object_pos_w[frame_indices]
        self._future_obj_quat = buffer.object_quat_w[frame_indices]
        self._future_contact_label = buffer.contact_label[frame_indices]

    # ------------------------------------------------------------------- command

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    # ------------------------------------------------------------------- goal poses

    @property
    def goal_joint_pos(self) -> torch.Tensor:
        return self._goal_joint_pos

    @property
    def goal_anchor_pos_w(self) -> torch.Tensor:
        return self._goal_anchor_pos + self._env.scene.env_origins

    @property
    def goal_anchor_quat_w(self) -> torch.Tensor:
        return self._goal_anchor_quat

    @property
    def goal_obj_pos_w(self) -> torch.Tensor:
        return self._goal_obj_pos + self._env.scene.env_origins

    @property
    def goal_obj_quat_w(self) -> torch.Tensor:
        return self._goal_obj_quat

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
        return self._future_joint_pos

    @property
    def future_anchor_pos_w(self) -> torch.Tensor:
        return self._future_anchor_pos + self._env.scene.env_origins[:, None, :]

    @property
    def future_anchor_quat_w(self) -> torch.Tensor:
        return self._future_anchor_quat

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
        return self._future_obj_pos + self._env.scene.env_origins[:, None, :]

    @property
    def future_obj_quat_w(self) -> torch.Tensor:
        return self._future_obj_quat

    # ------------------------------------------------------------------- contact labels

    @property
    def ref_contact_label(self) -> torch.Tensor:
        return self._current_frame.contact_label

    @property
    def future_contact_label(self) -> torch.Tensor:
        return self._future_contact_label

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

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        extras = super().reset(env_ids)
        self._refresh_caches()
        return extras

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
        self.metrics["error_object_pos"] = torch.norm(self.ref_obj_pos_w - self.obj_pos_w, dim=-1)
        self.metrics["error_object_rot"] = quat_error_magnitude(self.ref_obj_quat_w, self.obj_quat_w)

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        objects = self.env_object[env_ids]

        # 1. pick a clip and starting frame within each environment's fixed object
        if self.cfg.eval_mode:
            self.env_clip[env_ids] = self._eval_env_clip[env_ids]
            if self.cfg.eval_start_mode == "random":
                lengths = self.motion.clip_lengths[self.env_clip[env_ids]]
                # Exclude the terminal frame so a strictly future goal exists.
                # Single-frame clips retain their only frame as both start and goal.
                counts = (lengths - 1).clamp(min=1)
                self.time_steps[env_ids] = (torch.rand(len(env_ids), device=self.device) * counts).long()
            else:
                self.time_steps[env_ids] = 0
        elif self.cfg.rsi:
            n = len(env_ids)
            self.env_clip[env_ids] = self.motion.sample_clip(objects)
            T_per_env = self.motion.clip_lengths[self.env_clip[env_ids]].float() - 1   # (n,)
            self.time_steps[env_ids] = (torch.rand(n, device=self.device) * T_per_env).long().clamp(min=0)
        else:
            self.env_clip[env_ids] = self.motion.first_clip(objects)
            self.time_steps[env_ids] = 0

        # Also runs on clip rollover: select goals relative to the new clip/start.
        self._resample_goal(env_ids)

        # 2. fetch fresh motion data for the reset envs (don't rely on cache yet)
        new_frames = self.motion.frames(self.env_clip[env_ids], self.time_steps[env_ids])

        # 3. root pose from motion's anchor pose
        root_pos = new_frames.body_pos_w[:, self.anchor_index] + self._env.scene.env_origins[env_ids]
        root_ori = new_frames.body_quat_w[:, self.anchor_index]
        root_lin_vel = new_frames.body_lin_vel_w[:, self.anchor_index]
        root_ang_vel = new_frames.body_ang_vel_w[:, self.anchor_index]

        # 4. joint positions from motion
        joint_pos = new_frames.joint_pos
        joint_vel = new_frames.joint_vel

        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos = torch.clip(
            joint_pos,
            soft_joint_pos_limits[:, :, 0],
            soft_joint_pos_limits[:, :, 1],
        )

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

        # Explicit env.reset() computes observations without calling command.compute().
        # Refresh reference caches now, without advancing time_steps.
        self._refresh_caches()

    def _update_command(self):
        self.time_steps += 1
        clip_end = self.motion.clip_lengths[self.env_clip]    # (num_envs,)
        if self.cfg.eval_mode:
            torch.minimum(self.time_steps, clip_end - 1, out=self.time_steps)
        else:
            self._resample_command(torch.where(self.time_steps >= clip_end)[0])

        if self.cfg.goal_sampling_mode == "random_future":
            # Rollover may have changed clips and already sampled their new goals.
            last = self.motion.clip_lengths[self.env_clip] - 1
            reached = self.time_steps >= self.goal_time_steps
            can_advance = self.time_steps < last
            self._resample_goal(torch.where(reached & can_advance)[0])

        # refresh caches once for the rest of the step
        self._refresh_caches()

    # ------------------------------------------------------------------- debug viz

    def _set_debug_vis_impl(self, debug_vis: bool):
        # CommandTerm calls this before motion/goal caches have been initialized.
        if debug_vis and not hasattr(self, "_pose_visualizers"):
            ids = self.cfg.goal_vis_env_ids
            if ids is not None and any(i < 0 or i >= self.num_envs for i in ids):
                raise ValueError(f"goal_vis_env_ids must be in [0, {self.num_envs}), got {ids}")
            self._vis_env_ids = slice(None) if ids is None else torch.tensor(ids, device=self.device, dtype=torch.long)
            self._pose_visualizers = {}
            specs = {
                "reference_anchor": (self.cfg.anchor_visualizer_cfg, self.cfg.show_reference_bodies),
                "reference_bodies": (self.cfg.body_visualizer_cfg, self.cfg.show_reference_bodies),
                "current_bodies": (self.cfg.body_visualizer_cfg, self.cfg.show_reference_bodies),
                "current_anchor": (self.cfg.body_visualizer_cfg, True),
                "target_root": (self.cfg.anchor_visualizer_cfg, self.cfg.show_goal_poses),
                "target_object": (self.cfg.anchor_visualizer_cfg, self.cfg.show_goal_poses),
                "current_object": (self.cfg.body_visualizer_cfg, self.cfg.show_goal_poses),
            }
            for name, (cfg, enabled) in specs.items():
                if enabled:
                    self._pose_visualizers[name] = VisualizationMarkers(
                        cfg.replace(prim_path=f"/World/Visuals/Command/{name}")
                    )
            # Position dots distinguish the targets while preserving standard XYZ axis colors.
            if self.cfg.show_goal_poses:
                for name, color in (("target_root_dot", (0.0, 1.0, 1.0)),
                                    ("target_object_dot", (1.0, 0.0, 1.0))):
                    self._pose_visualizers[name] = VisualizationMarkers(VisualizationMarkersCfg(
                        prim_path=f"/World/Visuals/Command/{name}",
                        markers={"dot": sim_utils.SphereCfg(
                            radius=0.025,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                        )},
                    ))

        for vis in getattr(self, "_pose_visualizers", {}).values():
            vis.set_visibility(debug_vis and self.cfg.goal_vis_env_ids != [])

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized or not hasattr(self, "_goal_anchor_pos"):
            return
        if self.cfg.goal_vis_env_ids == []:
            return
        ids = self._vis_env_ids

        def draw(name, pos, quat):
            vis = self._pose_visualizers.get(name)
            if vis is not None:
                vis.visualize(pos[ids].reshape(-1, 3), quat[ids].reshape(-1, 4))

        draw("current_anchor", self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        if self.cfg.show_goal_poses:
            # These are the same sampled goals used by observations, already in world coordinates.
            draw("target_root", self.goal_anchor_pos_w, self.goal_anchor_quat_w)
            draw("target_object", self.goal_obj_pos_w, self.goal_obj_quat_w)
            draw("target_root_dot", self.goal_anchor_pos_w, self.goal_anchor_quat_w)
            draw("target_object_dot", self.goal_obj_pos_w, self.goal_obj_quat_w)
            draw("current_object", self.obj_pos_w, self.obj_quat_w)
        if self.cfg.show_reference_bodies:
            draw("reference_anchor", self.anchor_pos_w, self.anchor_quat_w)
            bi = self.body_indices
            draw("reference_bodies", self.body_pos_w[:, bi], self.body_quat_w[:, bi])
            draw("current_bodies", self.robot_body_pos_w[:, bi], self.robot_body_quat_w[:, bi])


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

    future_offsets: list[int] = [0, 5, 10, 15, 20]
    """Frame offsets for observation (0 = current frame)."""

    goal_sampling_mode: str = "terminal"
    """terminal: fixed final frame; random_future: resample future goals when reference time reaches them."""

    rsi: bool = True
    """Random State Initialization during training; evaluation uses eval_start_mode."""

    eval_mode: bool = False
    """Clip-wise evaluation. Requires a single-object motion set."""

    eval_start_mode: str = "zero"
    """Evaluation start: zero or uniformly random non-terminal frame (frame 0 for single-frame clips)."""

    motion_files: list[str] = []
    """List of motion npz files. N=1 is single-object training; N>1 is multi-object."""

    show_goal_poses: bool = False
    """Show sampled goal root/object frames and the actual object frame when debug_vis is enabled."""

    show_reference_bodies: bool = True
    """Show timeline reference anchor/bodies and actual bodies when debug_vis is enabled."""

    goal_vis_env_ids: list[int] | None = None
    """Environment subset for all command markers; None shows all, [] hides all. Does not affect evaluation."""

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/World/Visuals/Command/anchor")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/World/Visuals/Command/bodies")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
