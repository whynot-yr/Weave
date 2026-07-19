"""Goal-conditioned motion command for the DAgger distillation stage."""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass

from g1_hoi_learning.tasks.hoi.mdp.commands import MotionCommand, MotionCommandCfg

from .utils import goal_channel


class GoalMotionCommand(MotionCommand):
    """``MotionCommand`` + a sparse root/object-pose goal (target frame ``j``) with per-entity masking."""

    cfg: "GoalMotionCommandCfg"

    def __init__(self, cfg: "GoalMotionCommandCfg", env):
        super().__init__(cfg, env)
        self.goal_step = torch.zeros_like(self.time_steps)                  # target frame j, per env
        self.gap = (int(cfg.gap[0]), None if cfg.gap[1] is None else int(cfg.gap[1]))
        self.goal_mask = torch.ones(self.num_envs, len(goal_channel.channels), device=self.device)
        self.mask_prob = torch.tensor(goal_channel.probs, device=self.device)  # per-channel reveal prob
        self._refresh_goal()

    def _sample_goal(self, env_ids: torch.Tensor, ref_step: torch.Tensor) -> None:
        """``goal_step ~ U[ref_step + gap[0], hi]``; ``gap[1]=None`` makes ``hi`` the clip end."""
        clip_len = self.motion.clip_lengths[self.env_clip[env_ids]]
        hi = clip_len - 1 if self.gap[1] is None else torch.minimum(ref_step + self.gap[1], clip_len - 1)
        lo = torch.minimum(ref_step + self.gap[0], hi)
        d = (torch.rand(len(env_ids), device=self.device) * (hi - lo + 1).float()).long()
        self.goal_step[env_ids] = torch.minimum(lo + d, hi)

    def _sample_goal_mask(self, env_ids: torch.Tensor) -> None:
        """Reveal each goal channel with prob ``goal_mask_prob``; an all-masked draw becomes all-revealed."""
        k = self.goal_mask.shape[-1]
        m = (torch.rand(len(env_ids), k, device=self.device) < self.mask_prob).float()
        m[m.sum(-1) == 0] = 1.0
        self.goal_mask[env_ids] = m

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        if len(env_ids) == 0:
            return
        self._sample_goal(env_ids, self.time_steps[env_ids])
        self._sample_goal_mask(env_ids)

    def _update_command(self):
        super()._update_command()
        reached = torch.where(self.time_steps >= self.goal_step)[0]        # keep the goal ahead of the reference
        if len(reached) > 0:
            self._sample_goal(reached, self.time_steps[reached])
        resample = torch.where(torch.rand(self.num_envs, device=self.device) < self.cfg.mask_resample_prob)[0]
        if len(resample) > 0:
            self._sample_goal_mask(resample)                              # occasional within-episode mask change
        self._refresh_goal()

    def _refresh_goal(self):
        self._goal_frame = self.motion.frames(self.env_clip, self.goal_step)

    # goal-frame object pose
    @property
    def goal_obj_pos_w(self) -> torch.Tensor:
        return self._goal_frame.object_pos_w + self._env.scene.env_origins

    @property
    def goal_obj_quat_w(self) -> torch.Tensor:
        return self._goal_frame.object_quat_w

    # goal-frame root/anchor pose
    @property
    def goal_root_pos_w(self) -> torch.Tensor:
        return self._goal_frame.body_pos_w[:, self.anchor_index] + self._env.scene.env_origins

    @property
    def goal_root_quat_w(self) -> torch.Tensor:
        return self._goal_frame.body_quat_w[:, self.anchor_index]

    @property
    def goal_body_pos_w(self) -> torch.Tensor:
        return self._goal_frame.body_pos_w + self._env.scene.env_origins[:, None, :]

    @property
    def goal_body_quat_w(self) -> torch.Tensor:
        return self._goal_frame.body_quat_w

    @property
    def goal_contact_label(self) -> torch.Tensor:
        return self._goal_frame.contact_label

    # ---- per-goal debug viz (markers drawn only for currently-revealed channels) ----
    @staticmethod
    def _frame_vis(prim_path: str, scale: float) -> VisualizationMarkers:
        cfg = FRAME_MARKER_CFG.replace(prim_path=prim_path)
        cfg.markers["frame"].scale = (scale, scale, scale)
        return VisualizationMarkers(cfg)

    @staticmethod
    def _sphere_vis(prim_path: str, color: tuple, radius: float = 0.03) -> VisualizationMarkers:
        cfg = VisualizationMarkersCfg(
            prim_path=prim_path,
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=radius, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color)
                )
            },
        )
        return VisualizationMarkers(cfg)

    def _draw_goal(self, vis, channel: str, pos_w: torch.Tensor, quat_w: torch.Tensor | None = None):
        revealed = self.goal_mask[:, goal_channel.channels[channel]] > 0.5
        if not bool(revealed.any()):
            vis.set_visibility(False)
            return
        vis.set_visibility(True)
        p = pos_w[revealed].reshape(-1, 3)
        q = quat_w[revealed].reshape(-1, 4) if quat_w is not None else None
        vis.visualize(p, q)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "_goal_vis"):
                self._ee_vis_idx = self.robot.find_bodies(["left_wrist_yaw_link", "right_wrist_yaw_link"])[0]
                b = "/World/Visuals/Command/goal"
                self._goal_vis = {
                    "root_pose": self._frame_vis(f"{b}/root_pose", 0.2),
                    "ee_pose": self._frame_vis(f"{b}/ee_pose", 0.1),
                    "object_pose": self._frame_vis(f"{b}/object_pose", 0.2),
                    "obj_ori_traj": self._frame_vis(f"{b}/obj_ori_traj", 0.08),
                    "keypoint": self._sphere_vis(f"{b}/keypoint", (0.1, 0.9, 0.1)),
                    "root_wp": self._sphere_vis(f"{b}/root_wp", (1.0, 0.6, 0.0)),
                    "object_wp": self._sphere_vis(f"{b}/object_wp", (1.0, 0.0, 1.0)),
                    "body_traj": self._sphere_vis(f"{b}/body_traj", (0.5, 1.0, 0.5), radius=0.02),
                }
            for vis in self._goal_vis.values():
                vis.set_visibility(True)
        elif hasattr(self, "_goal_vis"):
            for vis in self._goal_vis.values():
                vis.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        bi, ee, v = self.body_indices, self._ee_vis_idx, self._goal_vis
        self._draw_goal(v["root_pose"], "goal_root_pos_b", self.goal_root_pos_w, self.goal_root_quat_w)
        self._draw_goal(v["ee_pose"], "goal_end_effector_pose_b", self.goal_body_pos_w[:, ee], self.goal_body_quat_w[:, ee])
        self._draw_goal(v["object_pose"], "goal_object_pos_b", self.goal_obj_pos_w, self.goal_obj_quat_w)
        self._draw_goal(v["obj_ori_traj"], "goal_obj_ori_traj_b", self.future_obj_pos_w, self.future_obj_quat_w)
        self._draw_goal(v["keypoint"], "goal_keypoint_pos_b", self.goal_body_pos_w[:, bi])
        self._draw_goal(v["root_wp"], "goal_root_waypoint_b", self.future_anchor_pos_w)
        self._draw_goal(v["object_wp"], "goal_object_waypoint_b", self.future_obj_pos_w)
        self._draw_goal(v["body_traj"], "goal_body_pos_traj_b", self.future_body_pos_w[:, :, bi])


@configclass
class GoalMotionCommandCfg(MotionCommandCfg):
    class_type: type = GoalMotionCommand

    gap: tuple[int, int | None] = (50, None)
    """Goal horizon [min, max] in frames; max=None => goal reaches the clip end."""

    mask_resample_prob: float = 0.02
    """Per-step probability to resample an env's goal mask."""
