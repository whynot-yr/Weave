"""Goal-conditioned motion command for the DAgger distillation stage.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.markers import VisualizationMarkers
from isaaclab.utils import configclass

from g1_hoi_learning.tasks.hoi.mdp.commands import MotionCommand, MotionCommandCfg


class GoalMotionCommand(MotionCommand):
    """``MotionCommand`` + a sparse object-pose goal (target frame ``j``)."""

    cfg: "GoalMotionCommandCfg"

    def __init__(self, cfg: "GoalMotionCommandCfg", env):
        super().__init__(cfg, env)
        self.goal_step = torch.zeros_like(self.time_steps)  # target frame j, per env
        self.gap = tuple(int(x) for x in cfg.gap)  # (min, max) goal horizon in frames
        self._refresh_goal()

    # ----- goal sampling -----
    def _sample_goal(self, env_ids: torch.Tensor, ref_step: torch.Tensor) -> None:
        """Set ``goal_step = clamp(ref_step + d, clip_len - 1)`` with ``d ~ U[gap[0], gap[1]]`` per env."""
        d = torch.randint(self.gap[0], self.gap[1] + 1, (len(env_ids),), device=self.device)
        clip_len = self.motion.clip_lengths[self.env_clip[env_ids]]
        self.goal_step[env_ids] = (ref_step + d).clamp(max=clip_len - 1)

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)  # RSI: picks env_clip + start i, writes sim state
        if len(env_ids) == 0:
            return
        # goal frame j = i + d, from the RSI start frame i
        self._sample_goal(env_ids, self.time_steps[env_ids])

    def _update_command(self):
        super()._update_command()  # advance time_steps, reset expired envs (-> _resample_command), refresh ref caches
        # reach-resample: envs whose reference reached/passed the goal get a fresh future goal (keeps goal ahead)
        reached = torch.where(self.time_steps >= self.goal_step)[0]
        if len(reached) > 0:
            self._sample_goal(reached, self.time_steps[reached])
        self._refresh_goal()

    def _refresh_goal(self):
        self._goal_frame = self.motion.frames(self.env_clip, self.goal_step)

    # ----- goal-frame object pose (frame j) -----
    @property
    def goal_obj_pos_w(self) -> torch.Tensor:
        return self._goal_frame.object_pos_w + self._env.scene.env_origins

    @property
    def goal_obj_quat_w(self) -> torch.Tensor:
        return self._goal_frame.object_quat_w

    # ----- debug viz: current object pose + goal object pose -----
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_obj_vis"):
                self.current_obj_vis = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/World/Visuals/Command/current/object")
                )
                self.goal_obj_vis = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/World/Visuals/Command/goal/object")
                )
            self.current_obj_vis.set_visibility(True)
            self.goal_obj_vis.set_visibility(True)
        elif hasattr(self, "current_obj_vis"):
            self.current_obj_vis.set_visibility(False)
            self.goal_obj_vis.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        self.current_obj_vis.visualize(self.obj_pos_w, self.obj_quat_w)
        self.goal_obj_vis.visualize(self.goal_obj_pos_w, self.goal_obj_quat_w)


@configclass
class GoalMotionCommandCfg(MotionCommandCfg):
    class_type: type = GoalMotionCommand

    """Goal horizon range [min, max] in frames: 
    goal frame j = i + d, d ~ U[gap[0], gap[1]]
    (50 Hz -> [50, 200] = [1.0, 4.0] s)."""
    gap: tuple[int, int] = (50, 200)
