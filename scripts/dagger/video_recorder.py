"""Isaac Lab lifecycle adapter for evaluation-only RGB recording."""

import numpy as np

from isaaclab.managers import DatasetExportMode, RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass


class EvaluationVideoTerm(RecorderTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if env.num_envs != 1:
            raise ValueError("Per-clip video recording requires exactly one environment")
        self.recorder = None
        env.eval_video_term = self

    def record_post_step(self):
        if self.recorder is not None:
            self.recorder.record_post_step(terminal=bool(self._env.reset_buf[0]))
        return None, None

    def close(self, file_path):
        if self.recorder is not None:
            self.recorder.finish(complete=False)


@configclass
class EvaluationVideoCfg(RecorderManagerBaseCfg):
    dataset_export_mode: DatasetExportMode = DatasetExportMode.EXPORT_NONE
    export_in_record_pre_reset: bool = False
    video: RecorderTermCfg = RecorderTermCfg(class_type=EvaluationVideoTerm)


def read_rgb(env):
    """Frame both the robot and object; render without advancing simulation physics."""
    robot = env.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
    obj = env.scene["object"].data.root_pos_w[0].detach().cpu().numpy()
    target = (robot + obj) / 2
    target[2] += 0.25
    # Increase viewing distance while approaching a distant object so neither is cropped.
    separation = float(np.linalg.norm((robot - obj)[:2]))
    scale = max(1.0, (separation + 3.0) / 4.0)
    eye = target + np.array([-4.0, -4.0, 2.5]) * scale
    env.sim.set_camera_view(eye=eye, target=target)
    env.sim.render()
    return env.render()


def warmup_rgb(env):
    """Create the RGB render product and warm it up without advancing physics."""
    for _ in range(12):
        frame = read_rgb(env)
        if frame is not None and frame.size and frame.any():
            return
        env.sim.render()
    raise RuntimeError("Evaluation RGB camera did not produce a nonblack frame after warmup")
