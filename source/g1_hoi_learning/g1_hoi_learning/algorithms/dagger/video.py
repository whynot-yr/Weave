"""Stream one evaluation clip to MP4 without retaining an RGB frame archive."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class VideoConfig:
    enabled: bool = True
    width: int = 1280
    height: int = 720
    frame_stride: int = 2

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("video.enabled must be a boolean")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in (self.width, self.height, self.frame_stride)
        ):
            raise ValueError("Video width, height, and frame_stride must be positive integers")
        if self.width % 2 or self.height % 2:
            raise ValueError("H.264 video width and height must be even")


def create_video_writer(path: Path, fps: float):
    import imageio.v2 as imageio

    return imageio.get_writer(
        str(path),
        format="FFMPEG",
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        quality=7,
        macro_block_size=1,
        ffmpeg_log_level="error",
        output_params=["-movflags", "+faststart"],
    )


class ClipVideoRecorder:
    def __init__(
        self,
        output_dir: Path,
        cfg: VideoConfig,
        step_dt: float,
        frame_source: Callable[[], np.ndarray],
        writer_factory=create_video_writer,
    ):
        if not np.isfinite(step_dt) or step_dt <= 0:
            raise ValueError("Video requires a positive finite simulation step_dt")
        self.output_dir, self.cfg, self.frame_source = Path(output_dir), cfg, frame_source
        self.fps = 1.0 / (step_dt * cfg.frame_stride)
        self.step_dt, self.writer_factory = step_dt, writer_factory
        self.writer = None
        self.steps = 0
        self.frame_steps: list[int] = []
        self.terminal_seen = False

    @property
    def active(self):
        return self.writer is not None

    def start(self, clip_id: int, motion_name: str, trial_id: int | None = None):
        if self.active:
            raise RuntimeError("Finish the previous video before starting another clip")
        # Clip id ensures uniqueness even for duplicate or sanitized motion names.
        safe_name = re.sub(r"[^\w.-]+", "_", str(motion_name)).strip("._")[:100] or "motion"
        directory = self.output_dir / "videos"
        directory.mkdir(parents=True, exist_ok=True)
        suffix = "" if trial_id is None else f"_trial_{trial_id:04d}"
        self.path = directory / f"clip_{clip_id:04d}_{safe_name}{suffix}.mp4"
        self.temporary = self.path.with_name(self.path.stem + ".partial.mp4")
        self.steps, self.frame_steps, self.terminal_seen = 0, [], False
        first_frame = self._read_frame()
        self.writer = self.writer_factory(self.temporary, self.fps)
        self.writer.append_data(first_frame)
        self.frame_steps.append(0)

    def _read_frame(self):
        frame = self.frame_source()
        if (
            not isinstance(frame, np.ndarray)
            or frame.dtype != np.uint8
            or frame.shape != (self.cfg.height, self.cfg.width, 3)
        ):
            raise ValueError("RGB frame must be uint8 [video.height, video.width, 3]")
        # Render buffers can be RGBA views or overwritten on the next step.
        return np.ascontiguousarray(frame)

    def _capture(self):
        self.writer.append_data(self._read_frame())
        self.frame_steps.append(self.steps)

    def record_post_step(self, terminal: bool):
        """Called inside env.step, after physics and BEFORE automatic reset."""
        if not self.active or self.terminal_seen:
            return
        self.steps += 1
        if terminal or self.steps % self.cfg.frame_stride == 0:
            self._capture()
        self.terminal_seen = terminal

    def finish(self, complete: bool) -> dict | None:
        if not self.active:
            return None
        writer, self.writer = self.writer, None
        writer.close()
        if not self.frame_steps or not self.temporary.is_file() or self.temporary.stat().st_size == 0:
            raise RuntimeError("Video encoder did not produce a nonempty MP4")
        self.temporary.replace(self.path)
        return {
            "video_path": str(self.path.relative_to(self.output_dir)),
            "video_frames": len(self.frame_steps),
            "video_fps": self.fps,
            "video_complete": bool(complete and self.terminal_seen),
            "video_frame_steps": self.frame_steps.copy(),
            "video_simulation_seconds": self.steps * self.step_dt,
        }
