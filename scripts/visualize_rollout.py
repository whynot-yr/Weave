#!/usr/bin/env python3
"""Export one collected rollout's head-camera RGB-D stream to MP4."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="rollout directory or rollouts.h5")
    parser.add_argument("--episode", type=int, default=0, help="zero-based episode index")
    parser.add_argument("--output", type=Path, required=True, help="output MP4 path")
    parser.add_argument("--depth-max-m", type=float, default=4.0, help="white depth in meters")
    args = parser.parse_args()
    if args.depth_max_m <= 0:
        parser.error("--depth-max-m must be positive")

    source = args.source / "rollouts.h5" if args.source.is_dir() else args.source
    with h5py.File(source, "r") as data:
        lengths = data["episodes/length"]
        if not 0 <= args.episode < len(lengths):
            parser.error(f"episode must be in [0, {len(lengths) - 1}]")
        length = int(lengths[args.episode])
        if length == 0:
            parser.error("selected episode has no frames")
        metadata = json.loads(data.attrs["metadata_json"])
        fps = float(metadata["fps"])
        rgb_data = data["camera/head/rgb"]
        depth_data = data["camera/head/depth_mm"]
        height, width = rgb_data.shape[2:4]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{width * 2}x{height}",
            "-framerate", str(fps), "-i", "-", "-an", "-c:v", "libx264",
            "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", str(args.output),
        ]
        with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
            assert process.stdin is not None
            for step in range(length):
                rgb = rgb_data[args.episode, step]
                depth = depth_data[args.episode, step]
                gray = np.clip(depth.astype(np.float32) * (255.0 / (args.depth_max_m * 1000)), 0, 255).astype(np.uint8)
                gray[depth == 0] = 0
                frame = np.concatenate((rgb, np.repeat(gray[:, :, None], 3, axis=2)), axis=1)
                process.stdin.write(frame.tobytes())
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError("ffmpeg failed to encode the rollout")
    print(f"Wrote {length} frames at {fps:g} fps to {args.output}")


if __name__ == "__main__":
    main()
