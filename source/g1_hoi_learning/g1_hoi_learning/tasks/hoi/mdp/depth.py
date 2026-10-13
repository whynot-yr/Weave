"""Head-camera observation for the visual student."""

import torch

from g1_hoi_learning.algorithms.vision import DEPTH_SIZE, preprocess_depth


def head_depth(env, sensor_name: str = "head_camera") -> torch.Tensor:
    depth = env.scene[sensor_name].data.output["depth"]
    if depth.shape[1:] != (DEPTH_SIZE, DEPTH_SIZE, 1):
        raise ValueError(f"Expected NHWC 128x128 image-plane depth, got {tuple(depth.shape)}")
    return preprocess_depth(depth.permute(0, 3, 1, 2))
