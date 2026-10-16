"""Simulator-independent initial-state perturbations."""

import torch


def offset_object_pose(state: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Apply world x/y translation and world-z yaw to a wxyz root state.

    Rotate about the object's root position; retain height and world velocities.
    Neither input is modified, so reference motion tensors remain unchanged.
    Offsets have columns (x metres, y metres, yaw radians).
    """
    result = state.clone()
    result[:, :2] += offsets[:, :2]
    c, s = torch.cos(offsets[:, 2] / 2), torch.sin(offsets[:, 2] / 2)
    w, x, y, z = state[:, 3:7].unbind(-1)
    result[:, 3:7] = torch.stack((c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w), -1)
    return result
