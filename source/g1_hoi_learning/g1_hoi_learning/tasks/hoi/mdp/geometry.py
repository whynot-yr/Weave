from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import warp as wp

BPS_COUNT = 128
BPS_FILENAME = f"geometry/bps_{BPS_COUNT}.npy"
SDF_FILENAME = "sdf_128.npz"
SDF_GRID_BOUND = 1.25


@wp.kernel
def sampled_surface_nearest_kernel(
    surface_points: wp.array(dtype=wp.vec3),
    object_ids: wp.array(dtype=wp.int64),
    query_points: wp.array(dtype=wp.vec3),
    num_queries: int,
    num_surface_points: int,
    nearest_points: wp.array(dtype=wp.vec3),
):
    query_index = wp.tid()
    env_index = query_index // num_queries
    surface_offset = object_ids[env_index] * wp.int64(num_surface_points)
    query = query_points[query_index]

    nearest = surface_points[surface_offset]
    delta = query - nearest
    min_distance_squared = wp.dot(delta, delta)
    for surface_index in range(1, num_surface_points):
        candidate = surface_points[surface_offset + wp.int64(surface_index)]
        delta = query - candidate
        distance_squared = wp.dot(delta, delta)
        if distance_squared < min_distance_squared:
            min_distance_squared = distance_squared
            nearest = candidate
    nearest_points[query_index] = nearest


class SampledSurfaceQuery:
    """Fused nearest-neighbor queries against fixed per-object surface samples."""

    def __init__(self, surface_points: torch.Tensor, object_ids: torch.Tensor):
        wp.init()
        self.surface_points = surface_points.contiguous()
        self.object_ids = object_ids.contiguous()
        self.num_envs = object_ids.shape[0]
        self.num_surface_points = surface_points.shape[1]
        self.device = surface_points.device
        self.wp_device = wp.device_from_torch(self.device)
        self.surface_points_wp = wp.from_torch(self.surface_points.reshape(-1, 3), dtype=wp.vec3)
        self.object_ids_wp = wp.from_torch(self.object_ids)

    def __call__(self, query_points: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """Return the nearest sampled surface point for each object-frame query."""
        query_points = query_points.contiguous()
        num_queries = query_points.shape[1]
        query_points_wp = wp.from_torch(query_points.reshape(-1, 3), dtype=wp.vec3)
        output_wp = wp.from_torch(output.reshape(-1, 3), dtype=wp.vec3)
        stream = wp.stream_from_torch(self.device) if self.device.type == "cuda" else None
        wp.launch(
            sampled_surface_nearest_kernel,
            dim=self.num_envs * num_queries,
            inputs=[
                self.surface_points_wp,
                self.object_ids_wp,
                query_points_wp,
                num_queries,
                self.num_surface_points,
            ],
            outputs=[output_wp],
            device=self.wp_device,
            stream=stream,
            record_tape=False,
        )
        return output


def load_bps_sdf_assets(
    objects_dir: str | Path,
    object_names: Sequence[str],
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Load the shared BPS basis and ordered per-object SDF grids."""
    objects_dir = Path(objects_dir)
    basis = np.asarray(np.load(objects_dir / BPS_FILENAME), dtype=np.float32)

    sdf_grids: list[np.ndarray] = []
    for object_name in object_names:
        with np.load(objects_dir / object_name / SDF_FILENAME) as data:
            sdf_grids.append(np.asarray(data["sdf"], dtype=np.float32))

    basis_tensor = torch.as_tensor(basis, device=device).contiguous()
    sdf_tensor = torch.as_tensor(np.stack(sdf_grids), device=device).contiguous()
    return basis_tensor, sdf_tensor, SDF_GRID_BOUND


def sample_sdf_trilinear(
    sdf_grids: torch.Tensor,
    object_ids: torch.Tensor,
    query_points_o: torch.Tensor,
    grid_bound: float,
) -> torch.Tensor:
    """Sample per-object cubic SDF grids without expanding grids per environment.

    Args:
        sdf_grids: Signed-distance grids with shape ``(num_objects, resolution, resolution, resolution)``.
        object_ids: Object index for each environment with shape ``(num_envs,)``.
        query_points_o: Normalized object-frame query points with shape ``(num_envs, num_queries, 3)``.
        grid_bound: Symmetric grid bound. Grid coordinates span ``[-grid_bound, grid_bound]``.

    Returns:
        Interpolated signed distances with shape ``(num_envs, num_queries)``.

    Notes:
        Queries outside the grid are clamped to its boundary. The hot path gathers eight scalar values per
        query from a single flattened ``(object, x, y, z)`` table; it never materializes an
        ``(num_envs, resolution, resolution, resolution)`` tensor.
    """
    resolution = sdf_grids.shape[1]

    voxel = query_points_o.add(grid_bound).mul((resolution - 1) / (2.0 * grid_bound))
    voxel.clamp_(0.0, float(resolution - 1))
    lower = voxel.floor().to(torch.long)
    fraction = voxel - lower
    upper = (lower + 1).clamp_max_(resolution - 1)

    x0, y0, z0 = lower.unbind(dim=-1)
    x1, y1, z1 = upper.unbind(dim=-1)
    wx, wy, wz = fraction.unbind(dim=-1)

    grid_size = resolution * resolution * resolution
    object_base = object_ids.to(torch.long).unsqueeze(1) * grid_size
    sdf_flat = sdf_grids.reshape(-1)

    def gather(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        indices = object_base + x * (resolution * resolution) + y * resolution + z
        return sdf_flat[indices]

    c00 = torch.lerp(gather(x0, y0, z0), gather(x0, y0, z1), wz)
    c01 = torch.lerp(gather(x0, y1, z0), gather(x0, y1, z1), wz)
    c10 = torch.lerp(gather(x1, y0, z0), gather(x1, y0, z1), wz)
    c11 = torch.lerp(gather(x1, y1, z0), gather(x1, y1, z1), wz)
    c0 = torch.lerp(c00, c01, wy)
    c1 = torch.lerp(c10, c11, wy)
    return torch.lerp(c0, c1, wx)
