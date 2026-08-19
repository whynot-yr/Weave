#!/usr/bin/env python3
"""Precompute normalized signed-distance grids and a shared BPS basis for HOI objects."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh
import warp as wp

GENERATOR_VERSION = "g1-hoi-bps-sdf-v1"
DEFAULT_GRID_BOUND = 1.25
DEFAULT_RESOLUTION = 128
BPS_COUNT = 128


@wp.kernel
def bake_sdf_kernel(
    mesh_id: wp.uint64,
    start_index: int,
    resolution: int,
    grid_bound: float,
    max_distance: float,
    sdf: wp.array(dtype=float),
    failures: wp.array(dtype=int),
):
    """Evaluate exact triangle-mesh signed distances at a contiguous block of grid points."""
    local_index = wp.tid()
    linear_index = start_index + local_index
    plane_size = resolution * resolution
    x = linear_index // plane_size
    remainder = linear_index - x * plane_size
    y = remainder // resolution
    z = remainder - y * resolution
    spacing = 2.0 * grid_bound / float(resolution - 1)
    point = wp.vec3(
        -grid_bound + float(x) * spacing,
        -grid_bound + float(y) * spacing,
        -grid_bound + float(z) * spacing,
    )

    query = wp.mesh_query_point(mesh_id, point, max_distance)
    if query.result:
        nearest = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
        sdf[linear_index] = query.sign * wp.length(point - nearest)
    else:
        sdf[linear_index] = max_distance
        wp.atomic_add(failures, 0, 1)


def _repo_objects_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "source/g1_hoi_learning/g1_hoi_learning/objects"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _van_der_corput(indices: np.ndarray, base: int) -> np.ndarray:
    values = np.zeros(indices.shape, dtype=np.float64)
    denominator = 1.0
    work = indices.copy()
    while np.any(work):
        work, remainder = np.divmod(work, base)
        denominator *= base
        values += remainder / denominator
    return values


def _fibonacci_sphere(count: int, radius: float, phase: float) -> np.ndarray:
    indices = np.arange(count, dtype=np.float64)
    z = 1.0 - 2.0 * (indices + 0.5) / count
    theta = indices * (np.pi * (3.0 - np.sqrt(5.0))) + phase
    xy = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return radius * np.stack((xy * np.cos(theta), xy * np.sin(theta), z), axis=-1)


def make_bps_basis() -> np.ndarray:
    """Build 128 deterministic points spanning the normalized object volume and two shells."""
    indices = np.arange(1, 65, dtype=np.int64)
    u_radius = _van_der_corput(indices, 2)
    u_z = _van_der_corput(indices, 3)
    u_theta = _van_der_corput(indices, 5)
    radius = 0.9 * np.cbrt(u_radius)
    z = 1.0 - 2.0 * u_z
    xy = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    volume = radius[:, None] * np.stack(
        (xy * np.cos(2.0 * np.pi * u_theta), xy * np.sin(2.0 * np.pi * u_theta), z),
        axis=-1,
    )
    middle_shell = _fibonacci_sphere(32, radius=0.75, phase=0.37)
    outer_shell = _fibonacci_sphere(32, radius=1.10, phase=1.11)
    basis = np.concatenate((volume, middle_shell, outer_shell), axis=0).astype(np.float32)
    if basis.shape != (BPS_COUNT, 3):
        raise RuntimeError(f"Unexpected BPS shape: {basis.shape}")
    return basis


def ensure_bps_asset(objects_dir: Path, force: bool) -> Path:
    """Create or validate the shared deterministic BPS asset."""
    path = objects_dir / "geometry" / f"bps_{BPS_COUNT}.npy"
    expected = make_bps_basis()
    if path.exists() and not force:
        actual = np.asarray(np.load(path), dtype=np.float32)
        if not np.array_equal(actual, expected):
            raise RuntimeError(f"Existing BPS asset does not match {GENERATOR_VERSION}: {path}")
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, expected)
    return path


def _load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load_mesh(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.dump(concatenate=True)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Expected a triangle mesh in {path}, got {type(loaded).__name__}")
    if not loaded.is_watertight:
        raise ValueError(f"SDF sign requires a watertight mesh: {path}")
    if not loaded.is_winding_consistent:
        raise ValueError(f"SDF sign requires consistent face winding: {path}")
    return loaded


def _sample_numpy_sdf(sdf: np.ndarray, points: np.ndarray, grid_bound: float) -> np.ndarray:
    """Reference trilinear sampler used only for offline asset validation."""
    resolution = sdf.shape[0]
    voxel = np.clip((points + grid_bound) * ((resolution - 1) / (2.0 * grid_bound)), 0, resolution - 1)
    lower = np.floor(voxel).astype(np.int64)
    upper = np.minimum(lower + 1, resolution - 1)
    fraction = voxel - lower
    x0, y0, z0 = lower.T
    x1, y1, z1 = upper.T
    wx, wy, wz = fraction.T
    c00 = sdf[x0, y0, z0] * (1.0 - wz) + sdf[x0, y0, z1] * wz
    c01 = sdf[x0, y1, z0] * (1.0 - wz) + sdf[x0, y1, z1] * wz
    c10 = sdf[x1, y0, z0] * (1.0 - wz) + sdf[x1, y0, z1] * wz
    c11 = sdf[x1, y1, z0] * (1.0 - wz) + sdf[x1, y1, z1] * wz
    c0 = c00 * (1.0 - wy) + c01 * wy
    c1 = c10 * (1.0 - wy) + c11 * wy
    return c0 * (1.0 - wx) + c1 * wx


def bake_object_sdf(
    object_dir: Path,
    resolution: int,
    grid_bound: float,
    device: str,
    chunk_size: int,
    force: bool,
) -> dict[str, object]:
    """Bake one object's normalized SDF and write it beside the mesh."""
    object_name = object_dir.name
    obj_path = object_dir / f"{object_name}.obj"
    surface_path = object_dir / "surface.npy"
    output_path = object_dir / f"sdf_{resolution}.npz"
    if not obj_path.is_file() or not surface_path.is_file():
        raise FileNotFoundError(f"Missing OBJ or surface.npy in {object_dir}")
    if output_path.exists() and not force:
        with np.load(output_path) as data:
            valid = (
                str(data["object_name"].item()) == object_name
                and int(data["resolution"].item()) == resolution
                and np.isclose(float(data["grid_bound"].item()), grid_bound)
                and str(data["generator_version"].item()) == GENERATOR_VERSION
                and str(data["mesh_sha256"].item()) == _sha256(obj_path)
                and str(data["surface_sha256"].item()) == _sha256(surface_path)
            )
        if not valid:
            raise RuntimeError(f"Stale or incompatible SDF asset: {output_path}. Re-run with --force.")
        return {"object": object_name, "output": str(output_path), "status": "verified"}

    mesh = _load_mesh(obj_path)
    surface = np.asarray(np.load(surface_path), dtype=np.float32)
    center = surface.mean(axis=0, dtype=np.float64).astype(np.float32)
    radius = float(np.linalg.norm(surface - center, axis=-1).max())
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError(f"Invalid normalization radius for {object_name}: {radius}")
    vertices = ((np.asarray(mesh.vertices, dtype=np.float32) - center) / radius).astype(np.float32)
    max_vertex_radius = float(np.linalg.norm(vertices, axis=-1).max())
    if max_vertex_radius >= grid_bound:
        raise ValueError(
            f"Normalized {object_name} mesh reaches radius {max_vertex_radius:.6f}, outside grid bound {grid_bound}"
        )
    faces = np.asarray(mesh.faces, dtype=np.int32).reshape(-1)

    warp_mesh = wp.Mesh(
        points=wp.array(vertices, dtype=wp.vec3, device=device),
        indices=wp.array(faces, dtype=wp.int32, device=device),
    )
    num_voxels = resolution**3
    sdf_wp = wp.empty(num_voxels, dtype=float, device=device)
    failures_wp = wp.zeros(1, dtype=int, device=device)
    max_distance = 4.0 * grid_bound
    for start in range(0, num_voxels, chunk_size):
        count = min(chunk_size, num_voxels - start)
        wp.launch(
            bake_sdf_kernel,
            dim=count,
            inputs=[warp_mesh.id, start, resolution, grid_bound, max_distance],
            outputs=[sdf_wp, failures_wp],
            device=device,
        )
    wp.synchronize_device(device)

    failures = int(failures_wp.numpy()[0])
    if failures:
        raise RuntimeError(f"{failures} mesh queries failed while baking {object_name}")
    sdf = sdf_wp.numpy().reshape(resolution, resolution, resolution).astype(np.float32, copy=False)
    if not np.isfinite(sdf).all():
        raise RuntimeError(f"Non-finite SDF values generated for {object_name}")
    boundary = np.concatenate(
        (
            sdf[0].ravel(),
            sdf[-1].ravel(),
            sdf[:, 0].ravel(),
            sdf[:, -1].ravel(),
            sdf[:, :, 0].ravel(),
            sdf[:, :, -1].ravel(),
        )
    )
    if float(boundary.min()) < -1.0e-5:
        raise RuntimeError(f"Negative SDF found on the grid boundary for {object_name}: {boundary.min()}")

    normalized_surface = (surface - center) / radius
    surface_sdf = np.abs(_sample_numpy_sdf(sdf, normalized_surface, grid_bound))
    spacing = 2.0 * grid_bound / (resolution - 1)
    metadata = {
        "sdf": sdf,
        "center": center,
        "radius": np.float32(radius),
        "grid_bound": np.float32(grid_bound),
        "resolution": np.int32(resolution),
        "object_name": np.asarray(object_name),
        "mesh_sha256": np.asarray(_sha256(obj_path)),
        "surface_sha256": np.asarray(_sha256(surface_path)),
        "generator_version": np.asarray(GENERATOR_VERSION),
        "sign_convention": np.asarray("outside_positive_inside_negative"),
        "surface_abs_sdf_mean": np.float32(surface_sdf.mean()),
        "surface_abs_sdf_p95": np.float32(np.quantile(surface_sdf, 0.95)),
        "surface_abs_sdf_max": np.float32(surface_sdf.max()),
    }
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    with temporary_path.open("wb") as stream:
        np.savez_compressed(stream, **metadata)
    temporary_path.replace(output_path)
    return {
        "object": object_name,
        "output": str(output_path),
        "status": "written",
        "resolution": resolution,
        "voxel_spacing_normalized": spacing,
        "surface_abs_sdf_mean": float(surface_sdf.mean()),
        "surface_abs_sdf_p95": float(np.quantile(surface_sdf, 0.95)),
        "surface_abs_sdf_max": float(surface_sdf.max()),
        "surface_abs_sdf_p95_m": float(np.quantile(surface_sdf, 0.95) * radius),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects-root", type=Path, default=_repo_objects_dir())
    parser.add_argument("--object-names", nargs="*", help="Objects to bake; default is every object directory.")
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--grid-bound", type=float, default=DEFAULT_GRID_BOUND)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-size", type=int, default=65_536)
    parser.add_argument(
        "--basis-only",
        action="store_true",
        help="Generate or validate BPS points without baking SDFs.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.resolution < 2:
        parser.error("--resolution must be at least 2")
    if args.grid_bound <= 1.10:
        parser.error("--grid-bound must exceed the outer BPS radius of 1.10")
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")

    objects_root = args.objects_root.resolve()
    basis_path = ensure_bps_asset(objects_root, force=args.force)
    print(json.dumps({"basis": str(basis_path), "basis_sha256": _sha256(basis_path)}, sort_keys=True))
    if args.basis_only:
        return

    wp.init()
    if args.device.startswith("cuda") and not wp.is_cuda_available():
        raise RuntimeError("CUDA was requested but the NVIDIA driver is unavailable; refusing a high-memory fallback")

    if args.object_names:
        object_dirs = [objects_root / name for name in args.object_names]
    else:
        object_dirs = sorted(path.parent for path in objects_root.glob("*/*.obj"))
    if not object_dirs:
        raise RuntimeError(f"No object OBJ files found under {objects_root}")

    for object_dir in object_dirs:
        result = bake_object_sdf(
            object_dir=object_dir,
            resolution=args.resolution,
            grid_bound=args.grid_bound,
            device=args.device,
            chunk_size=args.chunk_size,
            force=args.force,
        )
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
