"""Sample uniformly distributed surface points on an object's mesh.

Reads <objects_root>/<name>/<name>.obj, samples N points (face-area-weighted uniform),
saves to <objects_root>/<name>/surface_points.npy with shape (N, 3) float32.

The sampled points are in the same coordinate frame as the source obj — which (for
clothesstand) matches the USD's local frame exactly, so they can be transformed by
the object's world pose at runtime without any extra scaling.

Usage:
    python scripts/sample_object_points.py --name clothesstand --num_points 512
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import trimesh

DEFAULT_OBJECTS_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "source/g1_hoi_learning/g1_hoi_learning/objects",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True, help="Object folder name (e.g. clothesstand)")
    parser.add_argument("--num_points", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--objects_root", type=str, default=DEFAULT_OBJECTS_ROOT)
    parser.add_argument("--out_name", type=str, default="surface_points.npy")
    args = parser.parse_args()

    obj_dir = os.path.join(args.objects_root, args.name)
    obj_path = os.path.join(obj_dir, f"{args.name}.obj")
    if not os.path.isfile(obj_path):
        raise FileNotFoundError(f"obj not found: {obj_path}")

    mesh = trimesh.load(obj_path, force="mesh")
    print(f"loaded {obj_path}: vertices={len(mesh.vertices)}, faces={len(mesh.faces)}")
    print(f"  bounds min: {mesh.bounds[0]}")
    print(f"  bounds max: {mesh.bounds[1]}")
    print(f"  extents:    {mesh.extents}")

    rng = np.random.default_rng(args.seed)
    pts, face_idx = trimesh.sample.sample_surface(mesh, args.num_points, seed=int(rng.integers(0, 2**31)))
    pts = pts.astype(np.float32)
    print(f"sampled {pts.shape[0]} points; sample bounds min={pts.min(0)}, max={pts.max(0)}")

    out_path = os.path.join(obj_dir, args.out_name)
    np.save(out_path, pts)
    print(f"saved -> {out_path}  shape={pts.shape}  dtype={pts.dtype}")


if __name__ == "__main__":
    main()
