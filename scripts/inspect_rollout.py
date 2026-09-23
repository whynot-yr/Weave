#!/usr/bin/env python3
"""Validate and summarize a WEAVE rollout HDF5 file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def decode(values) -> list[str]:
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in values]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="rollouts.h5 or its containing directory")
    args = parser.parse_args()
    path = args.path / "rollouts.h5" if args.path.is_dir() else args.path

    errors: list[str] = []
    with h5py.File(path, "r") as data:
        lengths = data["episodes/length"][:]
        success = data["episodes/success"][:]
        reasons = decode(data["episodes/termination_reason"][:])
        clip_ids = data["episodes/clip_id"][:]
        valid = data["frame/valid"][:]
        expected_valid = np.arange(valid.shape[1])[None, :] < lengths[:, None]
        if not np.array_equal(valid, expected_valid):
            errors.append("frame/valid does not match episodes/length")
        if len(np.unique(clip_ids)) != len(clip_ids):
            errors.append("clip ids are not unique")
        if "camera/head/rgb" not in data or "camera/head/depth_mm" not in data:
            errors.append("head RGB-D datasets are missing")
        else:
            if data["camera/head/rgb"].shape[2:] != (224, 224, 3):
                errors.append(f"unexpected RGB shape: {data['camera/head/rgb'].shape}")
            if data["camera/head/depth_mm"].shape[2:] != (224, 224):
                errors.append(f"unexpected depth shape: {data['camera/head/depth_mm'].shape}")
        for name in ("robot/root_quat", "object/quat", "camera/head/quaternion_world"):
            if name in data:
                values = data[name][:]
                norms = np.linalg.norm(values[valid], axis=-1)
                if len(norms) and not np.allclose(norms, 1.0, atol=2.0e-3):
                    errors.append(f"{name} contains non-unit quaternions")
        summary = {
            "schema_version": data.attrs.get("schema_version", "unknown"),
            "episodes": int(len(lengths)),
            "frames": int(lengths.sum()),
            "unique_clips": int(len(np.unique(clip_ids))),
            "successes": int(success.sum()),
            "success_rate": float(success.mean()) if len(success) else 0.0,
            "termination_counts": {reason: reasons.count(reason) for reason in sorted(set(reasons))},
            "length_min": int(lengths.min()) if len(lengths) else 0,
            "length_max": int(lengths.max()) if len(lengths) else 0,
            "errors": errors,
        }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
