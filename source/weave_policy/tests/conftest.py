import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_num_threads(2)


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "rollouts.h5"
    episodes, steps = 4, 12
    with h5py.File(path, "w") as f:
        f.attrs.update(
            schema_version="weave-rollout-v1",
            quaternion_order="wxyz",
            coordinate_frame="env_local_world_axes",
            metadata_json=json.dumps(
                {
                    "joint_names": [f"j{i}" for i in range(53)],
                    "body_names": ["pelvis"] + [f"b{i}" for i in range(53)],
                    "fps": 50.0,
                }
            ),
        )
        f.create_dataset("camera/head/rgb", shape=(episodes, steps, 224, 224, 3), dtype="u1", compression="lzf")
        f["episodes/length"] = np.full(episodes, steps)
        f["episodes/success"] = np.ones(episodes, dtype=bool)
        f["episodes/clip_name"] = np.array(["a_v1", "a_v2", "b_v1", "b_v2"], dtype=h5py.string_dtype())
        f["episodes/object_name"] = np.array(["lamp"] * episodes, dtype=h5py.string_dtype())
        f["frame/valid"] = np.ones((episodes, steps), dtype=bool)
        f["reference/joint_pos"] = np.arange(episodes * steps * 53).reshape(episodes, steps, 53) / 1000
        f["reference/contact_label"] = np.zeros((episodes, steps, 54))
        pos = np.zeros((episodes, steps, 54, 3), dtype=np.float32)
        pos[:, :, 0, 0] = np.arange(steps) * 0.1
        quat = np.zeros((episodes, steps, 54, 4), dtype=np.float32)
        quat[..., 0] = 1
        for group in ("robot", "reference"):
            f[group + "/body_pos"] = pos
            f[group + "/body_quat"] = quat
        f["reference/object_pos"] = pos[:, :, 0] + [0, 1, 0]
        f["reference/object_quat"] = quat[:, :, 0]
    return path


@pytest.fixture
def statistics():
    return {
        "statistics": {
            "min": [-2.0] * 125,
            "max": [2.0] * 125,
            "q01": [-1.0] * 125,
            "q99": [1.0] * 125,
            "mean": [0.0] * 125,
            "std": [2.0] * 125,
        }
    }
