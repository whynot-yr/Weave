"""Episode-local windows from raw WEAVE HDF5, without Isaac or LeRobot."""

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import ReferencePolicyConfig
from .reference_codec import encode_reference


def inspect_source(path):
    path = Path(path).resolve()
    with h5py.File(path, "r") as source:
        for key, expected in (
            ("schema_version", "weave-rollout-v1"),
            ("quaternion_order", "wxyz"),
            ("coordinate_frame", "env_local_world_axes"),
        ):
            if source.attrs.get(key) != expected:
                raise ValueError(f"Unsupported source {key}")
        metadata = json.loads(source.attrs["metadata_json"])
        if len(metadata["joint_names"]) != 53 or len(metadata["body_names"]) != 54:
            raise ValueError("Expected 53 joint and 54 body names")
        if "pelvis" not in metadata["body_names"]:
            raise ValueError("Source requires pelvis anchor")
        rgb = source["camera/head/rgb"]
        if rgb.shape[2:] != (224, 224, 3) or rgb.dtype != np.uint8:
            raise ValueError("Expected recorded RGB uint8[episode,step,224,224,3]")
        lengths = source["episodes/length"][:].astype(int).tolist()
        if any(n < 0 or n > rgb.shape[1] for n in lengths):
            raise ValueError("Invalid episode lengths")
        episodes = [
            {
                "episode_index": i,
                "clip_name": str(source["episodes/clip_name"].asstr()[i]),
                "object_name": str(source["episodes/object_name"].asstr()[i]),
                "length": n,
                "success": bool(source["episodes/success"][i]),
            }
            for i, n in enumerate(lengths)
        ]
    stat = path.stat()
    identity = {"metadata": metadata, "episodes": episodes, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    source_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    calibration_path = path.parent / "resolved_config.json"
    calibration = None
    if calibration_path.is_file():
        calibration = json.loads(calibration_path.read_text())["env"]["scene"]["head_camera"]
    return {
        "source_id": source_id,
        "source": str(path),
        "metadata": metadata,
        "episodes": episodes,
        "camera_calibration": calibration,
    }


class ReferenceWindowDataset(Dataset):
    def __init__(self, source, episodes, offsets=ReferencePolicyConfig.reference_offsets, features=None):
        self.info = inspect_source(source)
        self.source = self.info["source"]
        self.episodes = sorted(episodes)
        if not self.episodes or len(set(self.episodes)) != len(self.episodes):
            raise ValueError("Episode selection must be nonempty and unique")
        if not set(self.episodes) <= set(range(len(self.info["episodes"]))):
            raise ValueError("Unknown episode index")
        if any(self.info["episodes"][i]["length"] <= 0 for i in self.episodes):
            raise ValueError("Empty episodes cannot form windows")
        self.offsets = torch.tensor(offsets, dtype=torch.long)
        if (
            len(offsets) != ReferencePolicyConfig.horizon
            or offsets[0] != 0
            or (self.offsets[1:] <= self.offsets[:-1]).any()
        ):
            raise ValueError("Expected five strictly increasing offsets starting at zero")
        self.features_path = str(Path(features).resolve()) if features is not None else None
        self.feature_metadata = None
        self.feature_rows = {}
        if features is not None:
            self._inspect_features()
        self._handles = None
        self.numeric = {}
        self.windows = []
        pelvis = self.info["metadata"]["body_names"].index("pelvis")
        with h5py.File(self.source, "r") as file:
            for episode in self.episodes:
                length = self.info["episodes"][episode]["length"]
                if not file["frame/valid"][episode, :length].all():
                    raise ValueError(f"Episode {episode} has missing valid frames")
                data = {}
                keys = {
                    "joint_pos": "reference/joint_pos",
                    "object_pos": "reference/object_pos",
                    "object_quat": "reference/object_quat",
                    "contact": "reference/contact_label",
                }
                for key, field in keys.items():
                    data[key] = torch.from_numpy(file[field][episode, :length].astype(np.float32))
                for key, field in {
                    "pelvis_pos": "reference/body_pos",
                    "pelvis_quat": "reference/body_quat",
                    "actual_pos": "robot/body_pos",
                    "actual_quat": "robot/body_quat",
                }.items():
                    data[key] = torch.from_numpy(file[field][episode, :length, pelvis].astype(np.float32))
                if not all(torch.isfinite(value).all() for value in data.values()):
                    raise ValueError(f"Episode {episode} has nonfinite numeric values")
                if not torch.isin(data["contact"], torch.tensor([-1.0, 0.0, 1.0])).all():
                    raise ValueError("Unexpected contact labels")
                self.numeric[episode] = data
                self.windows.extend((episode, step) for step in range(length))

    def _inspect_features(self):
        with h5py.File(self.features_path, "r") as file:
            if not file.attrs.get("complete", False) or file.attrs.get("source_id") != self.info["source_id"]:
                raise ValueError("Incomplete feature cache or mismatched source")
            self.feature_metadata = json.loads(file.attrs["encoder_json"])
            if file["tokens"].shape[1:] != (257, 768):
                raise ValueError("Expected DINOv2 token cache [N,257,768]")
            for row, (episode, step) in enumerate(zip(file["episode_index"][:], file["frame_index"][:], strict=True)):
                key = (int(episode), int(step))
                if key in self.feature_rows:
                    raise ValueError("Duplicate frame in feature cache")
                self.feature_rows[key] = row
            for episode in self.episodes:
                length = self.info["episodes"][episode]["length"]
                if any((episode, step) not in self.feature_rows for step in range(length)):
                    raise ValueError(f"Feature cache missing episode {episode} frames")

    def __len__(self):
        return len(self.windows)

    def episode_windows(self, episode):
        """Vectorized numeric windows used for statistics, without decoding RGB."""
        data = self.numeric[episode]
        length = len(data["joint_pos"])
        indices = torch.arange(length)[:, None] + self.offsets[None]
        selected = indices.clamp_max(length - 1)
        reference = {key: value[selected] for key, value in data.items() if not key.startswith("actual_")}
        value = encode_reference(reference, data["actual_pos"], data["actual_quat"])
        return value, indices < length

    def numeric_item(self, index):
        episode, step = self.windows[index]
        data = self.numeric[episode]
        length = len(data["joint_pos"])
        indices = step + self.offsets
        mask = (indices < length).unsqueeze(-1).expand(-1, 125)
        selected = indices.clamp_max(length - 1)
        reference = {key: value[selected] for key, value in data.items() if not key.startswith("actual_")}
        value = encode_reference(reference, data["actual_pos"][step], data["actual_quat"][step])
        return {"reference": value, "reference_mask": mask, "episode_index": episode, "frame_index": step}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = None
        return state

    def __getitem__(self, index):
        item = self.numeric_item(index)
        if self._handles is None:
            self._handles = h5py.File(self.features_path or self.source, "r")
        episode, step = self.windows[index]
        if self.features_path:
            value = self._handles["tokens"][self.feature_rows[(episode, step)]].astype(np.float32)
            if not np.isfinite(value).all():
                raise ValueError("Nonfinite cached DINOv2 tokens")
            item["image_tokens"] = torch.from_numpy(value)
        else:
            item["rgb"] = torch.from_numpy(self._handles["camera/head/rgb"][episode, step])
        return item

    def close(self):
        if self._handles is not None:
            self._handles.close()
            self._handles = None
