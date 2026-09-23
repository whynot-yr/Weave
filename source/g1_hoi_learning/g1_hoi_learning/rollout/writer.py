from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .schema import CAMERA_ORIENTATION, COORDINATE_FRAME, DEPTH_ENCODING, QUATERNION_ORDER, SCHEMA_VERSION

try:
    import h5py
except ImportError as exc:  # pragma: no cover - exercised in the Isaac environment
    raise ImportError(
        "Rollout collection requires h5py. Re-run install.sh or install h5py in the Isaac environment."
    ) from exc


class RolloutH5Writer:
    """Stream a fixed set of parallel episodes into a chunked HDF5 file.

    Frame datasets are episode-major: ``[episode, step, ...]``. Unwritten tail
    frames are invalid and must be ignored using ``episodes/length`` or
    ``frame/valid``.
    """

    def __init__(
        self,
        output_dir: str,
        *,
        clip_ids: np.ndarray,
        clip_names: list[str],
        reference_lengths: np.ndarray,
        object_names: list[str],
        episode_seed: int,
        max_steps: int,
        metadata: dict,
    ):
        self.output_dir = Path(output_dir).resolve()
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError(f"output directory is not empty: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / "rollouts.h5"
        self.num_episodes = len(clip_ids)
        self.max_steps = int(max_steps)
        self._file = h5py.File(self.path, "w")
        self._datasets: dict[str, h5py.Dataset] = {}

        self._file.attrs["schema_version"] = SCHEMA_VERSION
        self._file.attrs["coordinate_frame"] = COORDINATE_FRAME
        self._file.attrs["quaternion_order"] = QUATERNION_ORDER
        self._file.attrs["depth_encoding"] = DEPTH_ENCODING
        self._file.attrs["camera_orientation"] = CAMERA_ORIENTATION
        self._file.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)

        strings = h5py.string_dtype(encoding="utf-8")
        episodes = self._file.require_group("episodes")
        episodes.create_dataset("id", data=np.arange(self.num_episodes, dtype=np.int64))
        episodes.create_dataset("clip_id", data=np.asarray(clip_ids, dtype=np.int32))
        episodes.create_dataset("clip_name", data=np.asarray(clip_names, dtype=object), dtype=strings)
        episodes.create_dataset("object_name", data=np.asarray(object_names, dtype=object), dtype=strings)
        episodes.create_dataset("reference_length", data=np.asarray(reference_lengths, dtype=np.int32))
        episodes.create_dataset(
            "seed", data=np.full(self.num_episodes, int(episode_seed), dtype=np.int64)
        )
        episodes.create_dataset("length", data=np.zeros(self.num_episodes, dtype=np.int32))
        episodes.create_dataset("success", data=np.zeros(self.num_episodes, dtype=np.bool_))
        episodes.create_dataset(
            "termination_reason", data=np.asarray([""] * self.num_episodes, dtype=object), dtype=strings
        )
        episodes.create_dataset("attempt_id", data=np.zeros(self.num_episodes, dtype=np.int32))
        frame = self._file.require_group("frame")
        frame.create_dataset(
            "valid",
            shape=(self.num_episodes, self.max_steps),
            dtype=np.bool_,
            chunks=(1, min(self.max_steps, 256)),
            compression="lzf",
        )

    def _dataset(self, name: str, sample: np.ndarray) -> h5py.Dataset:
        if name in self._datasets:
            return self._datasets[name]
        sample = np.asarray(sample)
        tail_shape = sample.shape[1:]
        shape = (self.num_episodes, self.max_steps, *tail_shape)
        chunk = (1, 1, *tail_shape)
        parent, _, leaf = name.rpartition("/")
        group = self._file.require_group(parent) if parent else self._file
        dataset = group.create_dataset(
            leaf,
            shape=shape,
            dtype=sample.dtype,
            chunks=chunk,
            compression="lzf",
            shuffle=sample.dtype.itemsize > 1,
        )
        self._datasets[name] = dataset
        return dataset

    def write_step(self, env_ids: np.ndarray, step: int, values: dict[str, np.ndarray]) -> None:
        if step >= self.max_steps:
            raise IndexError(f"step {step} exceeds allocated max_steps={self.max_steps}")
        for name, value in values.items():
            value = np.asarray(value)
            if value.shape[0] != len(env_ids):
                raise ValueError(f"{name}: expected leading dimension {len(env_ids)}, got {value.shape}")
            self._dataset(name, value)[env_ids, step] = value
        self._file["frame/valid"][env_ids, step] = True

    def finish_episode(self, env_id: int, *, length: int, success: bool, reason: str) -> None:
        self._file["episodes/length"][env_id] = int(length)
        self._file["episodes/success"][env_id] = bool(success)
        self._file["episodes/termination_reason"][env_id] = reason

    def close(self, manifest: dict) -> None:
        if self._file is None:
            return
        self._file.flush()
        self._file.close()
        self._file = None
        manifest = {
            **manifest,
            "schema_version": SCHEMA_VERSION,
            "coordinate_frame": COORDINATE_FRAME,
            "quaternion_order": QUATERNION_ORDER,
            "depth_encoding": DEPTH_ENCODING,
            "files": [os.path.basename(self.path)],
        }
        with open(self.output_dir / "manifest.json", "w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, ensure_ascii=False)
