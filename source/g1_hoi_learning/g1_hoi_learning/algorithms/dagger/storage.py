"""CPU uint16 depth storage, reservoir aggregation, and fresh/history minibatches."""

from __future__ import annotations

import numpy as np
import torch

from g1_hoi_learning.algorithms.dagger.networks import STUDENT_GROUPS
from g1_hoi_learning.algorithms.vision import DEPTH_FAR, depth_from_mm


def capture_samples(obs, teacher_actions, metadata: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """Own every array BEFORE env.step; camera/observation buffers may be overwritten in-place."""
    samples = {g: obs[g].detach().cpu().numpy().copy() for g in STUDENT_GROUPS if g != "head_depth"}
    samples["head_depth"] = (obs["head_depth"].detach().cpu().numpy() * (DEPTH_FAR * 1000)).round().astype(np.uint16)
    samples["teacher_action"] = teacher_actions.detach().cpu().numpy().copy()
    samples.update({k: v.detach().cpu().numpy().copy() for k, v in metadata.items()})
    if not all(np.isfinite(v).all() for v in samples.values()):
        raise ValueError("Non-finite DAgger observation/action; refusing to contaminate the replay buffer")
    return samples


def concatenate_samples(chunks: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not chunks:
        raise ValueError("Cannot train on an empty rollout")
    return {key: np.concatenate([chunk[key] for chunk in chunks], axis=0) for key in chunks[0]}


class ReservoirBuffer:
    """Algorithm R: each observed sample has equal probability of surviving the capacity limit."""

    def __init__(self, capacity: int, seed: int):
        if capacity < 1:
            raise ValueError("Buffer capacity must be positive")
        self.capacity = capacity
        self.size = 0
        self.seen = 0
        self.rng = np.random.default_rng(seed)
        self.arrays: dict[str, np.ndarray] = {}

    def add(self, samples: dict[str, np.ndarray]):
        count = len(samples["teacher_action"])
        if not self.arrays:
            self.arrays = {k: np.empty((self.capacity, *v.shape[1:]), dtype=v.dtype) for k, v in samples.items()}
        if set(samples) != set(self.arrays):
            raise ValueError("Replay schema changed")
        for key, value in samples.items():
            target = self.arrays[key]
            if len(value) != count or value.shape[1:] != target.shape[1:] or value.dtype != target.dtype:
                raise ValueError(f"Replay shape/dtype mismatch for {key}")
        for row in range(count):
            self.seen += 1
            slot = self.size if self.size < self.capacity else int(self.rng.integers(self.seen))
            if slot < self.capacity:
                for key, values in samples.items():
                    self.arrays[key][slot] = values[row]
                self.size = min(self.size + 1, self.capacity)

    def sample(self, current: dict[str, np.ndarray], batch_size: int, device: str, current_fraction: float = 0.5):
        fresh_count = batch_size if not self.size else round(batch_size * current_fraction)
        history_count = batch_size - fresh_count
        fresh_ids = self.rng.integers(len(current["teacher_action"]), size=fresh_count)
        history_ids = self.rng.integers(self.size, size=history_count) if history_count else None
        order = self.rng.permutation(batch_size)
        batch = {}
        for key in current:
            chunks = [current[key][fresh_ids]]
            if history_count:
                chunks.append(self.arrays[key][history_ids])
            batch[key] = torch.from_numpy(np.concatenate(chunks, axis=0)[order]).to(device)
        batch["head_depth"] = depth_from_mm(batch["head_depth"])
        return {g: batch[g] for g in STUDENT_GROUPS}, batch["teacher_action"]

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "size": self.size,
            "seen": self.seen,
            "rng": self.rng.bit_generator.state,
            # CPU tensor views let torch.save stream storage instead of pickling a huge NumPy byte string.
            "arrays": {key: torch.from_numpy(values[: self.size]) for key, values in self.arrays.items()},
        }

    def load_state_dict(self, state: dict):
        if state["capacity"] != self.capacity or not 0 <= state["size"] <= self.capacity:
            raise ValueError("Resume requires the same valid replay capacity")
        if state["seen"] < state["size"]:
            raise ValueError("Invalid reservoir seen counter")
        self.size, self.seen = state["size"], state["seen"]
        self.rng.bit_generator.state = state["rng"]
        self.arrays = {}
        for key, values in state["arrays"].items():
            if isinstance(values, torch.Tensor):
                values = values.numpy()
            if len(values) != self.size:
                raise ValueError(f"Corrupt buffer array: {key}")
            self.arrays[key] = np.empty((self.capacity, *values.shape[1:]), dtype=values.dtype)
            self.arrays[key][: self.size] = values
