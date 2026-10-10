"""GR00T min/max or mean/std scaling, with explicit contact-label bounds."""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .reference_codec import REFERENCE_FIELDS


def compute_statistics(dataset):
    rows = []
    for episode in dataset.episodes:
        values, valid = dataset.episode_windows(episode)
        rows.append(values[valid].numpy())
    values = np.concatenate(rows)
    statistics = {
        "min": values.min(0),
        "max": values.max(0),
        "mean": values.mean(0, dtype=np.float64),
        "std": values.std(0, dtype=np.float64),
        "q01": np.quantile(values, 0.01, axis=0),
        "q99": np.quantile(values, 0.99, axis=0),
    }
    return {
        "source_id": dataset.info["source_id"],
        "train_episodes": dataset.episodes,
        "reference_offsets": dataset.offsets.tolist(),
        "fields": REFERENCE_FIELDS,
        "statistics": {key: value.tolist() for key, value in statistics.items()},
        "valid_target_rows": len(values),
        "contact_bounds": [-1.0, 1.0],
    }


def load_statistics(path, dataset=None):
    payload = json.loads(Path(path).read_text())
    if dataset is not None:
        if (
            payload["source_id"] != dataset.info["source_id"]
            or payload["train_episodes"] != dataset.episodes
            or payload["reference_offsets"] != dataset.offsets.tolist()
        ):
            raise ValueError("Statistics source/train split/reference offsets mismatch")
    return payload


class ReferenceNormalizer(nn.Module):
    def __init__(self, statistics, mode="minmax", clip_outliers=True):
        super().__init__()
        if mode not in {"minmax", "minmax_extrema", "meanstd"}:
            raise ValueError("Unknown normalization mode")
        self.mode, self.clip_outliers = mode, clip_outliers
        stats = statistics.get("statistics", statistics)
        low_key, high_key = ("q01", "q99") if mode == "minmax" else ("min", "max")
        low, high = (
            torch.tensor(stats[low_key], dtype=torch.float32),
            torch.tensor(stats[high_key], dtype=torch.float32),
        )
        mean, std = torch.tensor(stats["mean"], dtype=torch.float32), torch.tensor(stats["std"], dtype=torch.float32)
        if any(value.shape != (125,) or not torch.isfinite(value).all() for value in (low, high, mean, std)):
            raise ValueError("Expected finite statistics for reference125")
        if (high < low).any() or (std < 0).any():
            raise ValueError("Invalid normalization bounds or variance")
        # Sparse contacts must retain the meanings -1/0/1 even when a training
        # dimension happens to contain only one class or its quantiles coincide.
        low[71:], high[71:], mean[71:], std[71:] = -1.0, 1.0, 0.0, 1.0
        constant = std < 1e-8 if mode == "meanstd" else torch.isclose(low, high)
        for name, value in (("low", low), ("high", high), ("mean", mean), ("std", std), ("constant", constant)):
            self.register_buffer(name, value)

    def _check(self, value):
        if value.shape[-1] != 125 or not torch.isfinite(value).all():
            raise ValueError("Expected finite reference125")

    def normalize(self, value):
        self._check(value)
        value = value.float()
        if self.mode == "meanstd":
            result = (value - self.mean) / self.std.clamp_min(1e-8)
        else:
            result = 2 * (value - self.low) / (self.high - self.low).clamp_min(1e-8) - 1
        result = torch.where(self.constant, 0.0, result)
        return result.clamp(-1, 1) if self.clip_outliers else result

    def unnormalize(self, value):
        self._check(value)
        value = value.float()
        if self.mode == "meanstd":
            result = value * self.std + self.mean
            return torch.where(self.constant, self.mean, result)
        result = (value.clamp(-1, 1) + 1) * 0.5 * (self.high - self.low) + self.low
        return torch.where(self.constant, self.low, result)
