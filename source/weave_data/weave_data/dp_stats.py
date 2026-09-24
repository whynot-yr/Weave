"""Train-split statistics computed AFTER sample-local window conversion."""

import json
from pathlib import Path

import torch

from .dp_codec import IMAGE_KEY, PROTOCOL


class Moments:
    def __init__(self, dim):
        self.n = 0
        self.mean = torch.zeros(dim, dtype=torch.float64)
        self.m2 = torch.zeros_like(self.mean)
        self.low = torch.full_like(self.mean, float("inf"))
        self.high = -self.low

    def update(self, x):
        x = x.double().reshape(-1, len(self.mean))
        if not len(x):
            return
        if not torch.isfinite(x).all():
            raise ValueError("Nonfinite training data")
        n = len(x)
        delta = x.mean(0) - self.mean
        self.m2 += ((x - x.mean(0)) ** 2).sum(0) + delta**2 * self.n * n / (self.n + n)
        self.mean += delta * n / (self.n + n)
        self.n += n
        self.low = self.low.minimum(x.amin(0))
        self.high = self.high.maximum(x.amax(0))

    def finish(self):
        return {
            "min": self.low.float(),
            "max": self.high.float(),
            "mean": self.mean.float(),
            "std": (self.m2 / self.n).sqrt().float(),
            "count": torch.tensor([self.n]),
        }


def compute_stats(dataset):
    state, action = Moments(88), Moments(125)
    for i in range(len(dataset)):
        item = dataset.numeric_item(i)
        state.update(item["observation.state"])
        action.update(item["action"][~item["action_is_pad"]])
    stats = {"observation.state": state.finish(), "action": action.finish()}
    # Analytical bounds prevent a constant training orientation/contact from
    # causing explosive normalization at inference. Other constant dims use eps.
    for key, slices in (("observation.state", [slice(0, 6)]), ("action", [slice(56, 62), slice(65, 71)])):
        for s in slices:
            stats[key]["min"][s], stats[key]["max"][s] = -1, 1
    stats["action"]["min"][71:], stats["action"]["max"][71:] = -1, 1
    for key in ("observation.state", "action"):
        low, high = stats[key]["min"], stats[key]["max"]
        constant = (high - low) < 1e-6
        low[constant] -= 1e-3
        high[constant] += 1e-3
    stats[IMAGE_KEY] = {
        "mean": torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1),
        "std": torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1),
    }
    return {
        "protocol": PROTOCOL,
        "dataset_id": dataset.protocol["dataset_id"],
        "episodes": dataset.episodes,
        "horizon": dataset.horizon,
        "stats": {k: {s: v.tolist() for s, v in fields.items()} for k, fields in stats.items()},
    }


def load_stats(path, dataset):
    payload = json.loads(Path(path).read_text())
    for key, value in {
        "protocol": PROTOCOL,
        "dataset_id": dataset.protocol["dataset_id"],
        "episodes": dataset.episodes,
        "horizon": dataset.horizon,
    }.items():
        if payload.get(key) != value:
            raise ValueError(f"Statistics {key} does not match training dataset/split/window")
    return {k: {s: torch.tensor(v) for s, v in fields.items()} for k, fields in payload["stats"].items()}
