"""Persisted motion-family splits; variants never cross the train/val boundary."""

import json
import math
import re
from pathlib import Path

import numpy as np

from .reference_dataset import inspect_source


def family(episode):
    return episode["object_name"] + "/" + re.sub(r"_v\d+$", "", episode["clip_name"])


def make_split(source, val_ratio=0.2, seed=42, success_only=False):
    if not 0 < val_ratio < 1:
        raise ValueError("Validation ratio must lie in (0,1)")
    info = inspect_source(source)
    selected = [e for e in info["episodes"] if e["length"] > 0 and (not success_only or e["success"])]
    families = sorted({family(e) for e in selected})
    if len(families) < 2:
        raise ValueError("Need at least two motion families")
    order = np.random.default_rng(seed).permutation(families).tolist()
    count = min(len(families) - 1, math.ceil(len(families) * val_ratio))
    val = set(order[:count])
    return {
        "source_id": info["source_id"],
        "seed": seed,
        "val_ratio": val_ratio,
        "success_only": success_only,
        "train_episodes": [e["episode_index"] for e in selected if family(e) not in val],
        "val_episodes": [e["episode_index"] for e in selected if family(e) in val],
    }


def load_split(path, source):
    split = json.loads(Path(path).read_text())
    info = inspect_source(source)
    if split["source_id"] != info["source_id"]:
        raise ValueError("Split does not match source")
    train, val = split["train_episodes"], split["val_episodes"]
    if not train or not val or len(set(train + val)) != len(train + val):
        raise ValueError("Split is empty, overlapping or contains duplicates")
    all_ids = {
        e["episode_index"]
        for e in info["episodes"]
        if e["length"] > 0 and (not split.get("success_only", False) or e["success"])
    }
    if set(train + val) != all_ids:
        raise ValueError("Split does not cover selected episodes")
    train_families = {family(info["episodes"][i]) for i in train}
    if train_families & {family(info["episodes"][i]) for i in val}:
        raise ValueError("Motion family leaked across train and validation")
    return split
