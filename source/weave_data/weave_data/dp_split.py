"""Reproducible, motion-family-disjoint splits using metadata only."""

import json
import math
import random
import re
from pathlib import Path

from .dp_codec import PROTOCOL

SPLIT_VERSION = "weave-motion-split-v1"


def motion_family(clip_name):
    return re.sub(r"_v\d+$", "", clip_name)


def read_protocol(root):
    protocol = json.loads((Path(root) / "meta/weave_protocol.json").read_text())
    if protocol.get("protocol") != PROTOCOL or not protocol.get("complete"):
        raise ValueError("Incomplete or incompatible WEAVE dataset")
    episodes = protocol.get("episodes", [])
    ids = [e["episode_index"] for e in episodes]
    if not episodes or len(set(ids)) != len(ids):
        raise ValueError("Empty dataset or duplicate episode indices")
    for e in episodes:
        if type(e["episode_index"]) is not int or e["episode_index"] < 0:
            raise ValueError("Invalid episode index")
        if not isinstance(e.get("clip_name"), str) or not e["clip_name"] or e.get("length", 0) <= 0:
            raise ValueError("Every episode needs a clip_name and positive length")
    return protocol


def make_split(protocol, val_ratio=0.2, seed=42, by_object=False):
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must lie in (0,1)")
    groups = {}
    for e in protocol["episodes"]:
        groups.setdefault(motion_family(e["clip_name"]), []).append(e)
    strata = {}
    for family, episodes in sorted(groups.items()):
        objects = {e.get("object_name") for e in episodes}
        if by_object and (len(objects) != 1 or not next(iter(objects))):
            raise ValueError("--by-object requires consistent object_name metadata for every motion family")
        stratum = next(iter(objects)) if by_object else "all"
        strata.setdefault(stratum, []).append(family)
    train_groups, val_groups = [], []
    rng = random.Random(seed)
    for stratum, families in sorted(strata.items()):
        if len(families) < 2:
            raise ValueError(f"Stratum {stratum!r} has fewer than two motion families; cannot split without leakage")
        rng.shuffle(families)
        n_val = min(len(families) - 1, max(1, math.ceil(len(families) * val_ratio)))
        val_groups.extend(families[:n_val])
        train_groups.extend(families[n_val:])
    result = {
        "split_version": SPLIT_VERSION,
        "dataset_id": protocol["dataset_id"],
        "seed": seed,
        "group_by": "original_motion",
        "variant_suffix": "_v[0-9]+$",
        "val_ratio": val_ratio,
        "by_object": by_object,
    }
    for name, families in (("train", train_groups), ("val", val_groups)):
        episodes = [e for g in families for e in groups[g]]
        result[f"{name}_groups"] = sorted(families)
        result[f"{name}_episodes"] = sorted(e["episode_index"] for e in episodes)
        result[f"{name}_summary"] = {
            "groups": len(families),
            "episodes": len(episodes),
            "frames": sum(e["length"] for e in episodes),
        }
    validate_split(result, protocol)
    return result


def validate_split(split, protocol):
    if split.get("split_version") != SPLIT_VERSION or split.get("group_by") != "original_motion":
        raise ValueError("Unsupported split format/grouping")
    if split.get("dataset_id") != protocol["dataset_id"]:
        raise ValueError("Split dataset_id does not match dataset")
    all_ids = {e["episode_index"] for e in protocol["episodes"]}
    sets = {}
    families = {}
    for name in ("train", "val"):
        ids = split.get(f"{name}_episodes", [])
        if not ids or any(type(i) is not int for i in ids) or len(set(ids)) != len(ids):
            raise ValueError(f"{name} episodes must be nonempty unique integer indices")
        sets[name] = set(ids)
        if not sets[name] <= all_ids:
            raise ValueError(f"Unknown {name} episode indices")
        families[name] = {
            motion_family(e["clip_name"]) for e in protocol["episodes"] if e["episode_index"] in sets[name]
        }
        if sorted(families[name]) != split.get(f"{name}_groups"):
            raise ValueError(f"{name} group list does not match episodes")
    if sets["train"] & sets["val"]:
        raise ValueError("Train/validation episodes overlap")
    if sets["train"] | sets["val"] != all_ids:
        raise ValueError("Split must cover every dataset episode")
    if families["train"] & families["val"]:
        raise ValueError("Train/validation motion families overlap")


def load_split(path, root):
    split = json.loads(Path(path).read_text())
    validate_split(split, read_protocol(root))
    return split
