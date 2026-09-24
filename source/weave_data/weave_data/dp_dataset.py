"""Offline window assembly following WEAVE PPO semantics, not LeRobot transforms."""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .dp_codec import IMAGE_KEY, PROTOCOL, REFERENCE_DIMS, encode_reference
from .reference_window import window_indices


class DPWindowDataset(Dataset):
    def __init__(self, root, episodes=None, horizon=40, repo_id="weave/local"):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.root = Path(root)
        self.protocol = json.loads((self.root / "meta/weave_protocol.json").read_text())
        if self.protocol["protocol"] != PROTOCOL or not self.protocol.get("complete"):
            raise ValueError("Incomplete or incompatible WEAVE dataset")
        if horizon <= 0:
            raise ValueError("Horizon must be positive")
        all_episodes = [e["episode_index"] for e in self.protocol["episodes"]]
        self.episodes = sorted(all_episodes if episodes is None else episodes)
        if (
            not self.episodes
            or len(set(self.episodes)) != len(self.episodes)
            or not set(self.episodes) <= set(all_episodes)
        ):
            raise ValueError("Invalid/empty episode selection")
        self.horizon = horizon
        # No LeRobot delta_timestamps: WEAVE owns selection/clamping and transforms.
        self.raw = LeRobotDataset(repo_id, root=self.root, episodes=self.episodes, video_backend="pyav")
        keys = ["observation.state", "aux.pelvis_pos", "aux.pelvis_quat", "episode_index", "frame_index"]
        keys += [f"reference.{key}" for key in REFERENCE_DIMS]
        # Only small numeric columns are cached; never load RGB to compute stats.
        columns = self.raw.select_columns(keys).with_format("numpy")[:]
        self.data = {key: torch.as_tensor(value.copy()) for key, value in columns.items()}
        episode = self.data["episode_index"].reshape(-1)
        self.starts = torch.empty(len(episode), dtype=torch.long)
        self.lengths = torch.empty_like(self.starts)
        cursor = 0
        for e, length in zip(*torch.unique_consecutive(episode, return_counts=True), strict=True):
            n = int(length)
            if not torch.equal(self.data["frame_index"][cursor : cursor + n].reshape(-1).long(), torch.arange(n)):
                raise ValueError(f"Episode {int(e)} contains missing/reordered frames")
            self.starts[cursor : cursor + n] = cursor
            self.lengths[cursor : cursor + n] = n
            cursor += n
        self.offsets = torch.arange(horizon)

    def __len__(self):
        return len(self.starts)

    def numeric_item(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        indices, pad = window_indices(
            torch.tensor(index) - self.starts[index], self.offsets, self.lengths[index], self.starts[index]
        )
        reference = {k: self.data[f"reference.{k}"][indices].float() for k in REFERENCE_DIMS}
        action = encode_reference(
            reference, self.data["aux.pelvis_pos"][index].float(), self.data["aux.pelvis_quat"][index].float()
        )
        return {
            "observation.state": self.data["observation.state"][index].float().unsqueeze(0),
            "action": action,
            "action_is_pad": pad,
        }

    def __getitem__(self, index):
        item = self.numeric_item(index)
        item[IMAGE_KEY] = self.raw[index][IMAGE_KEY].unsqueeze(0)
        return item

    def __getitems__(self, indices):
        return [self[index] for index in indices]
