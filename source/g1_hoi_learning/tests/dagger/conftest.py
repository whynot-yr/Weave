"""Load simulator-free algorithm modules without launching the extension's UI/task registration."""

import sys
import types
from pathlib import Path

import pytest
import torch

PACKAGE = Path(__file__).resolve().parents[2] / "g1_hoi_learning"
for name, path in (("g1_hoi_learning", PACKAGE), ("g1_hoi_learning.algorithms", PACKAGE / "algorithms")):
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module

torch.set_num_threads(2)

from g1_hoi_learning.algorithms.dagger.networks import (
    TeacherWeaveActor,
    actor_spec,
    make_actor,
    normalizer,
)


@pytest.fixture
def spec():
    groups = ["ref_motion_body", "ref_motion_object", "object_state", "robot_proprio"]
    cfg = {
        "obs_groups": {"policy": groups},
        "policy": {
            "class_name": "SimBaActorCritic",
            "latent_dim": 8,
            "actor_hidden_dim": 16,
            "actor_num_blocks": 1,
            "expansion": 1,
            "actor_obs_normalization": True,
            "encoder_hidden_dims": {g: {"type": "mlp", "hidden_dims": [16]} for g in groups},
        },
    }
    return actor_spec(cfg, dict(zip(groups, [5, 4, 7, 6])), 3)


@pytest.fixture
def teacher(spec):
    actor = make_actor(spec)
    norm = normalizer(sum(spec["group_dims"].values()), True)
    with torch.no_grad():
        norm._mean.copy_(torch.arange(22).float().unsqueeze(0) / 20)
        norm._std.fill_(2.0)
        norm._var.fill_(4.0)
        norm.count.fill_(100)
    state = {f"actor.{k}": v for k, v in actor.state_dict().items()}
    state.update({f"actor_obs_normalizer.{k}": v for k, v in norm.state_dict().items()})
    return TeacherWeaveActor(spec, state)


@pytest.fixture
def observations(spec):
    obs = {g: torch.randn(4, dim) for g, dim in spec["group_dims"].items()}
    obs["head_depth"] = torch.full((4, 1, 128, 128), 0.2)
    return obs
