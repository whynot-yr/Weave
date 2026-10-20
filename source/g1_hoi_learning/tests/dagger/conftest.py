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
    student_actor_spec,
)
from g1_hoi_learning.algorithms.distillation.networks import SimBaActorCriticTeacher
from g1_hoi_learning.algorithms.networks import build_group_backbone


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
    teacher = TeacherWeaveActor(spec, state)
    groups = ["ref_motion_body", "ref_motion_object", "object_state", "robot_privileged"]
    dims = dict(spec["group_dims"], robot_privileged=8)
    encoders = dict(spec["encoder_specs"], robot_privileged={"type": "mlp", "hidden_dims": [16]})
    teacher.critic_spec = {
        "groups": groups,
        "group_dims": {g: dims[g] for g in groups},
        "encoder_specs": encoders,
        "latent_dim": 8,
        "hidden_dim": 16,
        "num_blocks": 2,
        "expansion": 1,
        "normalize": True,
    }
    critic = build_group_backbone({g: torch.empty(1, dims[g]) for g in groups}, groups, encoders, 8, 1, 16, 2, 1)
    critic_norm = normalizer(sum(dims[g] for g in groups), True)
    state.update({f"critic.{k}": v for k, v in critic.state_dict().items()})
    state.update({f"critic_obs_normalizer.{k}": v for k, v in critic_norm.state_dict().items()})
    teacher.tracker_state = state
    return teacher


@pytest.fixture
def student_spec(spec):
    return student_actor_spec(spec, {**spec["group_dims"], "goal_robot_pose": 62, "goal_object_pose": 9})


@pytest.fixture
def observations(spec):
    obs = {g: torch.randn(4, dim) for g, dim in spec["group_dims"].items()}
    obs["robot_privileged"] = torch.randn(4, 8)
    obs["goal_robot_pose"] = torch.randn(4, 62)
    obs["goal_object_pose"] = torch.randn(4, 9)
    return obs


def make_policy(spec, teacher, student=None):
    from g1_hoi_learning.algorithms.dagger.networks import StudentWeaveActor

    if student is None:
        student = StudentWeaveActor(
            student_actor_spec(spec, {**spec["group_dims"], "goal_robot_pose": 62, "goal_object_pose": 9})
        )
        student.initialize_from_teacher(teacher)
    return SimBaActorCriticTeacher(student, teacher, teacher.critic_spec, teacher.tracker_state)
