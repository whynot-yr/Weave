"""Preserved perception, evaluation, legacy buffer and simulation-device binding."""

import copy

import numpy as np
import pytest
import torch
from g1_hoi_learning.algorithms.dagger.evaluation import assign_eval_clips, summarize
from g1_hoi_learning.algorithms.dagger.io import read_yaml
from g1_hoi_learning.algorithms.dagger.networks import (
    TEACHER_GROUPS,
    VECTOR_GROUPS,
    StudentWeaveActor,
    TeacherWeaveActor,
    export_student,
)
from g1_hoi_learning.algorithms.dagger.storage import RolloutBuffer
from g1_hoi_learning.algorithms.ppo.networks import SimBaActorCritic
from g1_hoi_learning.algorithms.vision import depth_from_mm, depth_to_mm, preprocess_depth
from tensordict import TensorDict


def test_depth_invalid_units_and_replay_round_trip():
    depth = torch.tensor([float("nan"), float("inf"), -1.0, 0.099, 0.1, 2.1234, 10.0, 10.01])
    mm = depth_to_mm(depth)
    assert mm.tolist() == [0, 0, 0, 0, 100, 2123, 10000, 0]
    torch.testing.assert_close(preprocess_depth(depth), depth_from_mm(mm), rtol=0, atol=0)


def test_isaaclab_yaml_snapshot_tuple_and_slice(tmp_path):
    snapshot = tmp_path / "env.yaml"
    snapshot.write_text(
        "pose: !!python/tuple [1.0, 2.0, 3.0]\nbody_ids: !!python/object/apply:builtins.slice [null, null, null]\n"
    )
    parsed = read_yaml(str(snapshot))
    assert parsed == {"pose": (1.0, 2.0, 3.0), "body_ids": slice(None)}


def test_teacher_rejects_wrong_checkpoint_dimensions(spec, teacher):
    state = {f"actor.{key}": value for key, value in teacher.actor.state_dict().items()}
    state.update({f"actor_obs_normalizer.{key}": value for key, value in teacher.normalizer.state_dict().items()})
    wrong = copy.deepcopy(spec)
    wrong["group_dims"]["object_state"] += 1
    with pytest.raises(RuntimeError, match="size mismatch"):
        TeacherWeaveActor(wrong, state)


def test_eval_clip_batch_reuses_scene_and_pads_final_batch():
    from types import SimpleNamespace

    command = SimpleNamespace(
        cfg=SimpleNamespace(eval_mode=True, eval_clip_ids=[]),
        motion=SimpleNamespace(num_objects=1, num_clips=7),
        num_envs=4,
        _eval_env_clip=torch.zeros(4, dtype=torch.long),
    )
    assign_eval_clips(command, [2, 5, 6])
    assert command.cfg.eval_clip_ids == [2, 5, 6, 6]
    assign_eval_clips(command, [1])
    assert command._eval_env_clip.tolist() == [1, 1, 1, 1]
    with pytest.raises(ValueError, match="out of range"):
        assign_eval_clips(command, [7])


def test_eval_aggregation_weights_action_changes_by_edges():
    rows = [
        dict(
            steps=10,
            action_change_steps=9,
            action_change_mse=1.0,
            tracking=2.0,
            success=True,
            finished=True,
            progress=1.0,
            cause="clip_end",
        ),
        dict(
            steps=2,
            action_change_steps=1,
            action_change_mse=9.0,
            tracking=4.0,
            success=False,
            finished=True,
            progress=0.2,
            cause="object_pos",
        ),
    ]
    result = summarize(rows, ["tracking", "action_change_mse"])
    assert result["metrics_all"]["tracking"] == pytest.approx(28 / 12)
    assert result["metrics_all"]["action_change_mse"] == pytest.approx(1.8)
    assert result["metrics_success"]["action_change_mse"] == 1.0
    assert result["failures"] == {"object_pos": 1}


def test_teacher_matches_original_actorcritic(spec, observations):
    obs = TensorDict(observations, batch_size=[4])
    original = SimBaActorCritic(
        obs,
        {"policy": list(TEACHER_GROUPS), "critic": list(TEACHER_GROUPS)},
        spec["num_actions"],
        latent_dim=spec["latent_dim"],
        actor_hidden_dim=spec["hidden_dim"],
        critic_hidden_dim=16,
        actor_num_blocks=spec["num_blocks"],
        expansion=spec["expansion"],
        encoder_hidden_dims=spec["encoder_specs"],
        compile=False,
    ).eval()
    original.actor_obs_normalizer.update(torch.randn(4, 22))  # eval prevents normalizer updates
    teacher = TeacherWeaveActor(spec, original.state_dict())
    torch.testing.assert_close(teacher(obs), original.act_inference(obs), rtol=0, atol=0)
    teacher.train()
    assert not teacher.training and not teacher.normalizer.training


def test_transfer_group_normalizers_and_student_has_no_object_leak(spec, teacher, observations):
    student = StudentWeaveActor(spec)
    student.initialize_from_teacher(teacher)
    offset = 0
    for group in TEACHER_GROUPS:
        width = spec["group_dims"][group]
        if group in VECTOR_GROUPS:
            norm = student.vector_normalizers[group]
            torch.testing.assert_close(norm._mean, teacher.normalizer._mean[:, offset : offset + width])
            for key, value in student.vector_encoders[group].state_dict().items():
                torch.testing.assert_close(
                    value, teacher.actor[0].encoders[TEACHER_GROUPS.index(group)].state_dict()[key]
                )
        offset += width
    for key, value in student.backbone.state_dict().items():
        torch.testing.assert_close(value, teacher.actor[1].state_dict()[key])
    expected = student(observations)
    changed = dict(observations, object_state=torch.full_like(observations["object_state"], 1e6))
    torch.testing.assert_close(student(changed), expected, rtol=0, atol=0)
    assert len([m for m in student.depth_encoder.modules() if isinstance(m, torch.nn.Conv2d)]) == 4
    teacher_before = copy.deepcopy(teacher.state_dict())
    with torch.no_grad():
        next(student.backbone.parameters()).add_(1)
    for key, value in teacher.state_dict().items():
        torch.testing.assert_close(value, teacher_before[key], rtol=0, atol=0)


def make_rollout(observations, target, steps=1):
    rollout = RolloutBuffer(len(target), steps)
    for _ in range(steps):
        rollout.add(observations, target, {"episode_step": torch.arange(len(target))})
    return rollout


def test_rollout_owns_pre_step_data_reuses_allocation_and_has_no_history(observations):
    target = torch.arange(12).float().reshape(4, 3)
    expected = target.clone()
    rollout = make_rollout(observations, target)
    arrays = dict(rollout.arrays)
    target.zero_()
    observations["head_depth"].zero_()
    np.testing.assert_array_equal(rollout.arrays["teacher_action"], expected.numpy())
    assert rollout.arrays["head_depth"].dtype == np.uint16
    assert rollout.arrays["head_depth"].min() == 2000
    obs, actions = rollout.batch(np.arange(4), "cpu")
    assert obs["head_depth"].shape == (4, 1, 128, 128)
    torch.testing.assert_close(obs["head_depth"], torch.full_like(obs["head_depth"], 0.2))
    np.testing.assert_array_equal(actions.numpy(), expected.numpy())
    rollout.clear()
    with pytest.raises(ValueError, match="incomplete"):
        list(rollout.minibatches(2))
    rollout.add(observations, target, {"episode_step": torch.arange(4)})
    assert all(rollout.arrays[key] is value for key, value in arrays.items())
    assert rollout.arrays["teacher_action"].max() == 0
    assert rollout.arrays["head_depth"].max() == 0
    with pytest.raises(ValueError, match="full"):
        rollout.add(observations, target, {"episode_step": torch.arange(4)})


def test_rollout_rejects_nonfinite_data_before_it_can_be_trained(observations):
    rollout = RolloutBuffer(4, 1)
    observations["head_depth"][0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        rollout.add(observations, torch.zeros(4, 3), {})
    assert rollout.size == 0


def test_epoch_balanced_partition_covers_all_samples_and_reshuffles():
    rollout = RolloutBuffer(7, 1)
    rollout.size = 7  # Index-only contract; no depth allocation needed.
    torch.manual_seed(42)
    first = list(rollout.minibatches(3))
    second = list(rollout.minibatches(3))
    assert [len(batch) for batch in first] == [3, 2, 2]
    np.testing.assert_array_equal(np.sort(np.concatenate(first)), np.arange(7))
    np.testing.assert_array_equal(np.sort(np.concatenate(second)), np.arange(7))
    assert not np.array_equal(np.concatenate(first), np.concatenate(second))


def test_export_is_student_only_and_matches_online(spec, teacher, observations, tmp_path):
    student = StudentWeaveActor(spec)
    student.initialize_from_teacher(teacher)
    path = tmp_path / "student.pt"
    export_student(student, str(path))
    inference = torch.jit.load(str(path))
    depth = torch.full((4, 1, 128, 128), 2.0)
    actual = inference(*(observations[g] for g in VECTOR_GROUPS), depth)
    torch.testing.assert_close(actual, student(observations))


@pytest.mark.parametrize("rank", [0, 1])
def test_torchrun_binds_local_device_before_simulation_app_starts(monkeypatch, rank):
    import sys
    import types
    from pathlib import Path

    observed = {}

    class Launcher:
        @staticmethod
        def add_app_launcher_args(parser):
            parser.add_argument("--device", default="cuda:0")

        def __init__(self, args):
            observed.update(device=args.device, distributed=args.distributed, cameras=args.enable_cameras)
            self.app = object()

    parent = types.ModuleType("isaaclab")
    parent.__path__ = []
    module = types.ModuleType("isaaclab.app")
    module.AppLauncher = Launcher
    monkeypatch.setitem(sys.modules, "isaaclab", parent)
    monkeypatch.setitem(sys.modules, "isaaclab.app", module)
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", str(rank))
    monkeypatch.setattr(sys, "argv", ["train.py", "--device", "cuda:0"])
    path = Path(__file__).resolve().parents[4] / "scripts/dagger/train.py"
    prefix = path.read_text().split("import g1_hoi_learning.tasks", 1)[0]
    exec(compile(prefix, str(path), "exec"), {})
    assert observed == {"device": f"cuda:{rank}", "distributed": True, "cameras": True}
