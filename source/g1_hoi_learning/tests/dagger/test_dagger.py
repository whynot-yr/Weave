"""Online action distillation: parameter freezing, full epochs, accumulation, state alignment and resume."""

import copy

import numpy as np
import pytest
import torch
from g1_hoi_learning.algorithms.dagger.algorithm import DaggerAlgorithm
from g1_hoi_learning.algorithms.dagger.config import DaggerTrainConfig
from g1_hoi_learning.algorithms.dagger.evaluation import assign_eval_clips, summarize
from g1_hoi_learning.algorithms.dagger.io import read_yaml
from g1_hoi_learning.algorithms.dagger.networks import (
    TEACHER_GROUPS,
    VECTOR_GROUPS,
    StudentWeaveActor,
    TeacherWeaveActor,
    export_student,
)
from g1_hoi_learning.algorithms.dagger.runner import SCHEMA, DaggerRunner
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


def small_config(**overrides):
    settings = dict(
        num_envs=4,
        rollout_steps=1,
        num_learning_epochs=2,
        num_mini_batches=2,
        micro_batch_size=2,
        num_iterations=4,
    )
    settings.update(overrides)
    return DaggerTrainConfig(**settings)


def make_rollout(observations, target, steps=1):
    rollout = RolloutBuffer(len(target), steps)
    for _ in range(steps):
        rollout.add(observations, target, {"episode_step": torch.arange(len(target))})
    return rollout


def test_joint_cnn_simba_updates_leave_teacher_vector_branches_and_norms_fixed(spec, teacher, observations):
    student = StudentWeaveActor(spec)
    student.initialize_from_teacher(teacher)
    teacher_before = copy.deepcopy(teacher.state_dict())
    vector_before = copy.deepcopy(student.vector_encoders.state_dict())
    norm_before = copy.deepcopy(student.vector_normalizers.state_dict())
    cnn_before = copy.deepcopy(student.depth_encoder.state_dict())
    simba_before = copy.deepcopy(student.backbone.state_dict())
    algorithm = DaggerAlgorithm(student, small_config(cnn_lr=1e-3, transferred_lr=1e-3))
    rollout = make_rollout(observations, torch.ones(4, spec["num_actions"]))
    result = algorithm.update(rollout, "cpu")
    assert result["optimizer_updates"] == 4
    assert any(not torch.equal(value, student.depth_encoder.state_dict()[key]) for key, value in cnn_before.items())
    assert any(not torch.equal(value, student.backbone.state_dict()[key]) for key, value in simba_before.items())
    assert all(p.grad is not None for p in algorithm.trainable_parameters)
    assert all(not p.requires_grad and p.grad is None for p in student.vector_encoders.parameters())
    assert all(not norm.training for norm in student.vector_normalizers.values())
    assert not student.vector_encoders.training
    for original, current in (
        (teacher_before, teacher.state_dict()),
        (vector_before, student.vector_encoders.state_dict()),
        (norm_before, student.vector_normalizers.state_dict()),
    ):
        for key in original:
            torch.testing.assert_close(original[key], current[key], rtol=0, atol=0)


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


@pytest.mark.parametrize(
    "settings",
    [
        {"num_mini_batches": 65537},
        {"num_learning_epochs": 0},
        {"micro_batch_size": 0},
        {"num_iterations": 0},
        {"cnn_lr": float("nan")},
        {"weight_decay": -1},
        {"keep_last_checkpoints": 0},
        {"keep_last_checkpoints": -1},
        {"keep_last_checkpoints": True},
        {"keep_last_checkpoints": 1.5},
    ],
)
def test_invalid_training_settings(settings):
    with pytest.raises(ValueError):
        DaggerTrainConfig(**settings)


def test_approved_config_performs_64_updates_and_transfers_only_microbatches(spec, teacher, observations):
    # Use a tiny encoder to exercise the full approved sample/step counts on CPU.
    cfg = DaggerTrainConfig()
    assert (cfg.num_envs, cfg.rollout_steps, cfg.num_learning_epochs, cfg.num_mini_batches, cfg.micro_batch_size) == (
        4096,
        16,
        2,
        32,
        128,
    )
    student = StudentWeaveActor(spec)
    student.initialize_from_teacher(teacher)
    student.depth_encoder = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(1, spec["latent_dim"]))
    large_obs = {key: value.repeat(1024, 1) for key, value in observations.items() if key != "head_depth"}
    large_obs["head_depth"] = torch.full((4096, 1, 1, 1), 0.2)
    rollout = make_rollout(large_obs, torch.zeros(4096, spec["num_actions"]), cfg.rollout_steps)
    algorithm = DaggerAlgorithm(student, cfg)
    batches = []
    batch = rollout.batch

    def tracked_batch(indices, device):
        batches.append(indices.copy())
        assert len(indices) <= 128
        return batch(indices, device)

    rollout.batch = tracked_batch
    result = algorithm.update(rollout, "cpu")
    assert result["optimizer_updates"] == 64
    assert result["samples_trained"] == 131072
    assert len(batches) == 1024  # 16 backwards per optimizer step.
    for epoch in range(2):
        used = np.concatenate(batches[epoch * 512 : (epoch + 1) * 512])
        np.testing.assert_array_equal(np.sort(used), np.arange(65536))
    assert all(algorithm.optimizer.state[p]["step"] == 64 for p in algorithm.trainable_parameters)


def test_export_is_student_only_and_matches_online(spec, teacher, observations, tmp_path):
    student = StudentWeaveActor(spec)
    student.initialize_from_teacher(teacher)
    path = tmp_path / "student.pt"
    export_student(student, str(path))
    inference = torch.jit.load(str(path))
    depth = torch.full((4, 1, 128, 128), 2.0)
    actual = inference(*(observations[g] for g in VECTOR_GROUPS), depth)
    torch.testing.assert_close(actual, student(observations))


class FakeEnv:
    """Overwrites observation buffers during step to expose state/action alignment bugs."""

    def __init__(self, observations):
        from types import SimpleNamespace

        self.num_envs = 4
        self.obs = observations
        self.actions = []
        command = SimpleNamespace(env_clip=torch.arange(4), time_steps=torch.zeros(4, dtype=torch.long))
        self.unwrapped = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: command))

    def step(self, actions):
        self.actions.append(actions.clone())
        for value in self.obs.values():
            value.add_(0.01)
        self.unwrapped.command_manager.get_term("motion").time_steps += 1
        return self.obs, torch.ones(4), torch.zeros(4, dtype=torch.long), {}


def test_runner_student_execution_same_state_labels_and_full_checkpoint_resume(spec, teacher, observations, tmp_path):
    student = StudentWeaveActor(spec)
    student.initialize_from_teacher(teacher)
    env = FakeEnv(observations)
    cfg = small_config()
    info, contract = {"sha256": "fixture"}, {"test": True}
    expected_label = teacher(observations).detach().clone()
    expected_action = student(observations).detach().clone()
    expected_obs = observations["robot_proprio"].clone()
    runner = DaggerRunner(env, teacher, student, cfg, str(tmp_path), info, contract, {})
    runner.learn(observations, max_iterations=1)
    np.testing.assert_array_equal(runner.rollout.arrays["teacher_action"], expected_label.numpy())
    np.testing.assert_array_equal(runner.rollout.arrays["robot_proprio"], expected_obs.numpy())
    torch.testing.assert_close(env.actions[0], expected_action)
    checkpoint = torch.load(tmp_path / "model_1.pt", weights_only=False)
    assert checkpoint["schema"] == SCHEMA
    assert not {"replay", "stage", "beta", "joint_enabled"} & checkpoint.keys()
    assert checkpoint["total_optimizer_updates"] == 4
    replacement = DaggerRunner(env, teacher, StudentWeaveActor(spec), cfg, str(tmp_path), info, contract, {})
    replacement.load(str(tmp_path / "model_1.pt"))
    assert replacement.iteration == 1 and replacement.rollout.size == 0
    assert replacement.total_transitions == 4 and replacement.total_optimizer_updates == 4
    assert len(replacement.algorithm.optimizer.param_groups) == 2
    assert replacement.algorithm.optimizer.state
    expected_permutation = torch.randperm(8)
    replacement.load(str(tmp_path / "model_1.pt"))
    torch.testing.assert_close(torch.randperm(8), expected_permutation)
    replacement.learn(observations, max_iterations=1)
    third = DaggerRunner(env, teacher, StudentWeaveActor(spec), cfg, str(tmp_path), info, contract, {})
    third.load(str(tmp_path / "model_2.pt"))
    assert third.total_transitions == 8 and third.total_optimizer_updates == 8
    for key, value in third.student.state_dict().items():
        torch.testing.assert_close(value, replacement.student.state_dict()[key], rtol=0, atol=0)


def test_runner_continues_across_windows_updates_before_next_sampling_and_handles_episode_resets(
    spec, teacher, observations, tmp_path
):
    student = StudentWeaveActor(spec)
    student.initialize_from_teacher(teacher)
    env = FakeEnv(observations)
    cfg = small_config(rollout_steps=2, num_iterations=3, save_interval=1, keep_last_checkpoints=2)
    runner = DaggerRunner(env, teacher, student, cfg, str(tmp_path), {"sha256": "fixture"}, {}, {})
    original_step = env.step
    original_update = runner.algorithm.update
    updates = []
    step_ids = []
    initial_weights = copy.deepcopy(student.state_dict())
    teacher_before = copy.deepcopy(teacher.state_dict())

    def step(actions):
        torch.testing.assert_close(actions, student(observations), rtol=0, atol=0)
        assert not student.training
        result = original_step(actions)
        if len(env.actions) == 2:
            result[2][0] = 1
        return result

    def update(rollout, device):
        assert len(env.actions) == (len(updates) + 1) * 2
        step_ids.append(rollout.arrays["episode_step"].copy())
        updates.append(original_update(rollout, device))
        assert any(not torch.equal(value, student.state_dict()[key]) for key, value in initial_weights.items())
        return updates[-1]

    env.step = step
    runner.algorithm.update = update
    runner.learn(observations)
    assert runner.iteration == 3 and len(env.actions) == 6
    assert runner.total_transitions == 24 and runner.rollout.size == 8
    assert runner.total_optimizer_updates == 12
    np.testing.assert_array_equal(step_ids[0], [0, 0, 0, 0, 1, 1, 1, 1])
    np.testing.assert_array_equal(step_ids[1], [0, 2, 2, 2, 1, 3, 3, 3])
    np.testing.assert_array_equal(runner.rollout.arrays["ref_step"], [4] * 4 + [5] * 4)
    for key, value in teacher.state_dict().items():
        torch.testing.assert_close(value, teacher_before[key], rtol=0, atol=0)
    assert sorted(path.name for path in tmp_path.glob("model_*.pt")) == ["model_2.pt", "model_3.pt"]
    weights = torch.load(tmp_path / "student_weights.pt", weights_only=False)
    assert weights["kind"] == "student_weights"
    assert "replay" not in weights and "optimizer_state_dict" not in weights
    with pytest.raises(ValueError, match="full model"):
        runner.load(str(tmp_path / "student_weights.pt"))


@pytest.mark.parametrize("micro_size", [2, 4, 128])
def test_microbatch_accumulation_matches_full_batch_including_uneven_batches(spec, teacher, observations, micro_size):
    student = StudentWeaveActor(spec)
    student.initialize_from_teacher(teacher)
    accumulated = copy.deepcopy(student)
    settings = dict(rollout_steps=3, num_mini_batches=5, cnn_lr=1e-3, transferred_lr=1e-3)
    full = DaggerAlgorithm(student, small_config(**settings, micro_batch_size=None))
    micro = DaggerAlgorithm(accumulated, small_config(**settings, micro_batch_size=micro_size))
    rollout = make_rollout(observations, torch.randn(4, spec["num_actions"]), 3)
    torch.manual_seed(42)
    full_result = full.update(rollout, "cpu")
    torch.manual_seed(42)
    micro_result = micro.update(rollout, "cpu")
    assert micro_result["train_action_mse"] == pytest.approx(full_result["train_action_mse"], rel=1e-5, abs=1e-7)
    for original, other in zip(student.parameters(), accumulated.parameters()):
        torch.testing.assert_close(original, other, rtol=1e-4, atol=2e-6)
        if original.requires_grad:
            assert micro.optimizer.state[other]["step"] == full.optimizer.state[original]["step"] == 10


def test_checkpoint_retention_preserves_old_snapshots_on_save_failure(
    spec, teacher, observations, tmp_path, monkeypatch
):
    runner = DaggerRunner(
        FakeEnv(observations),
        teacher,
        StudentWeaveActor(spec),
        small_config(keep_last_checkpoints=2),
        str(tmp_path),
        {},
        {},
        {},
    )
    runner.iteration = 9
    runner.save(tmp_path / "model_9.pt")
    runner.iteration = 10
    runner.save(tmp_path / "model_10.pt")
    unrelated = tmp_path / "model_notes.pt"
    unrelated.write_text("keep")
    runner.iteration = 100
    with monkeypatch.context() as patch:

        def failed_save(state, path):
            path.write_bytes(b"incomplete")
            raise OSError("simulated save failure")

        patch.setattr(torch, "save", failed_save)
        with pytest.raises(OSError, match="simulated save failure"):
            runner.save(tmp_path / "model_100.pt")
    assert (tmp_path / "model_9.pt").exists() and (tmp_path / "model_10.pt").exists()
    assert not (tmp_path / "model_100.pt").exists()
    runner.save(tmp_path / "model_100.pt")
    assert not (tmp_path / "model_9.pt").exists()
    assert (tmp_path / "model_10.pt").exists() and (tmp_path / "model_100.pt").exists()
    assert unrelated.read_text() == "keep"


def test_resume_allows_microbatch_change_but_preserves_training_contract(spec, teacher, observations, tmp_path):
    cfg = small_config(save_interval=100, keep_last_checkpoints=2)
    runner = DaggerRunner(
        FakeEnv(observations), teacher, StudentWeaveActor(spec), cfg, str(tmp_path), {"sha256": "fixture"}, {}, {}
    )
    runner.save(tmp_path / "model_0.pt")
    cfg.micro_batch_size = 64
    cfg.save_interval = 1000
    cfg.keep_last_checkpoints = None
    runner.load(str(tmp_path / "model_0.pt"))
    cfg.num_mini_batches = 1
    with pytest.raises(ValueError, match="original training"):
        runner.load(str(tmp_path / "model_0.pt"))


def test_keep_all_preserves_every_thousand_iteration_checkpoint(spec, teacher, observations, tmp_path):
    cfg = small_config()
    assert cfg.save_interval == 1000 and cfg.keep_last_checkpoints is None
    runner = DaggerRunner(FakeEnv(observations), teacher, StudentWeaveActor(spec), cfg, str(tmp_path), {}, {}, {})
    for iteration in range(1000, 11001, 1000):
        runner.iteration = iteration
        runner.save(tmp_path / f"model_{iteration}.pt")
    assert {path.name for path in tmp_path.glob("model_*.pt")} == {
        f"model_{iteration}.pt" for iteration in range(1000, 11001, 1000)
    }


def test_runner_saves_at_thousand_boundary_and_nonmilestone_end(spec, teacher, observations, tmp_path):
    cfg = small_config(num_iterations=11000)
    runner = DaggerRunner(
        FakeEnv(observations), teacher, StudentWeaveActor(spec), cfg, str(tmp_path), {"sha256": "fixture"}, {}, {}
    )
    runner.iteration = 999
    runner.learn(observations, max_iterations=2)
    assert runner.iteration == 1001
    assert {path.name for path in tmp_path.glob("model_*.pt")} == {"model_1000.pt", "model_1001.pt"}


@pytest.mark.parametrize("schema", ["weave-depth-dagger-v1", SCHEMA])
def test_warm_start_accepts_v1_v2_weights_without_restoring_training_state(
    spec, teacher, observations, tmp_path, schema
):
    runner = DaggerRunner(
        FakeEnv(observations),
        teacher,
        StudentWeaveActor(spec),
        small_config(),
        str(tmp_path),
        {"sha256": "fixture"},
        {},
        {},
    )
    runner.iteration = 3
    state = copy.deepcopy(runner._student_snapshot())
    state["schema"] = schema
    state["kind"] = "training"
    torch.save(state, tmp_path / "weights.pt")
    if schema != SCHEMA:
        with pytest.raises(ValueError, match="Legacy"):
            runner.load(str(tmp_path / "weights.pt"))
    runner.iteration = 0
    with torch.no_grad():
        next(runner.student.backbone.parameters()).add_(1)
    runner.load_student(str(tmp_path / "weights.pt"))
    assert runner.iteration == 0 and runner.total_optimizer_updates == 0
    assert not runner.algorithm.optimizer.state
    for key, value in runner.student.state_dict().items():
        torch.testing.assert_close(value, state["student_state_dict"][key], rtol=0, atol=0)


def test_partial_rollout_does_not_update_or_save(spec, teacher, observations, tmp_path):
    runner = DaggerRunner(
        FakeEnv(observations),
        teacher,
        StudentWeaveActor(spec),
        small_config(rollout_steps=2),
        str(tmp_path),
        {},
        {},
        {},
    )
    runner.learn(observations, is_running=lambda: len(runner.env.actions) < 1)
    assert runner.iteration == 0 and runner.total_optimizer_updates == 0
    assert not list(tmp_path.glob("*.pt"))


def test_earlier_single_rank_v2_checkpoint_still_resumes(spec, teacher, observations, tmp_path):
    runner = DaggerRunner(
        FakeEnv(observations),
        teacher,
        StudentWeaveActor(spec),
        small_config(),
        str(tmp_path),
        {"sha256": "fixture"},
        {},
        {},
    )
    runner.save(tmp_path / "new.pt")
    state = torch.load(tmp_path / "new.pt", weights_only=False)
    state.update(state.pop("rank_states")[0])
    state.pop("world_size")
    state.pop("distributed_state_version")
    state["rng"]["cuda"] = []
    torch.save(state, tmp_path / "earlier.pt")
    runner.load(str(tmp_path / "earlier.pt"))
    assert runner.iteration == 0 and runner.total_optimizer_updates == 0


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
