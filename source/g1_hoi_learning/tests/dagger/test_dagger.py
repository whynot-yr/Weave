"""Critical contracts: transfer, frozen-backbone gradients, state alignment, replay, and resume."""

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
from g1_hoi_learning.algorithms.dagger.runner import DaggerRunner, choose_actions
from g1_hoi_learning.algorithms.dagger.storage import ReservoirBuffer, capture_samples
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


def test_cnn_only_then_joint_update_preserves_adam(spec, teacher, observations):
    student = StudentWeaveActor(spec)
    student.initialize_from_teacher(teacher)
    cfg = DaggerTrainConfig(batch_size=4, updates_per_iteration=1, cnn_lr=1e-3)
    algorithm = DaggerAlgorithm(student, cfg)
    algorithm.set_stage("visual_warmup")
    current = capture_samples(observations, torch.ones(4, 3), {})
    buffer = ReservoirBuffer(10, 42)
    cnn_before = next(student.depth_encoder.parameters()).detach().clone()
    transferred_before = copy.deepcopy(student.backbone.state_dict())
    algorithm.update(current, buffer, "cpu")
    assert not torch.equal(cnn_before, next(student.depth_encoder.parameters()))
    assert next(student.depth_encoder.parameters()).grad.abs().sum() > 0
    for key, value in student.backbone.state_dict().items():
        torch.testing.assert_close(value, transferred_before[key], rtol=0, atol=0)
    cnn_param = next(student.depth_encoder.parameters())
    step_before = algorithm.optimizer.state[cnn_param]["step"].clone()
    algorithm.set_stage("joint_warmup")
    torch.testing.assert_close(algorithm.optimizer.state[cnn_param]["step"], step_before)
    algorithm.update(current, buffer, "cpu")
    assert all(p.requires_grad for p in student.parameters())
    assert next(student.backbone.parameters()).grad.abs().sum() > 0
    assert all(not norm.training for norm in student.vector_normalizers.values())


def test_capture_owns_pre_step_data_and_reservoir_resume(observations):
    target = torch.arange(12).float().reshape(4, 3)
    expected = target.clone()
    samples = capture_samples(observations, target, {"episode_step": torch.arange(4)})
    target.zero_()
    observations["head_depth"].zero_()
    np.testing.assert_array_equal(samples["teacher_action"], expected.numpy())
    assert samples["head_depth"].min() == 2000
    replay = ReservoirBuffer(3, 42)
    replay.add(samples)
    assert replay.seen == 4 and replay.size == 3
    restored = ReservoirBuffer(3, 1)
    restored.load_state_dict(replay.state_dict())
    replay.add(samples)
    restored.add(samples)
    for key in samples:
        np.testing.assert_array_equal(restored.arrays[key], replay.arrays[key])
    obs, actions = replay.sample(samples, 10, "cpu")
    assert obs["head_depth"].shape == (10, 1, 128, 128) and actions.shape == (10, 3)
    assert obs["head_depth"].min().item() == pytest.approx(0.2)


def test_stage_boundaries_and_action_selection():
    cfg = DaggerTrainConfig(
        visual_warmup_iterations=2, joint_warmup_iterations=2, decay_iterations=2, student_iterations=1
    )
    assert [cfg.stage(i) for i in range(7)] == [
        ("visual_warmup", 1),
        ("visual_warmup", 1),
        ("joint_warmup", 1),
        ("joint_warmup", 1),
        ("dagger_decay", 1),
        ("dagger_decay", 0.5),
        ("student_only", 0),
    ]
    teacher, student = torch.ones(20, 3), torch.zeros(20, 3)
    assert torch.equal(choose_actions(teacher, student, 1)[0], teacher)
    assert torch.equal(choose_actions(teacher, student, 0)[0], student)
    actions, mask = choose_actions(teacher, student, 0.5)
    assert torch.equal(actions, mask[:, None].expand_as(actions).float())


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


def test_runner_same_state_labels_and_full_checkpoint_resume(spec, teacher, observations, tmp_path):
    student = StudentWeaveActor(spec)
    student.initialize_from_teacher(teacher)
    env = FakeEnv(observations)
    cfg = DaggerTrainConfig(
        num_envs=4,
        rollout_steps=1,
        batch_size=4,
        updates_per_iteration=1,
        buffer_capacity=10,
        visual_warmup_iterations=1,
        joint_warmup_iterations=1,
        decay_iterations=1,
        student_iterations=1,
    )
    info, contract = {"sha256": "fixture"}, {"test": True}
    expected_label = teacher(observations).detach().clone()
    expected_obs = observations["robot_proprio"].clone()
    runner = DaggerRunner(env, teacher, student, cfg, str(tmp_path), info, contract, {})
    runner.learn(observations, max_iterations=1)
    np.testing.assert_array_equal(runner.buffer.arrays["teacher_action"][:4], expected_label.numpy())
    np.testing.assert_array_equal(runner.buffer.arrays["robot_proprio"][:4], expected_obs.numpy())
    torch.testing.assert_close(env.actions[0], expected_label)
    replacement = DaggerRunner(env, teacher, StudentWeaveActor(spec), cfg, str(tmp_path), info, contract, {})
    replacement.load(str(tmp_path / "model_1.pt"))
    assert replacement.iteration == 1 and replacement.buffer.size == 4
    assert len(replacement.algorithm.optimizer.param_groups) == 2  # transition into joint warmup
    assert replacement.algorithm.optimizer.state
    replacement.learn(observations, max_iterations=1)
    third = DaggerRunner(env, teacher, StudentWeaveActor(spec), cfg, str(tmp_path), info, contract, {})
    third.load(str(tmp_path / "model_2.pt"))
    assert third.algorithm.joint_enabled and third.buffer.seen == 8
    for key, value in third.student.state_dict().items():
        torch.testing.assert_close(value, replacement.student.state_dict()[key], rtol=0, atol=0)


def test_runner_all_stages_include_student_execution(spec, teacher, observations, tmp_path):
    student = StudentWeaveActor(spec)
    student.initialize_from_teacher(teacher)
    teacher_before = copy.deepcopy(teacher.state_dict())
    cfg = DaggerTrainConfig(
        num_envs=4,
        rollout_steps=1,
        batch_size=4,
        updates_per_iteration=1,
        buffer_capacity=24,
        visual_warmup_iterations=1,
        joint_warmup_iterations=1,
        decay_iterations=2,
        student_iterations=1,
    )
    runner = DaggerRunner(FakeEnv(observations), teacher, student, cfg, str(tmp_path), {}, {}, {})
    runner.learn(observations)
    assert runner.iteration == 5 and runner.buffer.seen == 20
    np.testing.assert_array_equal(runner.buffer.arrays["stage"][:20], np.repeat([0, 1, 2, 2, 3], 4))
    assert runner.buffer.arrays["executed_teacher"][:12].all()
    assert not runner.buffer.arrays["executed_teacher"][16:20].any()
    for key, value in teacher.state_dict().items():
        torch.testing.assert_close(value, teacher_before[key], rtol=0, atol=0)
