"""Terminal target selection, actor isolation and goal PPO/checkpoint contracts."""

import ast
import copy
import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from conftest import PACKAGE, make_policy
from g1_hoi_learning.algorithms.dagger.networks import (
    GOAL_VECTOR_GROUPS,
    StudentWeaveActor,
    export_student,
    student_actor_spec,
)
from g1_hoi_learning.algorithms.dagger.runner import SCHEMA, DaggerRunner
from g1_hoi_learning.algorithms.distillation.algorithm import MuonPPODistill
from g1_hoi_learning.algorithms.perturbations import offset_object_pose
from tensordict import TensorClass
from test_ppo import FakeEnv, collect, config


def _load_definitions(path, names, namespace):
    """Execute production definitions without importing simulation/UI-only dependencies."""
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


@pytest.fixture
def command():
    namespace = _load_definitions(
        PACKAGE / "tasks/hoi/mdp/commands.py",
        {"MotionData", "MotionLoader", "MotionCommand"},
        dict(
            torch=torch,
            np=np,
            Sequence=Sequence,
            TensorClass=TensorClass,
            CommandTerm=object,
            offset_object_pose=offset_object_pose,
        ),
    )
    loader = namespace["MotionLoader"].__new__(namespace["MotionLoader"])
    loader.device = torch.device("cpu")
    loader.clip_lengths = torch.tensor([4, 6, 3])
    loader.clip_starts = torch.tensor([0, 4, 10, 13])
    loader.object_nclips = torch.tensor([2, 1])
    loader._clips_by_object = torch.tensor([[0, 1], [2, 0]])
    frame = torch.arange(13).float()
    quats = torch.tensor([1.0, 0, 0, 0]).expand(13, 54, 4).clone()
    loader.buffer = namespace["MotionData"](
        motion_id=torch.repeat_interleave(torch.arange(3), loader.clip_lengths),
        step=torch.cat([torch.arange(length) for length in loader.clip_lengths]),
        joint_pos=frame[:, None].expand(-1, 53).clone() / 100,
        joint_vel=torch.zeros(13, 53),
        body_pos_w=frame[:, None, None].expand(-1, 54, 3).clone(),
        body_quat_w=quats,
        body_lin_vel_w=torch.zeros(13, 54, 3),
        body_ang_vel_w=torch.zeros(13, 54, 3),
        object_pos_w=frame[:, None].expand(-1, 3).clone() + 0.5,
        object_quat_w=quats[:, 0].clone(),
        object_lin_vel_w=torch.zeros(13, 3),
        object_ang_vel_w=torch.zeros(13, 3),
        contact_label=torch.ones(13, 54),
        batch_size=[13],
    )
    cmd = namespace["MotionCommand"].__new__(namespace["MotionCommand"])
    cmd.motion = loader
    cmd.num_envs, cmd.device, cmd.anchor_index = 3, "cpu", 0
    cmd.env_object, cmd.env_clip = torch.tensor([0, 1, 0]), torch.tensor([0, 2, 1])
    cmd.time_steps = torch.tensor([1, 0, 3])
    cmd.future_offsets = torch.tensor([0, 2])
    cmd.cfg = SimpleNamespace(eval_mode=False, rsi=True)
    origins = torch.tensor([[100.0, 0, 0], [0, 100.0, 0], [0, 0, 100.0]])
    cmd._env = SimpleNamespace(scene=SimpleNamespace(env_origins=origins))
    actual = SimpleNamespace(
        body_pos_w=origins[:, None].expand(-1, 54, -1).clone(),
        body_quat_w=quats[:3].clone(),
        soft_joint_pos_limits=torch.tensor([-1.0, 1.0]).expand(3, 53, 2),
    )

    def write_root(state, env_ids):
        actual.body_pos_w[env_ids, 0] = state[:, :3]
        actual.body_quat_w[env_ids, 0] = state[:, 3:7]

    cmd.robot = SimpleNamespace(
        data=actual, write_root_state_to_sim=write_root, write_joint_state_to_sim=lambda *args, **kwargs: None
    )
    cmd.object = SimpleNamespace(write_root_state_to_sim=lambda *args, **kwargs: None)
    cmd._refresh_goal()
    cmd._refresh_caches()
    return cmd


def test_terminal_goals_use_each_clips_end_and_ignore_rsi_time(command):
    torch.testing.assert_close(command.goal_joint_pos[:, 0], torch.tensor([0.03, 0.12, 0.09]))
    origins = command._env.scene.env_origins
    torch.testing.assert_close(command.goal_anchor_pos_w - origins, torch.tensor([[3.0] * 3, [12.0] * 3, [9.0] * 3]))
    before = command.goal_obj_pos_w.clone()
    command.time_steps += 1
    command._refresh_caches()
    torch.testing.assert_close(command.goal_obj_pos_w, before)
    torch.testing.assert_close(command.joint_pos[:, 0], torch.tensor([0.02, 0.11, 0.08]))
    # Goal storage owns its data rather than aliasing the reference buffer.
    command._goal_joint_pos.zero_()
    assert command.motion.buffer.joint_pos[3, 0] == pytest.approx(0.03)


def test_partial_resampling_and_training_rollover_refresh_only_selected_goals(command):
    command.motion.sample_clip = lambda objects: torch.ones(len(objects), dtype=torch.long)
    before = command.goal_joint_pos.clone()
    command._resample_command(torch.tensor([0]))
    assert command.env_clip.tolist() == [1, 2, 1]
    torch.testing.assert_close(command.goal_joint_pos[0], torch.full((53,), 0.09))
    torch.testing.assert_close(command.goal_joint_pos[1:], before[1:])
    command.motion.sample_clip = lambda objects: torch.zeros(len(objects), dtype=torch.long)
    command.time_steps = torch.tensor([5, 0, 0])
    command._update_command()
    assert command.env_clip.tolist() == [0, 2, 1]
    torch.testing.assert_close(command.goal_joint_pos[0], torch.full((53,), 0.03))
    torch.testing.assert_close(command.goal_joint_pos[1:], before[1:])


def test_eval_reassignment_and_initial_perturbation_preserve_reference_goal(command):
    command.cfg.eval_mode = True
    command._eval_env_clip = torch.tensor([1, 0, 2])
    command._eval_object_pose_offsets = torch.tensor([[1.0, 2.0, 0.5]]).expand(3, -1)
    states = []
    command.object.write_root_state_to_sim = lambda state, env_ids: states.append(state.clone())
    command._resample_command(torch.arange(3))
    assert command.time_steps.tolist() == [0, 0, 0]
    expected = torch.tensor([[9.5] * 3, [3.5] * 3, [12.5] * 3]) + command._env.scene.env_origins
    torch.testing.assert_close(command.goal_obj_pos_w, expected)
    assert not torch.equal(states[0][:, :3], command.ref_obj_pos_w)


def test_goal_observation_reanchors_fixed_target_and_preserves_rot6d_layout(command):
    # IsaacLab's math module is simulator independent; load the installed source directly.
    specs = (finder.find_spec("isaaclab", None) for finder in sys.meta_path)
    package_spec = next(spec for spec in specs if spec is not None and spec.origin is not None)
    root = Path(package_spec.origin).parent
    spec = importlib.util.spec_from_file_location("goal_test_math", root / "utils/math.py")
    math_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(math_module)
    observations = _load_definitions(
        PACKAGE / "tasks/hoi/mdp/observations.py",
        {"goal_robot_pose", "goal_object_pose"},
        dict(
            torch=torch,
            matrix_from_quat=math_module.matrix_from_quat,
            subtract_frame_transforms=math_module.subtract_frame_transforms,
        ),
    )
    env = SimpleNamespace(num_envs=3, command_manager=SimpleNamespace(get_term=lambda name: command))
    robot = observations["goal_robot_pose"](env, "motion")
    obj = observations["goal_object_pose"](env, "motion")
    assert robot.shape == (3, 62) and obj.shape == (3, 9)
    torch.testing.assert_close(robot[:, :53], command.goal_joint_pos)
    torch.testing.assert_close(obj[0], torch.tensor([3.5, 3.5, 3.5, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]))
    world_goal = command.goal_obj_pos_w.clone()
    command.robot.data.body_pos_w[:, 0, 0] += 1
    half = torch.tensor(torch.pi / 4)
    command.robot.data.body_quat_w[:, 0] = torch.tensor([half.cos(), 0.0, 0.0, half.sin()])
    changed = observations["goal_object_pose"](env, "motion")
    torch.testing.assert_close(
        changed[0], torch.tensor([3.5, -2.5, 3.5, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0]), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(command.goal_obj_pos_w, world_goal)


@pytest.fixture
def goal_setup(spec, teacher, observations):
    obs = {**observations, "goal_robot_pose": torch.randn(4, 62), "goal_object_pose": torch.randn(4, 9)}
    goal_spec = student_actor_spec(spec, {group: obs[group].shape[-1] for group in GOAL_VECTOR_GROUPS})
    student = StudentWeaveActor(goal_spec)
    student.initialize_from_teacher(teacher)
    return goal_spec, student, obs


def test_goal_encoders_transfer_live_object_proprio_and_backbone(teacher, goal_setup):
    spec, student, obs = goal_setup
    assert student.vector_encoders["goal_robot_pose"][0].in_features == 62
    assert student.vector_encoders["goal_object_pose"][0].in_features == 9
    assert isinstance(student.vector_normalizers["goal_robot_pose"], torch.nn.Identity)
    assert isinstance(student.vector_normalizers["goal_object_pose"], torch.nn.Identity)
    for name, value in student.backbone.state_dict().items():
        torch.testing.assert_close(value, teacher.actor[1].state_dict()[name], rtol=0, atol=0)
    for name, value in student.vector_encoders["robot_proprio"].state_dict().items():
        torch.testing.assert_close(value, teacher.actor[0].encoders[3].state_dict()[name], rtol=0, atol=0)
    width = spec["group_dims"]["robot_proprio"]
    torch.testing.assert_close(student.vector_normalizers["robot_proprio"]._mean, teacher.normalizer._mean[:, -width:])
    means = student(obs)
    changed = {key: value for key, value in obs.items() if key in student.observation_groups}
    torch.testing.assert_close(student(changed), means, rtol=0, atol=0)
    for group in ("ref_motion_body", "ref_motion_object", "robot_privileged"):
        changed[group] = torch.full_like(obs[group], float("nan"))
    torch.testing.assert_close(student(changed), means, rtol=0, atol=0)
    teacher_mean = teacher(obs)
    changed = dict(obs, goal_robot_pose=obs["goal_robot_pose"] + 100, goal_object_pose=obs["goal_object_pose"] - 100)
    torch.testing.assert_close(teacher(changed), teacher_mean, rtol=0, atol=0)
    assert not torch.equal(student(changed), means)


def test_object_state_normalization_and_encoder_match_teacher_without_parameter_sharing(teacher, goal_setup):
    _, student, obs = goal_setup
    captured = {}
    student_encoder = student.vector_encoders["object_state"]
    teacher_encoder = teacher.actor[0].encoders[2]
    for key, value in student_encoder.state_dict().items():
        torch.testing.assert_close(value, teacher_encoder.state_dict()[key], rtol=0, atol=0)
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(student_encoder.parameters(), teacher_encoder.parameters()))
    hooks = [
        student_encoder.register_forward_pre_hook(lambda module, args: captured.update(student=args[0].clone())),
        teacher_encoder.register_forward_pre_hook(lambda module, args: captured.update(teacher=args[0].clone())),
    ]
    try:
        student(obs)
        teacher(obs)
    finally:
        for hook in hooks:
            hook.remove()
    torch.testing.assert_close(captured["student"], captured["teacher"], rtol=0, atol=0)
    goal = obs["goal_object_pose"].clone()
    before = student(obs)
    obs["object_state"] += 1
    assert not torch.equal(student(obs), before)
    torch.testing.assert_close(obs["goal_object_pose"], goal, rtol=0, atol=0)
    with torch.no_grad():
        next(student_encoder.parameters()).add_(1)
    assert any(not torch.equal(a, b) for a, b in zip(student_encoder.parameters(), teacher_encoder.parameters()))


def test_goal_ppo_storage_gradients_and_teacher_isolation(spec, teacher, goal_setup):
    _, student, obs = goal_setup
    policy = make_policy(spec, teacher, student)
    algorithm = MuonPPODistill(policy, config())
    teacher_before = copy.deepcopy(teacher.state_dict())
    norms_before = copy.deepcopy(student.vector_normalizers.state_dict())
    collect(algorithm, obs)
    assert set(student.observation_groups).issubset(algorithm.storage.obs_keys)
    np.testing.assert_array_equal(algorithm.storage.arrays["goal_robot_pose"][:4], obs["goal_robot_pose"].numpy())
    before = [copy.deepcopy(student.vector_encoders[group].state_dict()) for group in GOAL_VECTOR_GROUPS]
    metrics = algorithm.update()
    assert metrics["optimizer_updates"] == 4
    for group, previous in zip(GOAL_VECTOR_GROUPS, before):
        assert all(parameter.grad is not None for parameter in student.vector_encoders[group].parameters())
        assert any(
            not torch.equal(value, previous[name])
            for name, value in student.vector_encoders[group].state_dict().items()
        )
    for name, value in teacher.state_dict().items():
        torch.testing.assert_close(value, teacher_before[name], rtol=0, atol=0)
    for name, value in student.vector_normalizers.state_dict().items():
        torch.testing.assert_close(value, norms_before[name], rtol=0, atol=0)


def test_goal_runner_resume_export_and_legacy_contract_rejection(spec, teacher, goal_setup, tmp_path):
    goal_spec, student, obs = goal_setup
    policy = make_policy(spec, teacher, student)
    runner = DaggerRunner(
        FakeEnv(obs), teacher, student, config(), str(tmp_path), {"sha256": "fixture"}, {}, {}, policy=policy
    )
    runner.learn(obs, max_iterations=1)
    state = torch.load(tmp_path / "model_1.pt", weights_only=False)
    assert state["schema"] == SCHEMA and state["actor_spec"] == goal_spec
    assert state["teacher_actor_spec"] == teacher.spec
    assert state["critic_spec"] == policy.critic_spec
    runner.load(str(tmp_path / "model_1.pt"))
    assert runner.iteration == runner.algorithm._it == 1
    inference = torch.jit.load(str(tmp_path / "student_policy.pt"))
    torch.testing.assert_close(inference(*(obs[g] for g in GOAL_VECTOR_GROUPS)), student(obs))
    legacy = copy.deepcopy(state)
    legacy["schema"], legacy["actor_spec"] = "weave-depth-dagger-v3", teacher.spec
    torch.save(legacy, tmp_path / "legacy.pt")
    with pytest.raises(ValueError, match="Legacy"):
        runner.load(str(tmp_path / "legacy.pt"))
    with pytest.raises(ValueError, match="Incompatible student"):
        runner.load_student(str(tmp_path / "legacy.pt"))


def test_goal_export_accepts_only_student_inputs(goal_setup, tmp_path):
    _, student, obs = goal_setup
    export_student(student, str(tmp_path / "goal.pt"))
    exported = torch.jit.load(str(tmp_path / "goal.pt"))
    torch.testing.assert_close(exported(*(obs[g] for g in GOAL_VECTOR_GROUPS)), student(obs))
