"""Historical PPO+BC equivalence, state alignment, GAE, gradients and checkpoints."""

import copy
import math
import subprocess
import types
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from conftest import make_policy
from g1_hoi_learning.algorithms.dagger.config import DaggerTrainConfig
from g1_hoi_learning.algorithms.dagger.runner import SCHEMA, STUDENT_SCHEMAS, DaggerRunner
from g1_hoi_learning.algorithms.distillation.algorithm import MuonPPODistill
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

HISTORICAL = "3904770889ee67b34113df01f3f5533add0a468b"


def config(**kwargs):
    values = dict(
        num_envs=4,
        rollout_steps=3,
        num_learning_epochs=2,
        num_mini_batches=2,
        micro_batch_size=3,
        num_iterations=4,
        schedule="fixed",
        bc_coef=0.5,
    )
    values.update(kwargs)
    return DaggerTrainConfig(**values)


def collect(algorithm, observations):
    obs = {k: v.clone() for k, v in observations.items()}
    cfg = algorithm.cfg
    algorithm.init_storage("rl", cfg.num_envs, cfg.rollout_steps, obs)
    with torch.no_grad():
        for step in range(cfg.rollout_steps):
            algorithm.act(obs)
            for key in obs:
                if key != "head_depth":
                    obs[key].add_(0.01)
            dones = torch.zeros(cfg.num_envs, dtype=torch.long)
            if step == 1:
                dones[0] = 1
            algorithm.process_env_step(obs, torch.arange(cfg.num_envs).float() + step, dones, {})
        algorithm.compute_returns(obs)
    return obs


class FakeEnv:
    """Mutates observations during step and mixes failures with timeouts."""

    def __init__(self, observations):
        self.obs = observations
        self.num_envs = len(observations["head_depth"])
        self.actions = []
        command = SimpleNamespace(
            env_clip=torch.arange(self.num_envs), time_steps=torch.zeros(self.num_envs, dtype=torch.long)
        )
        self.unwrapped = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: command))

    def step(self, actions):
        self.actions.append(actions.clone())
        for key, value in self.obs.items():
            if key != "head_depth":
                value.add_(0.01)
        self.unwrapped.command_manager.get_term("motion").time_steps += 1
        dones = torch.zeros(self.num_envs, dtype=torch.long)
        timeouts = torch.zeros(self.num_envs, dtype=torch.bool)
        if len(self.actions) % 2 == 0:
            dones[:2] = 1
            timeouts[1] = True
        return self.obs, torch.arange(self.num_envs).float() + 1, dones, {"time_outs": timeouts}


def make_runner(spec, teacher, obs, directory, cfg=None):
    policy = make_policy(spec, teacher)
    return DaggerRunner(
        FakeEnv(obs),
        teacher,
        policy.actor,
        cfg or config(),
        str(directory),
        {"sha256": "fixture"},
        {"test": True},
        {},
        policy=policy,
    )


def test_current_mean_actor_is_unchanged_and_only_critic_has_privileged_inputs(spec, teacher, observations):
    policy = make_policy(spec, teacher)
    expected = policy.actor(observations).detach()
    torch.testing.assert_close(policy.act_inference(observations), expected, rtol=0, atol=0)
    changed = {**observations, "object_state": observations["object_state"] + 100}
    torch.testing.assert_close(policy.act_inference(changed), expected, rtol=0, atol=0)
    assert not torch.equal(policy.evaluate(changed), policy.evaluate(observations))
    assert len([m for m in policy.actor.depth_encoder.modules() if isinstance(m, torch.nn.Conv2d)]) == 4
    torch.testing.assert_close(policy.std, torch.full((spec["num_actions"],), 0.2))
    for key, value in policy.critic.state_dict().items():
        torch.testing.assert_close(value, teacher.tracker_state[f"critic.{key}"], rtol=0, atol=0)


def test_gaussian_sampling_old_log_prob_and_pre_step_snapshot(spec, teacher, observations):
    policy = make_policy(spec, teacher)
    algorithm = MuonPPODistill(policy, config(rollout_steps=1))
    algorithm.init_storage("rl", 4, 1, observations)
    before = {k: v.clone() for k, v in observations.items()}
    with torch.no_grad():
        mean = policy.actor(observations)
        expected_log_prob = None
        torch.manual_seed(100)
        expected = torch.distributions.Normal(mean, policy.std).sample()
        torch.manual_seed(100)
        action = algorithm.act(observations, {"episode_step": torch.arange(4)})
        torch.testing.assert_close(action, expected)
        assert not torch.equal(action, mean)
        expected_log_prob = torch.distributions.Normal(mean, policy.std).log_prob(action).sum(-1)
        for key in observations:
            observations[key].zero_()
        algorithm.process_env_step(observations, torch.ones(4), torch.zeros(4), {})
        algorithm.compute_returns(observations)
    stored_obs, stored_action, _, _, _, log_prob, stored_mean, _, _, _ = algorithm.storage.batch(np.arange(4), "cpu")
    for key in before:
        torch.testing.assert_close(stored_obs[key], before[key], rtol=0, atol=0)
    torch.testing.assert_close(stored_action, expected)
    torch.testing.assert_close(log_prob.squeeze(-1), expected_log_prob)
    torch.testing.assert_close(stored_mean, mean)
    assert algorithm.storage.arrays["head_depth"].dtype == np.uint16


def test_gae_reuses_rsl_rl_and_handles_timeout_bootstrap(spec, teacher, observations):
    algorithm = MuonPPODistill(make_policy(spec, teacher), config())
    obs = collect(algorithm, observations)
    storage = algorithm.storage
    reference = RolloutStorage("rl", 4, 3, TensorDict(observations, batch_size=[4]), [3])
    for name in ("rewards", "dones", "values"):
        getattr(reference, name).copy_(getattr(storage, name))
    with torch.no_grad():
        last = algorithm.policy.evaluate(obs)
    reference.compute_returns(last, algorithm.gamma, algorithm.lam)
    torch.testing.assert_close(storage.returns, reference.returns)
    torch.testing.assert_close(storage.advantages, reference.advantages)

    algorithm.storage.clear()
    with torch.no_grad():
        algorithm.act(obs)
        values = algorithm.transition.values.clone().squeeze(-1)
        algorithm.process_env_step(
            obs,
            torch.ones(4),
            torch.tensor([1, 1, 0, 0]),
            {"time_outs": torch.tensor([False, True, False, False])},
        )
    torch.testing.assert_close(
        storage.rewards[0, :, 0], torch.ones(4) + algorithm.gamma * values * torch.tensor([0, 1, 0, 0])
    )
    assert storage.dones[0, :, 0].tolist() == [1, 1, 0, 0]


@pytest.mark.parametrize(
    "iteration,expected", [(0, 1), (1000, (1 + math.cos(math.pi / 6)) / 2), (3000, 0.5), (6000, 0), (9000, 0)]
)
def test_historical_bc_schedule(spec, teacher, iteration, expected):
    algorithm = MuonPPODistill(make_policy(spec, teacher), config(bc_coef=1))
    algorithm.set_iteration(iteration)
    assert algorithm.bc_coef == pytest.approx(expected)


def test_ppo_bc_updates_all_student_branches_and_critic_but_not_teacher_or_norms(spec, teacher, observations):
    policy = make_policy(spec, teacher)
    algorithm = MuonPPODistill(policy, config())
    teacher_before = copy.deepcopy(teacher.state_dict())
    norms_before = copy.deepcopy(policy.actor.vector_normalizers.state_dict())
    critic_norm_before = copy.deepcopy(policy.critic_obs_normalizer.state_dict())
    branches = [policy.actor.depth_encoder, policy.actor.vector_encoders, policy.actor.backbone, policy.critic]
    before = [copy.deepcopy(branch.state_dict()) for branch in branches]
    std_before = policy.std.detach().clone()
    collect(algorithm, observations)
    metrics = algorithm.update()
    assert metrics["optimizer_updates"] == 4 and metrics["samples_trained"] == 24
    assert metrics["bc_coef"] == 0.5 and metrics["ppo_coef"] == 0.5
    for branch, previous in zip(branches, before):
        assert any(not torch.equal(value, branch.state_dict()[key]) for key, value in previous.items())
        assert all(p.grad is not None for p in branch.parameters())
    assert not torch.equal(policy.std, std_before)
    for module, previous in (
        (teacher, teacher_before),
        (policy.actor.vector_normalizers, norms_before),
        (policy.critic_obs_normalizer, critic_norm_before),
    ):
        for key, value in previous.items():
            torch.testing.assert_close(module.state_dict()[key], value, rtol=0, atol=0)
    assert not teacher.training and not policy.critic_obs_normalizer.training
    assert all(not norm.training for norm in policy.actor.vector_normalizers.values())


@pytest.mark.parametrize("micro_size", [1, 3, 128])
def test_accumulated_ppo_matches_full_batch_including_uneven_batches(spec, teacher, observations, micro_size):
    full_policy = make_policy(spec, teacher)
    micro_policy = copy.deepcopy(full_policy)
    full = MuonPPODistill(full_policy, config(num_mini_batches=5, micro_batch_size=None))
    micro = MuonPPODistill(micro_policy, config(num_mini_batches=5, micro_batch_size=micro_size))
    # Isolate accumulation from Muon's bfloat16 orthogonalization rounding.
    # The separate historical-source comparison exercises the real Muon optimizer.
    full.optimizer = torch.optim.SGD(full.trainable_parameters, lr=full.cfg.learning_rate)
    micro.optimizer = torch.optim.SGD(micro.trainable_parameters, lr=micro.cfg.learning_rate)
    collect(full, observations)
    micro.storage = copy.deepcopy(full.storage)
    torch.manual_seed(82)
    expected = full.update()
    torch.manual_seed(82)
    actual = micro.update()
    assert actual["optimizer_updates"] == expected["optimizer_updates"] == 10
    for key in ("train_action_mse", "surrogate", "value_function", "entropy", "total_loss"):
        assert actual[key] == pytest.approx(expected[key], rel=5e-5, abs=1e-6)
    for actual_p, expected_p in zip(micro_policy.parameters(), full_policy.parameters()):
        torch.testing.assert_close(actual_p, expected_p, rtol=5e-4, atol=5e-6)


def test_losses_gradients_and_schedule_match_actual_3904770_source(spec, teacher, observations):
    """Compare to the original commit, not a second copy of our loss formula."""
    path = "source/g1_hoi_learning/g1_hoi_learning/algorithms/distillation/algorithm.py"
    source = subprocess.check_output(
        ["git", "show", f"{HISTORICAL}:{path}"], cwd=Path(__file__).resolve().parents[4], text=True
    )
    historical = types.ModuleType("historical_distillation")
    exec(compile(source, f"{HISTORICAL}/algorithm.py", "exec"), historical.__dict__)
    policy = make_policy(spec, teacher)
    original_policy = copy.deepcopy(policy)
    cfg = config(rollout_steps=1, num_learning_epochs=1, num_mini_batches=1, micro_batch_size=None, bc_coef=1)
    algorithm = MuonPPODistill(policy, cfg)
    algorithm.set_iteration(3000)
    original = historical.MuonPPODistill(
        original_policy,
        bc_coef=1,
        bc_coef_anneal_iters=6000,
        num_learning_epochs=1,
        num_mini_batches=1,
        learning_rate=cfg.learning_rate,
        schedule="fixed",
        entropy_coef=cfg.entropy_coef,
        weight_decay=cfg.weight_decay,
        value_loss_coef=cfg.value_loss_coef,
        max_grad_norm=cfg.max_grad_norm,
    )
    original._it, original.bc_coef = 3000, 0.5
    collect(algorithm, observations)
    obs = algorithm.storage.batch(np.arange(4), "cpu")[0]
    original.init_storage("rl", 4, 1, TensorDict(obs, batch_size=[4]), [3])
    original.storage.observations[0].copy_(TensorDict(obs, batch_size=[4]))
    for name in ("actions", "values", "actions_log_prob", "mu", "sigma", "returns", "advantages"):
        getattr(original.storage, name).copy_(getattr(algorithm.storage, name))
    torch.manual_seed(10)
    expected = original.update()
    torch.manual_seed(10)
    actual = algorithm.update()
    for key in ("value_function", "surrogate", "entropy"):
        assert actual[key] == pytest.approx(expected[key], rel=2e-5, abs=1e-6)
    for actual_p, expected_p in zip(policy.parameters(), original_policy.parameters()):
        if actual_p.requires_grad:
            torch.testing.assert_close(actual_p.grad, expected_p.grad, rtol=2e-5, atol=2e-7)
        torch.testing.assert_close(actual_p, expected_p, rtol=2e-5, atol=2e-7)
    assert algorithm.bc_coef == pytest.approx(original.bc_coef)


@pytest.mark.parametrize(
    "values",
    [
        {"num_mini_batches": 131073},
        {"num_learning_epochs": 0},
        {"micro_batch_size": 0},
        {"learning_rate": float("nan")},
        {"weight_decay": -1},
        {"bc_coef": 1.1},
        {"gae_lambda": -0.1},
        {"init_noise_std": 0},
        {"schedule": "unknown"},
        {"keep_last_checkpoints": 0},
        {"keep_last_checkpoints": True},
    ],
)
def test_invalid_configuration(values):
    with pytest.raises(ValueError):
        DaggerTrainConfig(**values)


def test_approved_configuration():
    cfg = DaggerTrainConfig()
    assert (cfg.num_envs, cfg.rollout_steps, cfg.num_learning_epochs, cfg.num_mini_batches) == (4096, 32, 5, 4)
    assert cfg.num_envs * cfg.rollout_steps // cfg.num_mini_batches == 32768
    assert cfg.num_learning_epochs * cfg.num_mini_batches == 20
    assert cfg.save_interval == 100 and cfg.keep_last_checkpoints is None


def test_runner_executes_gaussian_policy_preserves_states_and_resumes_all_training_state(
    spec, teacher, observations, tmp_path
):
    runner = make_runner(spec, teacher, observations, tmp_path)
    initial = observations["robot_proprio"].clone()
    with torch.no_grad():
        mean = runner.student(observations)
    runner.learn(observations, max_iterations=1)
    assert not torch.equal(runner.env.actions[0], mean)
    np.testing.assert_array_equal(runner.rollout.arrays["robot_proprio"][:4], initial.numpy())
    assert runner.iteration == runner.algorithm._it == 1
    checkpoint = torch.load(tmp_path / "model_1.pt", weights_only=False)
    assert checkpoint["schema"] == SCHEMA and "critic_state_dict" in checkpoint
    assert checkpoint["algorithm_state"]["bc_coef"] == runner.algorithm.bc_coef
    replacement = make_runner(spec, teacher, observations, tmp_path)
    replacement.load(str(tmp_path / "model_1.pt"))
    assert replacement.iteration == replacement.algorithm._it == 1
    assert replacement.total_optimizer_updates == 4
    for left, right in zip(runner.policy.parameters(), replacement.policy.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    expected_random = torch.rand(8)
    replacement.load(str(tmp_path / "model_1.pt"))
    torch.testing.assert_close(torch.rand(8), expected_random, rtol=0, atol=0)
    assert replacement.algorithm.learning_rate == runner.algorithm.learning_rate
    assert replacement.algorithm.optimizer.state_dict()["optimizers"]
    replacement.learn(observations, max_iterations=1)
    assert replacement.iteration == replacement.algorithm._it == 2
    assert replacement.total_optimizer_updates == 8
    assert replacement.total_transitions == 24


def test_runner_continues_windows_and_saves_hundred_boundary(spec, teacher, observations, tmp_path):
    runner = make_runner(spec, teacher, observations, tmp_path, config(num_iterations=11000))
    runner.iteration = 99
    runner.algorithm.set_iteration(99)
    runner.learn(observations, max_iterations=2)
    assert runner.iteration == 101
    assert len(runner.env.actions) == 6
    assert {p.name for p in tmp_path.glob("model_*.pt")} == {"model_100.pt", "model_101.pt"}
    np.testing.assert_array_equal(runner.rollout.arrays["ref_step"], [3] * 4 + [4] * 4 + [5] * 4)
    np.testing.assert_array_equal(runner.rollout.arrays["episode_step"][:4], [1, 1, 3, 3])
    weights = torch.load(tmp_path / "student_weights.pt", weights_only=False)
    assert weights["kind"] == "student_weights"
    assert "critic_state_dict" not in weights and "optimizer_state_dict" not in weights
    with pytest.raises(ValueError, match="full model"):
        runner.load(str(tmp_path / "student_weights.pt"))


def test_keep_all_and_failed_atomic_save_preserve_snapshots(spec, teacher, observations, tmp_path, monkeypatch):
    runner = make_runner(spec, teacher, observations, tmp_path)
    for iteration in (100, 200, 300):
        runner.iteration = iteration
        runner.algorithm.set_iteration(iteration)
        runner.save(tmp_path / f"model_{iteration}.pt")
    assert len(list(tmp_path.glob("model_*.pt"))) == 3
    with monkeypatch.context() as patch:

        def fail(state, path):
            path.write_bytes(b"partial")
            raise OSError("save failure")

        patch.setattr(torch, "save", fail)
        with pytest.raises(OSError, match="save failure"):
            runner.save(tmp_path / "model_400.pt")
    assert not (tmp_path / "model_400.pt").exists()
    assert len(list(tmp_path.glob("model_*.pt"))) == 3


def test_resume_rejects_old_training_but_old_student_weights_still_load(spec, teacher, observations, tmp_path):
    runner = make_runner(spec, teacher, observations, tmp_path)
    state = runner._student_snapshot()
    state["schema"] = "weave-depth-dagger-v2"
    assert state["schema"] in STUDENT_SCHEMAS
    torch.save(state, tmp_path / "legacy.pt")
    with pytest.raises(ValueError, match="Legacy"):
        runner.load(str(tmp_path / "legacy.pt"))
    runner.load_student(str(tmp_path / "legacy.pt"))
    assert runner.iteration == 0 and runner.algorithm._it == 0


def test_resume_allows_memory_save_controls_but_preserves_schedule(spec, teacher, observations, tmp_path):
    runner = make_runner(spec, teacher, observations, tmp_path)
    runner.save(tmp_path / "model_0.pt")
    runner.cfg.micro_batch_size = 1
    runner.cfg.save_interval = 200
    runner.cfg.keep_last_checkpoints = 2
    runner.load(str(tmp_path / "model_0.pt"))
    runner.cfg.gae_lambda = 0.9
    with pytest.raises(ValueError, match="original training"):
        runner.load(str(tmp_path / "model_0.pt"))


def test_partial_rollout_does_not_update_or_save(spec, teacher, observations, tmp_path):
    runner = make_runner(spec, teacher, observations, tmp_path)
    runner.learn(observations, is_running=lambda: len(runner.env.actions) < 1)
    assert runner.iteration == 0 and runner.total_optimizer_updates == 0
    assert not list(tmp_path.glob("*.pt"))


def test_adaptive_lr_bounds_and_state(spec, teacher):
    algorithm = MuonPPODistill(make_policy(spec, teacher), config(schedule="adaptive"))
    algorithm._adapt_learning_rate(0.1)
    assert algorithm.learning_rate == pytest.approx(1e-4 / 1.5)
    algorithm._adapt_learning_rate(0.001)
    assert algorithm.learning_rate == pytest.approx(1e-4)
    for group in algorithm.optimizer.param_groups:
        assert group["lr"] == pytest.approx(algorithm.learning_rate)
    assert asdict(algorithm.cfg)["schedule"] == "adaptive"


def test_first_iteration_bc_has_zero_critic_and_std_gradients(spec, teacher, observations):
    algorithm = MuonPPODistill(make_policy(spec, teacher), config(bc_coef=1))
    collect(algorithm, observations)
    batch = algorithm.storage.batch(np.arange(4), "cpu")
    output = algorithm.policy(batch[0], batch[1])
    terms = algorithm._loss_terms(output, batch)
    torch.testing.assert_close(terms[0], terms[1], rtol=0, atol=0)
    terms[0].backward()
    assert all(torch.count_nonzero(p.grad) == 0 for p in algorithm.policy.critic.parameters())
    assert torch.count_nonzero(algorithm.policy.std.grad) == 0
    assert any(torch.count_nonzero(p.grad) > 0 for p in algorithm.policy.actor.parameters() if p.requires_grad)


def test_environment_rewards_change_pure_ppo_actor_and_critic_update(spec, teacher, observations):
    policy = make_policy(spec, teacher)
    other = copy.deepcopy(policy)
    cfg = config(bc_coef=0, num_learning_epochs=1, num_mini_batches=1)
    positive = MuonPPODistill(policy, cfg)
    negative = MuonPPODistill(other, cfg)
    positive.optimizer = torch.optim.SGD(positive.trainable_parameters, lr=cfg.learning_rate)
    negative.optimizer = torch.optim.SGD(negative.trainable_parameters, lr=cfg.learning_rate)
    last_obs = collect(positive, observations)
    negative.storage = copy.deepcopy(positive.storage)
    negative.storage.rewards.neg_()
    with torch.no_grad():
        negative.compute_returns(last_obs)
    torch.manual_seed(12)
    positive_metrics = positive.update()
    torch.manual_seed(12)
    negative_metrics = negative.update()
    assert positive_metrics["bc_coef"] == negative_metrics["bc_coef"] == 0
    assert any(not torch.equal(a, b) for a, b in zip(policy.actor.parameters(), other.actor.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(policy.critic.parameters(), other.critic.parameters()))


def test_yaml_and_dataclass_defaults_match():
    from g1_hoi_learning.algorithms.dagger.io import read_yaml

    path = Path(__file__).resolve().parents[4] / "configs/dagger/train.yaml"
    settings = read_yaml(str(path))
    assert DaggerTrainConfig(**settings["training"]) == DaggerTrainConfig()
