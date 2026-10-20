"""Two real Gloo ranks: PPO gradients, global GAE statistics and save/resume."""

import copy
import json
import random
from datetime import timedelta
from pathlib import Path

import conftest  # noqa: F401 -- bootstrap simulator-free packages in spawned workers.
import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from conftest import make_policy
from g1_hoi_learning.algorithms.dagger.config import DaggerTrainConfig
from g1_hoi_learning.algorithms.dagger.distributed import DistributedContext
from g1_hoi_learning.algorithms.dagger.runner import DaggerRunner
from g1_hoi_learning.algorithms.distillation.algorithm import MuonPPODistill
from test_ppo import FakeEnv
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks


def _context(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=60))
    return DistributedContext(rank, rank, 2, torch.device("cpu"))


def _obs(spec, count, seed):
    generator = torch.Generator().manual_seed(seed)
    obs = {key: torch.randn(count, width, generator=generator) for key, width in spec["group_dims"].items()}
    obs["robot_privileged"] = torch.randn(count, 8, generator=generator)
    obs["goal_robot_pose"] = torch.randn(count, 62, generator=generator)
    obs["goal_object_pose"] = torch.randn(count, 9, generator=generator)
    return obs


def _gradient_worker(rank, rendezvous, spec, teacher):
    context = _context(rank, rendezvous)
    try:
        torch.manual_seed(71)
        policy = make_policy(spec, teacher)
        baseline = copy.deepcopy(policy)
        cfg = DaggerTrainConfig(
            num_envs=7,
            rollout_steps=1,
            num_learning_epochs=2,
            num_mini_batches=3,
            micro_batch_size=2,
            bc_coef=0.5,
            schedule="fixed",
        )
        algorithm = MuonPPODistill(policy, cfg, context)
        # Test gradient transport independently of Muon's BF16 orthogonalization.
        algorithm.optimizer = torch.optim.SGD(algorithm.trainable_parameters, lr=cfg.learning_rate)
        baseline_optimizer = torch.optim.SGD(
            [p for p in baseline.parameters() if p.requires_grad], lr=cfg.learning_rate
        )
        all_obs = _obs(spec, 14, 731)
        obs = {key: value[rank * 7 : (rank + 1) * 7] for key, value in all_obs.items()}
        algorithm.init_storage("rl", 7, 1, obs)
        with torch.no_grad():
            torch.manual_seed(100 + rank)
            algorithm.act(obs)
            algorithm.process_env_step(obs, torch.arange(7).float() + rank, torch.zeros(7), {})
            algorithm.compute_returns(obs)
        local_batch = algorithm.storage.batch(np.arange(7), "cpu")
        rank_batches = context.gather_objects(local_batch)
        joined = (
            {key: torch.cat([b[0][key] for b in rank_batches]) for key in local_batch[0]},
            *[torch.cat([b[i] for b in rank_batches]) for i in range(1, 8)],
            (None, None),
            None,
        )
        torch.testing.assert_close(joined[3].mean(), torch.tensor(0.0), atol=1e-6, rtol=0)
        torch.testing.assert_close(joined[3].std(), torch.tensor(1.0), atol=1e-6, rtol=0)
        orders = [
            torch.randperm(7, generator=torch.Generator().manual_seed(40 + rank + epoch * 10)).numpy()
            for epoch in range(2)
        ]
        rank_orders = context.gather_objects(orders)
        epoch_index = [0]

        def minibatches(count):
            order = orders[epoch_index[0]]
            epoch_index[0] += 1
            yield from np.array_split(order, count)

        algorithm.storage.minibatches = minibatches
        counter = {"microbatch": 0, "sync_microbatches": set()}
        original_batch = algorithm.storage.batch

        def tracked_batch(indices, device):
            counter["microbatch"] += 1
            return original_batch(indices, device)

        algorithm.storage.batch = tracked_batch

        def hook(state, bucket):
            state["sync_microbatches"].add(state["microbatch"])
            return default_hooks.allreduce_hook(None, bucket)

        algorithm.model.register_comm_hook(counter, hook)
        actual_gradients = []
        original_step = algorithm.optimizer.step

        def step():
            actual_gradients.append([p.grad.clone() for p in algorithm.trainable_parameters])
            return original_step()

        algorithm.optimizer.step = step
        result = algorithm.update()
        assert result["optimizer_updates"] == 6 and result["samples_trained"] == 28
        assert counter["sync_microbatches"] == {2, 3, 4, 6, 7, 8}
        update_index = 0
        for epoch in range(2):
            partitions = [np.array_split(order[epoch], 3) for order in rank_orders]
            for batch_index in range(3):
                ids = np.concatenate([partitions[0][batch_index], partitions[1][batch_index] + 7])
                batch = (
                    {key: value[ids] for key, value in joined[0].items()},
                    *[joined[i][ids] for i in range(1, 8)],
                    (None, None),
                    None,
                )
                baseline_optimizer.zero_grad(set_to_none=True)
                output = baseline(batch[0], batch[1])
                loss = algorithm._loss_terms(output, batch)[0]
                loss.backward()
                baseline_parameters = [p for p in baseline.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(baseline_parameters, cfg.max_grad_norm)
                for actual, expected in zip(actual_gradients[update_index], baseline_parameters):
                    torch.testing.assert_close(actual, expected.grad, rtol=4e-4, atol=2e-6)
                baseline_optimizer.step()
                update_index += 1
        for actual, expected in zip(policy.parameters(), baseline.parameters()):
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-6)
        rank_parameters = context.gather_objects(policy.state_dict())
        for key in rank_parameters[0]:
            torch.testing.assert_close(rank_parameters[0][key], rank_parameters[1][key], rtol=0, atol=0)
    finally:
        context.close()


def test_ddp_accumulated_ppo_gradients_match_global_batch(spec, teacher, tmp_path):
    mp.spawn(
        _gradient_worker,
        args=((tmp_path / "gradient.rendezvous").as_uri(), spec, teacher),
        nprocs=2,
        join=True,
    )


def _runner_worker(rank, rendezvous, spec, teacher, directory, mode):
    context = _context(rank, rendezvous)
    try:
        cfg = DaggerTrainConfig(
            num_envs=4,
            rollout_steps=2,
            num_learning_epochs=2,
            num_mini_batches=2,
            micro_batch_size=3,
            num_iterations=2,
            save_interval=1,
            bc_coef=0.5,
        )
        torch.manual_seed(52)
        policy = make_policy(spec, teacher)
        obs = _obs(spec, 4, 99 + rank)
        env = FakeEnv(obs)
        runner = DaggerRunner(
            env, teacher, policy.actor, cfg, directory, {"sha256": "fixture"}, {}, {}, context, policy=policy
        )
        torch.manual_seed(111 + rank)
        np.random.seed(111 + rank)
        random.seed(111 + rank)
        root = Path(directory)
        if mode == "stop":
            runner.learn(obs, is_running=lambda: rank == 0 or len(env.actions) < 1)
            assert len(env.actions) == 1 and runner.iteration == 0
            assert not list(root.glob("*.pt"))
            return
        if mode == "error":
            if rank == 1:
                obs["object_state"][0, 0] = float("nan")
            with pytest.raises(RuntimeError, match="rank 1.*Non-finite"):
                runner.learn(obs)
            assert runner.iteration == 0 and not list(root.glob("*.pt"))
            return
        if mode == "bootstrap_error":
            original_step = env.step

            def corrupt_terminal_observation(actions):
                result = original_step(actions)
                if rank == 1 and len(env.actions) == cfg.rollout_steps:
                    result[0]["robot_privileged"][0, 0] = float("nan")
                return result

            env.step = corrupt_terminal_observation
            with pytest.raises(FloatingPointError, match="bootstrap value"):
                runner.learn(obs)
            assert runner.iteration == 0 and not list(root.glob("*.pt"))
            return
        runner.learn(obs)
        assert runner.total_transitions == 32 and runner.total_optimizer_updates == 8
        assert len(env.actions) == 4
        np.testing.assert_array_equal(runner.rollout.arrays["env_id"], np.tile(np.arange(4) + rank * 4, 2))
        state = torch.load(root / "model_2.pt", map_location="cpu", weights_only=False)
        assert state["world_size"] == 2 and len(state["rank_states"]) == 2
        assert state["algorithm_state"]["iteration"] == 2
        assert "critic_state_dict" in state and "action_std" in state
        assert not torch.equal(state["rank_states"][0]["rng"]["torch"], state["rank_states"][1]["rng"]["torch"])
        assert all(not key.startswith("module.") for key in state["student_state_dict"])
        rows = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
        assert len(rows) == 2 and all(row["rollout_samples"] == 16 for row in rows)
        assert all(row["samples_trained"] == 32 and row["optimizer_updates"] == 4 for row in rows)
        assert all(len(row["rank_resources"]) == 2 for row in rows)
        assert len(list((root / "tensorboard").glob("events.*"))) == 1
        assert (root / "student_policy.pt").exists()
        parameters = context.gather_objects(policy.state_dict())
        for key in parameters[0]:
            torch.testing.assert_close(parameters[0][key], parameters[1][key], rtol=0, atol=0)
        runner.load(str(root / "model_1.pt"))
        expected = torch.rand(5)
        runner.load(str(root / "model_1.pt"))
        torch.testing.assert_close(torch.rand(5), expected, rtol=0, atol=0)
        assert runner.episode_steps.sum() == 0 and runner.rollout.size == 0
        runner.learn(obs, max_iterations=1)
        assert runner.iteration == runner.algorithm._it == 2
        assert runner.total_transitions == 32 and runner.total_optimizer_updates == 8
        state["world_size"] = 1
        context.rank_zero_call(lambda: torch.save(state, root / "single.pt"))
        with pytest.raises(RuntimeError, match="same world size"):
            runner.load(str(root / "single.pt"))
    finally:
        context.close()


@pytest.mark.parametrize("mode", ["train", "stop", "error", "bootstrap_error"])
def test_two_rank_sampling_save_resume_and_coordinated_exit(spec, teacher, tmp_path, mode):
    mp.spawn(
        _runner_worker,
        args=((tmp_path / "runner.rendezvous").as_uri(), spec, teacher, str(tmp_path / "run"), mode),
        nprocs=2,
        join=True,
    )


def _errors_worker(rank, rendezvous):
    context = _context(rank, rendezvous)
    try:
        with pytest.raises(ValueError, match="Ranks disagree"):
            context.require_same(rank, "test protocol")
        with pytest.raises(FloatingPointError, match="Non-finite"):
            context.require_finite(torch.tensor(float("nan") if rank else 1.0), "test loss")
        with pytest.raises(RuntimeError, match="writer failed"):
            context.rank_zero_call(lambda: (_ for _ in ()).throw(OSError("writer failed")))
        with pytest.raises(RuntimeError, match="rank 1.*local failure"):
            context.all_call(lambda: (_ for _ in ()).throw(ValueError("local failure")) if rank else None)
        assert not context.all_running(rank == 0)
    finally:
        context.close()


def test_errors_and_writer_failures_propagate_to_both_ranks(tmp_path):
    mp.spawn(_errors_worker, args=((tmp_path / "errors.rendezvous").as_uri(),), nprocs=2, join=True)
