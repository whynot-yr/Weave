"""Two real Gloo processes: DAgger gradients, independent sampling, save/resume and errors."""

import copy
import json
import random
from datetime import timedelta
from pathlib import Path

import conftest  # noqa: F401 -- bootstrap simulator-free packages in spawned workers too.
import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from g1_hoi_learning.algorithms.dagger.algorithm import DaggerAlgorithm
from g1_hoi_learning.algorithms.dagger.config import DaggerTrainConfig
from g1_hoi_learning.algorithms.dagger.distributed import DistributedContext
from g1_hoi_learning.algorithms.dagger.networks import STUDENT_GROUPS, StudentWeaveActor
from g1_hoi_learning.algorithms.dagger.runner import DaggerRunner
from g1_hoi_learning.algorithms.dagger.storage import RolloutBuffer
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks


def _context(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=60))
    return DistributedContext(rank, rank, 2, torch.device("cpu"))


def _obs(spec, count, seed):
    generator = torch.Generator().manual_seed(seed)
    obs = {key: torch.randn(count, width, generator=generator) for key, width in spec["group_dims"].items()}
    depth = torch.rand(count, 1, 128, 128, generator=generator)
    obs["head_depth"] = (depth * 10000).round() / 10000  # Same quantization as rollout storage.
    return obs


class OrderedRollout(RolloutBuffer):
    def __init__(self, obs, target, orders):
        super().__init__(len(target), 1)
        self.add(obs, target, {})
        self.orders = orders
        self.epoch = 0

    def minibatches(self, num_mini_batches):
        order = self.orders[self.epoch]
        self.epoch += 1
        yield from np.array_split(order, num_mini_batches)


def _gradient_worker(rank, rendezvous, spec, teacher):
    context = _context(rank, rendezvous)
    try:
        torch.manual_seed(71)
        student = StudentWeaveActor(spec)
        student.initialize_from_teacher(teacher)
        cfg = DaggerTrainConfig(
            num_envs=7, rollout_steps=1, num_learning_epochs=2, num_mini_batches=3, micro_batch_size=2
        )
        algorithm = DaggerAlgorithm(student, cfg, context)
        baseline = copy.deepcopy(student)
        baseline_optimizer = torch.optim.AdamW(
            [
                {"params": baseline.depth_encoder.parameters(), "lr": cfg.cnn_lr},
                {"params": baseline.backbone.parameters(), "lr": cfg.transferred_lr},
            ],
            weight_decay=cfg.weight_decay,
        )
        all_obs = _obs(spec, 14, 731)
        targets = torch.randn(14, spec["num_actions"], generator=torch.Generator().manual_seed(32))
        orders = [
            torch.randperm(7, generator=torch.Generator().manual_seed(40 + rank + epoch * 10)).numpy()
            for epoch in range(2)
        ]
        rank_orders = context.gather_objects(orders)
        obs = {key: value[rank * 7 : (rank + 1) * 7] for key, value in all_obs.items()}
        rollout = OrderedRollout(obs, targets[rank * 7 : (rank + 1) * 7], orders)
        counter = {"microbatch": 0, "sync_microbatches": set()}
        original_batch = rollout.batch

        def tracked_batch(indices, device):
            counter["microbatch"] += 1
            return original_batch(indices, device)

        rollout.batch = tracked_batch

        def hook(state, bucket):
            state["sync_microbatches"].add(state["microbatch"])
            return default_hooks.allreduce_hook(None, bucket)

        algorithm.model.register_comm_hook(counter, hook)
        result = algorithm.update(rollout, "cpu")
        assert result["optimizer_updates"] == 6 and result["samples_trained"] == 28
        # DDP may split gradients into multiple buckets; only final microbatches may communicate.
        assert counter["sync_microbatches"] == {2, 3, 4, 6, 7, 8}
        loss_sum = 0.0
        for epoch in range(2):
            partitions = [np.array_split(order[epoch], 3) for order in rank_orders]
            for batch_index in range(3):
                ids = np.concatenate([partitions[0][batch_index], partitions[1][batch_index] + 7])
                baseline_optimizer.zero_grad(set_to_none=True)
                prediction = baseline({key: value[ids] for key, value in all_obs.items() if key in STUDENT_GROUPS})
                loss = torch.nn.functional.mse_loss(prediction, targets[ids])
                loss_sum += loss.item() * len(ids)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(baseline.depth_encoder.parameters()) + list(baseline.backbone.parameters()), cfg.max_grad_norm
                )
                baseline_optimizer.step()
        assert result["train_action_mse"] == pytest.approx(loss_sum / 28, rel=2e-5, abs=1e-7)
        for actual, expected in zip(student.parameters(), baseline.parameters()):
            torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-6)
            if actual.requires_grad:
                torch.testing.assert_close(actual.grad, expected.grad, rtol=3e-4, atol=3e-6)
            else:
                assert actual.grad is None
        rank_parameters = context.gather_objects(student.state_dict())
        for key in rank_parameters[0]:
            torch.testing.assert_close(rank_parameters[0][key], rank_parameters[1][key], rtol=0, atol=0)
    finally:
        context.close()


def test_ddp_accumulated_gradients_and_updates_match_global_batch(spec, teacher, tmp_path):
    mp.spawn(_gradient_worker, args=((tmp_path / "gradient.rendezvous").as_uri(), spec, teacher), nprocs=2, join=True)


class SamplingEnv:
    def __init__(self, obs):
        from types import SimpleNamespace

        self.obs = obs
        self.num_envs = len(obs["head_depth"])
        self.steps = 0
        command = SimpleNamespace(
            env_clip=torch.zeros(self.num_envs, dtype=torch.long),
            time_steps=torch.zeros(self.num_envs, dtype=torch.long),
        )
        self.unwrapped = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: command))

    def step(self, action):
        assert action.shape[0] == self.num_envs
        self.steps += 1
        for key in self.obs:
            if key != "head_depth":
                self.obs[key].add_(0.01)
        self.unwrapped.command_manager.get_term("motion").time_steps += 1
        return self.obs, torch.ones(self.num_envs), torch.zeros(self.num_envs, dtype=torch.long), {}


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
        )
        student = StudentWeaveActor(spec)
        student.initialize_from_teacher(teacher)
        obs = _obs(spec, 4, 99 + rank)
        env = SamplingEnv(obs)
        runner = DaggerRunner(env, teacher, student, cfg, directory, {"sha256": "fixture"}, {}, {}, context)
        torch.manual_seed(111 + rank)
        np.random.seed(111 + rank)
        random.seed(111 + rank)
        root = Path(directory)
        if mode == "stop":
            runner.learn(obs, is_running=lambda: rank == 0 or env.steps < 1)
            assert env.steps == 1 and runner.iteration == 0
            assert not list(root.glob("*.pt"))
            return
        if mode == "error":
            if rank == 1:
                obs["head_depth"][0, 0, 0, 0] = float("nan")
            with pytest.raises(RuntimeError, match="rank 1.*Non-finite"):
                runner.learn(obs)
            assert runner.iteration == 0 and not list(root.glob("*.pt"))
            return
        runner.learn(obs)
        assert runner.total_transitions == 32 and runner.total_optimizer_updates == 8
        assert env.steps == 4
        np.testing.assert_array_equal(runner.rollout.arrays["env_id"], np.tile(np.arange(4) + rank * 4, 2))
        state = torch.load(root / "model_2.pt", map_location="cpu", weights_only=False)
        assert state["world_size"] == 2 and len(state["rank_states"]) == 2
        assert not torch.equal(state["rank_states"][0]["rng"]["torch"], state["rank_states"][1]["rng"]["torch"])
        assert all(not key.startswith("module.") for key in state["student_state_dict"])
        rows = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
        assert len(rows) == 2 and all(row["rollout_samples"] == 16 for row in rows)
        assert all(row["samples_trained"] == 32 and row["optimizer_updates"] == 4 for row in rows)
        assert all(len(row["rank_resources"]) == 2 for row in rows)
        assert len(list((root / "tensorboard").glob("events.*"))) == 1
        assert (root / "student_policy.pt").exists()
        runner.load(str(root / "model_1.pt"))
        expected = torch.rand(5)
        runner.load(str(root / "model_1.pt"))
        torch.testing.assert_close(torch.rand(5), expected, rtol=0, atol=0)
        assert runner.episode_steps.sum() == 0 and runner.rollout.size == 0
        runner.learn(obs, max_iterations=1)
        assert runner.iteration == 2 and runner.total_transitions == 32 and runner.total_optimizer_updates == 8
        state["world_size"] = 1
        context.rank_zero_call(lambda: torch.save(state, root / "single.pt"))
        with pytest.raises(RuntimeError, match="same world size"):
            runner.load(str(root / "single.pt"))
        runner.load_student(str(root / "single.pt"))  # Weight initialization permits changing card count.
    finally:
        context.close()


@pytest.mark.parametrize("mode", ["train", "stop", "error"])
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
