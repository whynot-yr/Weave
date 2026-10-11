"""Two actual Gloo ranks: unequal masks, accumulation, and model gradients."""

from contextlib import nullcontext

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from weave_policy.config import ReferencePolicyConfig
from weave_policy.flow_matching_policy import FlowMatchingPolicy
from weave_policy.training.distributed import (
    DistributedContext,
    DistributedEvaluationSampler,
    backward_loss,
)


def _gradient_worker(rank, rendezvous, statistics):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    context = DistributedContext(rank, rank, 2, torch.device("cpu"))
    try:
        torch.manual_seed(71)
        config = ReferencePolicyConfig(width=24, num_heads=2, head_dim=12, num_layers=2, dropout=0)
        policy = FlowMatchingPolicy(config, statistics)
        ddp = DDP(policy)
        # Every rank builds identical global inputs; each processes its partition.
        images = torch.randn(4, 257, 768)
        reference = torch.rand(4, 5, 125)
        noise = torch.randn_like(reference)
        time = torch.tensor([0.1, 0.3, 0.6, 0.9])
        mask = torch.zeros_like(reference, dtype=torch.bool)
        for sample, length in enumerate([5, 1, 4, 2]):
            mask[sample, :length] = True
        baseline = FlowMatchingPolicy(config, statistics)
        baseline.load_state_dict(policy.state_dict())
        optimizer = torch.optim.SGD(policy.parameters(), lr=0.01)
        baseline_optimizer = torch.optim.SGD(baseline.parameters(), lr=0.01)
        # K=2, a short final K=1 update, and a rank with zero valid elements.
        for indices, zero_rank in [([0, 1, 2, 3], False), ([0, 1], False), ([0, 1], True)]:
            optimizer.zero_grad()
            baseline_optimizer.zero_grad()
            active_mask = mask.clone()
            if zero_rank:
                active_mask[1] = False
            local_indices = indices[rank::2]
            global_count = context.sum(active_mask[local_indices].sum())
            for index, sample in enumerate(local_indices):
                with ddp.no_sync() if index < len(local_indices) - 1 else nullcontext():
                    output = ddp(
                        images[sample : sample + 1], reference[sample : sample + 1], active_mask[sample : sample + 1],
                        noise=noise[sample : sample + 1], time=time[sample : sample + 1],
                    )
                    backward_loss(output["loss_sum"], global_count, 2).backward()
            expected = baseline(
                images[indices], reference[indices], active_mask[indices], noise=noise[indices], time=time[indices]
            )
            (expected["loss_sum"] / expected["valid_count"]).backward()
            for (name, parameter), expected_parameter in zip(policy.named_parameters(), baseline.parameters()):
                assert parameter.grad is not None, name
                torch.testing.assert_close(parameter.grad, expected_parameter.grad, atol=2e-6, rtol=2e-4, msg=name)
            optimizer.step()
            baseline_optimizer.step()
            for parameter, expected_parameter in zip(policy.parameters(), baseline.parameters()):
                torch.testing.assert_close(parameter, expected_parameter, atol=2e-7, rtol=2e-5)
        # Check collectively propagated writer errors and nonfinite loss flags.
        with pytest.raises(RuntimeError, match="write failed"):
            context.rank_zero_call(lambda: (_ for _ in ()).throw(OSError("write failed")))
        with pytest.raises(FloatingPointError):
            context.require_finite(torch.tensor(float("nan") if rank else 1.0), "loss")
    finally:
        context.close()


def test_ddp_gradients_equal_concatenated_batch(statistics, tmp_path):
    mp.spawn(_gradient_worker, args=((tmp_path / "rendezvous").as_uri(), statistics), nprocs=2, join=True)


def test_evaluation_sampler_has_no_duplicates():
    for size in [1, 5, 12]:
        dataset = list(range(size))
        shards = [list(DistributedEvaluationSampler(dataset, rank, 2)) for rank in range(2)]
        assert sorted(shards[0] + shards[1]) == dataset
        assert not set(shards[0]) & set(shards[1])
        for rank in range(2):
            assert len(DistributedEvaluationSampler(dataset, rank, 2)) == len(shards[rank])


def test_zero_global_count_is_rejected():
    with pytest.raises(ValueError, match="globally valid"):
        backward_loss(torch.tensor(0.0, requires_grad=True), torch.tensor(0), 2)
