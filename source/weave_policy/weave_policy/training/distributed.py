"""DDP setup, exact masked-loss scaling, and coordinated rank-zero writes."""

import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.utils.data import Sampler


@dataclass
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @classmethod
    def initialize(cls, requested_device):
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = torch.device(requested_device)
        if device.type not in {"cpu", "cuda"}:
            raise ValueError("Training supports CPU or CUDA devices")
        if device.type == "cuda":
            device = torch.device("cuda", local_rank if world_size > 1 else device.index or 0)
            torch.cuda.set_device(device)
        if world_size > 1:
            dist.init_process_group(
                backend="nccl" if device.type == "cuda" else "gloo",
                timeout=timedelta(minutes=10),
            )
        return cls(rank, local_rank, world_size, device)

    @property
    def distributed(self):
        return self.world_size > 1

    def sum(self, value):
        result = value.detach().clone().to(self.device)
        if self.distributed:
            dist.all_reduce(result, op=dist.ReduceOp.SUM)
        return result

    def gather_objects(self, value):
        if not self.distributed:
            return [value]
        result = [None] * self.world_size
        dist.all_gather_object(result, value)
        return result

    def require_same(self, value, description):
        values = self.gather_objects(value)
        if any(item != values[0] for item in values[1:]):
            raise ValueError(f"Ranks disagree on {description}")

    def rank_zero_call(self, function):
        """Broadcast success/errors so peers do not wait after a failed write."""
        status = [None]
        if self.rank == 0:
            try:
                status[0] = {"value": function()}
            except Exception as error:
                status[0] = {"error": f"{type(error).__name__}: {error}"}
        if self.distributed:
            dist.broadcast_object_list(status, src=0)
        if "error" in status[0]:
            raise RuntimeError(status[0]["error"])
        return status[0]["value"]

    def require_finite(self, value, description):
        invalid = (~torch.isfinite(value.detach()).all()).to(device=self.device, dtype=torch.int32)
        if self.sum(invalid).item():
            raise FloatingPointError(f"Nonfinite {description} on at least one rank")

    def close(self):
        if self.distributed and dist.is_initialized():
            dist.destroy_process_group()


def backward_loss(loss_sum, global_count, world_size):
    """Compensate DDP's gradient average; denominator spans the whole update.

    Each microbatch contributes W*S_rk/C_global. There is no additional divide
    by the accumulation count, even for a short final accumulation window.
    """
    if global_count.item() <= 0:
        raise ValueError("An optimizer update requires globally valid reference elements")
    return loss_sum * world_size / global_count.to(device=loss_sum.device, dtype=loss_sum.dtype)


class DistributedEvaluationSampler(Sampler):
    """Disjoint validation indices without duplicating or discarding samples."""

    def __init__(self, dataset, rank, world_size):
        self.size, self.rank, self.world_size = len(dataset), rank, world_size

    def __iter__(self):
        return iter(range(self.rank, self.size, self.world_size))

    def __len__(self):
        return len(range(self.rank, self.size, self.world_size))
