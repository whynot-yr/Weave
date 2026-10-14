"""One process per device, coordinated errors/writes, and rank-local random state."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist


@dataclass
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @classmethod
    def single(cls, device):
        return cls(0, 0, 1, torch.device(device))

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
            dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo", timeout=timedelta(minutes=10))
        return cls(rank, local_rank, world_size, device)

    @property
    def distributed(self):
        return self.world_size > 1

    def _reduce(self, value, operation):
        result = torch.as_tensor(value, device=self.device).detach().clone()
        if self.distributed:
            dist.all_reduce(result, op=operation)
        return result

    def sum(self, value):
        return self._reduce(value, dist.ReduceOp.SUM)

    def max(self, value):
        return self._reduce(value, dist.ReduceOp.MAX)

    def all_running(self, running: bool):
        return bool(self._reduce(int(running), dist.ReduceOp.MIN).item())

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

    def all_call(self, function):
        """Propagate local Python errors before peers enter the next collective."""
        if not self.distributed:
            return function()
        value, error = None, None
        try:
            value = function()
        except Exception as exception:
            error = f"rank {self.rank}: {type(exception).__name__}: {exception}"
        if self.sum(int(error is not None)).item():
            errors = self.gather_objects(error)
            raise RuntimeError("; ".join(item for item in errors if item is not None))
        return value

    def rank_zero_call(self, function):
        """Only rank zero writes; broadcast success or failure to every peer."""
        if not self.distributed:
            return function()
        status = [None]
        if self.rank == 0:
            try:
                status[0] = {"value": function()}
            except Exception as error:
                status[0] = {"error": f"{type(error).__name__}: {error}"}
        dist.broadcast_object_list(status, src=0)
        if "error" in status[0]:
            raise RuntimeError(status[0]["error"])
        return status[0]["value"]

    def require_finite(self, value, description):
        invalid = (~torch.isfinite(value.detach()).all()).to(device=self.device, dtype=torch.int32)
        if self.sum(invalid).item():
            raise FloatingPointError(f"Non-finite {description} on at least one rank")

    def close(self):
        if self.distributed and dist.is_initialized():
            dist.destroy_process_group()
