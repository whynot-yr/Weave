"""Current-rollout action MSE, jointly optimizing CNN and SimBa."""

from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from .config import DaggerTrainConfig
from .distributed import DistributedContext
from .networks import StudentWeaveActor
from .storage import RolloutBuffer


class DaggerAlgorithm:
    def __init__(self, student: StudentWeaveActor, cfg: DaggerTrainConfig, context: DistributedContext | None = None):
        self.student, self.cfg = student, cfg
        self.context = context or DistributedContext.single(next(student.parameters()).device)
        self.student.set_dagger_trainable()
        self.context.require_same((student.spec, cfg), "actor and per-rank training configuration")
        self.model = (
            DDP(
                student,
                device_ids=[self.context.device.index] if self.context.device.type == "cuda" else None,
                broadcast_buffers=False,  # Normalization is loaded identically and never changes.
            )
            if self.context.distributed
            else student
        )
        self.trainable_parameters = list(student.depth_encoder.parameters()) + list(student.backbone.parameters())
        self.optimizer = torch.optim.AdamW(
            [
                {"params": list(student.depth_encoder.parameters()), "lr": cfg.cnn_lr},
                {"params": list(student.backbone.parameters()), "lr": cfg.transferred_lr},
            ],
            weight_decay=cfg.weight_decay,
        )

    def update(self, rollout: RolloutBuffer, device: str) -> dict[str, float]:
        self.context.require_same((rollout.size, rollout.capacity), "rollout sizes")
        self.context.all_call(lambda: self._check_rollout(rollout))
        self.model.train()
        loss_sum, sample_count, grad_sum, updates = 0.0, 0, 0.0, 0
        for _ in range(self.cfg.num_learning_epochs):
            for indices in rollout.minibatches(self.cfg.num_mini_batches):
                self.optimizer.zero_grad(set_to_none=True)
                batch_size = len(indices)
                micro_size = self.cfg.micro_batch_size or batch_size
                total_loss = 0.0
                for start in range(0, batch_size, micro_size):
                    micro_indices = indices[start : start + micro_size]
                    obs, target = self.context.all_call(lambda: rollout.batch(micro_indices, device))
                    last_microbatch = start + micro_size >= batch_size
                    sync = self.model.no_sync() if self.context.distributed and not last_microbatch else nullcontext()
                    # Forward must also be inside no_sync; only the last backward synchronizes.
                    with sync:
                        prediction = self.model(obs)
                        loss = nn.functional.mse_loss(prediction, target)
                        self.context.require_finite(loss, "action MSE")
                        # Equal local batch sizes: DDP's average yields the global mean.
                        weighted_loss = loss * (len(micro_indices) / batch_size)
                        weighted_loss.backward()
                    total_loss += weighted_loss.detach().item()
                grad = nn.utils.clip_grad_norm_(
                    self.trainable_parameters, self.cfg.max_grad_norm, error_if_nonfinite=False
                )
                self.context.require_finite(grad, "gradient norm")
                self.optimizer.step()
                loss_sum += total_loss * batch_size
                sample_count += batch_size
                grad_sum += grad.item()
                updates += 1
        totals = self.context.sum(torch.tensor([loss_sum, sample_count], dtype=torch.float64))
        return {
            "train_action_mse": (totals[0] / totals[1]).item(),
            "grad_norm": grad_sum / updates,
            "optimizer_updates": updates,
            "samples_trained": int(totals[1].item()),
            "minibatches_per_epoch": self.cfg.num_mini_batches,
        }

    def _check_rollout(self, rollout):
        if rollout.size != rollout.capacity or self.cfg.num_mini_batches > rollout.size:
            raise ValueError("Cannot train on an incomplete or undersized rollout")
