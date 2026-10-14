"""Current-rollout action MSE, jointly optimizing CNN and SimBa."""

from __future__ import annotations

import torch
from torch import nn

from .config import DaggerTrainConfig
from .networks import StudentWeaveActor
from .storage import RolloutBuffer


class DaggerAlgorithm:
    def __init__(self, student: StudentWeaveActor, cfg: DaggerTrainConfig):
        self.student, self.cfg = student, cfg
        self.student.set_dagger_trainable()
        self.trainable_parameters = list(student.depth_encoder.parameters()) + list(student.backbone.parameters())
        self.optimizer = torch.optim.AdamW(
            [
                {"params": list(student.depth_encoder.parameters()), "lr": cfg.cnn_lr},
                {"params": list(student.backbone.parameters()), "lr": cfg.transferred_lr},
            ],
            weight_decay=cfg.weight_decay,
        )

    def update(self, rollout: RolloutBuffer, device: str) -> dict[str, float]:
        self.student.train()
        loss_sum, sample_count, grad_sum, updates = 0.0, 0, 0.0, 0
        for _ in range(self.cfg.num_learning_epochs):
            for indices in rollout.minibatches(self.cfg.num_mini_batches):
                self.optimizer.zero_grad(set_to_none=True)
                batch_size = len(indices)
                micro_size = self.cfg.micro_batch_size or batch_size
                total_loss = 0.0
                for start in range(0, batch_size, micro_size):
                    micro_indices = indices[start : start + micro_size]
                    obs, target = rollout.batch(micro_indices, device)
                    prediction = self.student(obs)
                    loss = nn.functional.mse_loss(prediction, target)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite action MSE")
                    weighted_loss = loss * (len(micro_indices) / batch_size)
                    weighted_loss.backward()
                    total_loss += weighted_loss.detach().item()
                grad = nn.utils.clip_grad_norm_(
                    self.trainable_parameters, self.cfg.max_grad_norm, error_if_nonfinite=True
                )
                self.optimizer.step()
                loss_sum += total_loss * batch_size
                sample_count += batch_size
                grad_sum += grad.item()
                updates += 1
        return {
            "train_action_mse": loss_sum / sample_count,
            "grad_norm": grad_sum / updates,
            "optimizer_updates": updates,
            "samples_trained": sample_count,
            "minibatches_per_epoch": self.cfg.num_mini_batches,
        }
