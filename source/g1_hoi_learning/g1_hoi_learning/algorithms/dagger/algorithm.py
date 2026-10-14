"""Behavior cloning updates with an explicit CNN-only warmup and preserved optimizer state."""

from __future__ import annotations

import torch
from torch import nn

from .config import DaggerTrainConfig
from .networks import StudentWeaveActor
from .storage import ReservoirBuffer


class DaggerAlgorithm:
    def __init__(self, student: StudentWeaveActor, cfg: DaggerTrainConfig):
        self.student, self.cfg = student, cfg
        self.student.set_visual_only(True)
        self.optimizer = torch.optim.AdamW(
            [{"params": list(student.depth_encoder.parameters()), "lr": cfg.cnn_lr}],
            weight_decay=cfg.weight_decay,
        )
        self.joint_enabled = False

    def set_stage(self, stage: str):
        visual_only = stage == "visual_warmup"
        if not visual_only and not self.joint_enabled:
            self.optimizer.add_param_group(
                {
                    "params": list(self.student.vector_encoders.parameters())
                    + list(self.student.backbone.parameters()),
                    "lr": self.cfg.transferred_lr,
                }
            )
            self.joint_enabled = True
        self.student.set_visual_only(visual_only)
        self.student.train()

    def update(self, current, buffer: ReservoirBuffer, device: str) -> dict[str, float]:
        losses, grad_norms = [], []
        for _ in range(self.cfg.updates_per_iteration):
            obs, target = buffer.sample(current, self.cfg.batch_size, device, self.cfg.current_fraction)
            self.optimizer.zero_grad(set_to_none=True)
            micro_size = self.cfg.micro_batch_size or self.cfg.batch_size
            total_loss = 0.0
            for start in range(0, self.cfg.batch_size, micro_size):
                stop = min(start + micro_size, self.cfg.batch_size)
                prediction = self.student({key: value[start:stop] for key, value in obs.items()})
                loss = nn.functional.mse_loss(prediction, target[start:stop])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite action MSE")
                # Weight a shorter final microbatch by its sample count as well.
                weighted_loss = loss * ((stop - start) / self.cfg.batch_size)
                weighted_loss.backward()
                total_loss += weighted_loss.detach().item()
            grad = nn.utils.clip_grad_norm_(self.student.parameters(), self.cfg.max_grad_norm, error_if_nonfinite=True)
            self.optimizer.step()
            losses.append(total_loss)
            grad_norms.append(grad.item())
        return {"action_mse": sum(losses) / len(losses), "grad_norm": sum(grad_norms) / len(grad_norms)}
