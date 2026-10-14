"""Simulator-independent settings for current-rollout online action distillation."""

import math
from dataclasses import dataclass


@dataclass
class DaggerTrainConfig:
    seed: int = 42
    num_envs: int = 4096
    rollout_steps: int = 16
    num_learning_epochs: int = 2
    num_mini_batches: int = 32
    micro_batch_size: int | None = 128
    num_iterations: int = 11000
    cnn_lr: float = 1e-4
    transferred_lr: float = 1e-5
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    save_interval: int = 100
    keep_last_checkpoints: int = 2
    log_interval: int = 10

    def __post_init__(self):
        for name in (
            "num_envs",
            "rollout_steps",
            "num_learning_epochs",
            "num_mini_batches",
            "num_iterations",
            "save_interval",
            "keep_last_checkpoints",
            "log_interval",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.num_mini_batches > self.num_envs * self.rollout_steps:
            raise ValueError("num_mini_batches must not exceed the rollout sample count")
        if self.micro_batch_size is not None and (
            isinstance(self.micro_batch_size, bool)
            or not isinstance(self.micro_batch_size, int)
            or self.micro_batch_size <= 0
        ):
            raise ValueError("micro_batch_size must be a positive integer or null")
        for name in ("cnn_lr", "transferred_lr", "max_grad_norm"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
