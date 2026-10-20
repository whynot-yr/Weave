"""3904770 PPO+BC settings with the goal/object-state student and per-rank storage."""

import math
from dataclasses import dataclass


@dataclass
class DaggerTrainConfig:
    seed: int = 42
    num_envs: int = 4096
    rollout_steps: int = 32
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    micro_batch_size: int | None = 128
    num_iterations: int = 11000
    init_noise_std: float = 0.2
    bc_coef: float = 1.0
    bc_coef_anneal_iters: int = 6000
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_param: float = 0.2
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    entropy_coef: float = 5e-4
    learning_rate: float = 1e-4
    schedule: str = "adaptive"
    desired_kl: float = 0.01
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    save_interval: int = 100
    keep_last_checkpoints: int | None = None
    log_interval: int = 10

    def __post_init__(self):
        for name in (
            "num_envs",
            "rollout_steps",
            "num_learning_epochs",
            "num_mini_batches",
            "num_iterations",
            "bc_coef_anneal_iters",
            "save_interval",
            "log_interval",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.num_mini_batches > self.num_envs * self.rollout_steps:
            raise ValueError("num_mini_batches must not exceed the rollout sample count")
        for name in ("keep_last_checkpoints", "micro_batch_size"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                raise ValueError(f"{name} must be a positive integer or null")
        for name in ("learning_rate", "init_noise_std", "desired_kl", "max_grad_norm", "clip_param"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("weight_decay", "value_loss_coef", "entropy_coef"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("bc_coef", "gamma", "gae_lambda"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.schedule not in {"adaptive", "fixed"}:
            raise ValueError("schedule must be adaptive or fixed")
