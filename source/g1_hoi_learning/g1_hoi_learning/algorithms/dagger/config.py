"""Simulator-independent training defaults and validation."""

from dataclasses import dataclass


@dataclass
class DaggerTrainConfig:
    seed: int = 42
    num_envs: int = 32
    rollout_steps: int = 32
    batch_size: int = 256
    updates_per_iteration: int = 20
    buffer_capacity: int = 50_000
    current_fraction: float = 0.5
    visual_warmup_iterations: int = 500
    joint_warmup_iterations: int = 500
    decay_iterations: int = 5000
    student_iterations: int = 5000
    cnn_lr: float = 1e-4
    transferred_lr: float = 1e-5
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    save_interval: int = 100
    log_interval: int = 10

    def __post_init__(self):
        positive = (
            "num_envs",
            "rollout_steps",
            "batch_size",
            "updates_per_iteration",
            "buffer_capacity",
            "save_interval",
            "log_interval",
            "cnn_lr",
            "transferred_lr",
            "max_grad_norm",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("visual_warmup_iterations", "joint_warmup_iterations", "decay_iterations", "student_iterations"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.total_iterations < 1 or not 0 < self.current_fraction <= 1 or self.weight_decay < 0:
            raise ValueError("Invalid training duration, current_fraction, or weight_decay")

    @property
    def total_iterations(self):
        return (
            self.visual_warmup_iterations
            + self.joint_warmup_iterations
            + self.decay_iterations
            + self.student_iterations
        )

    def stage(self, iteration: int) -> tuple[str, float]:
        if iteration < self.visual_warmup_iterations:
            return "visual_warmup", 1.0
        iteration -= self.visual_warmup_iterations
        if iteration < self.joint_warmup_iterations:
            return "joint_warmup", 1.0
        iteration -= self.joint_warmup_iterations
        if iteration < self.decay_iterations:
            return "dagger_decay", 1.0 - iteration / self.decay_iterations
        return "student_only", 0.0
