from __future__ import annotations

from typing import Any

from rsl_rl.algorithms import Distillation

from g1_hoi_learning.algorithms.optimizers import MuonAdamWWrapper


class MuonDistillation(Distillation):
    """``Distillation`` with Muon (2-D weights) + AdamW (rest) on the student network.
    """

    def __init__(self, policy, weight_decay: float = 0.01, **kwargs: Any) -> None:
        super().__init__(policy, **kwargs)
        self.optimizer = MuonAdamWWrapper(
            modules=[self.policy.student],  # student only; teacher is frozen
            lr=self.learning_rate,
            weight_decay=weight_decay,
        )
