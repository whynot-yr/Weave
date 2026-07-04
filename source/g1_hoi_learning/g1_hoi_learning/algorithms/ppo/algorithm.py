from __future__ import annotations

from typing import Any

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent

from g1_hoi_learning.algorithms.optimizers import MuonAdamWWrapper


class MuonPPO(PPO):
    """PPO with Muon (2-D weights) + AdamW (everything else) optimizers.
    """

    def __init__(
        self,
        policy: ActorCritic | ActorCriticRecurrent,
        weight_decay: float = 0.01,
        **kwargs: Any,
    ) -> None:
        super().__init__(policy, **kwargs)
        # Replace the Adam built by PPO.__init__ with Muon + AdamW.
        self.optimizer = MuonAdamWWrapper(
            modules=[self.policy],
            lr=self.learning_rate,
            weight_decay=weight_decay,
        )
