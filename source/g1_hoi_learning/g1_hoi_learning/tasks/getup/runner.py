"""Native RSL-RL runner with seed-only logging at the existing PPO update boundary."""
from functools import wraps
from rsl_rl.runners import OnPolicyRunner
from .seed_logging import reduce_seed_statistics, seed_metrics


class GetupOnPolicyRunner(OnPolicyRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed_log = {}
        native_update = self.alg.update

        @wraps(native_update)
        def update_with_seed_logging():
            losses = native_update()
            # Every rank reaches this boundary, even if no local episode ended.
            stats = self.env.unwrapped.pop_seed_statistics()
            self.seed_log = seed_metrics(reduce_seed_statistics(stats))
            return losses

        self.alg.update = update_with_seed_logging

    def log(self, locs, width=80, pad=35):
        super().log(locs, width, pad)
        for name, value in self.seed_log.items():
            self.writer.add_scalar(name, value, locs['it'])
