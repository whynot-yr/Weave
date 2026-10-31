"""Additive seed statistics; collectives run only at synchronized rollout boundaries."""
import torch.distributed as dist


def reduce_seed_statistics(stats):
    result = stats.clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return result


def seed_metrics(stats):
    """Compute ratios after summing counts, never average per-rank success rates."""
    result = {}
    reset_count = stats[:, 0].sum().item()
    for i, name in enumerate(('supine', 'prone')):
        values = stats[i].tolist()
        prefix = f'Seed/{name}/'
        result[prefix+'reset_count'] = values[0]
        result[prefix+'episode_count'] = values[1]
        if reset_count:
            result[prefix+'reset_fraction'] = values[0]/reset_count
        if values[1]:
            result[prefix+'success_rate'] = values[2]/values[1]
            result[prefix+'final_hold_rate'] = values[3]/values[1]
            result[prefix+'episode_steps'] = values[4]/values[1]
            for group, total in zip(('task', 'style', 'regu', 'post'), values[5:]):
                result[prefix+'return_'+group] = total/values[1]
    return result
