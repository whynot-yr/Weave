from __future__ import annotations

import functools

import torch


def goal_channel(func=None, *, prob=0.5):
    """Register + auto-mask a goal observation, keyed by its own name.
    """
    if func is None:
        return functools.partial(goal_channel, prob=prob)
    if func.__name__ not in goal_channel.channels:
        goal_channel.channels[func.__name__] = len(goal_channel.channels)
        goal_channel.probs.append(prob)
    col = goal_channel.channels[func.__name__]

    @functools.wraps(func)
    def wrapped(env, command_name, *args, **kwargs):
        m = env.command_manager.get_term(command_name).goal_mask[:, col : col + 1]
        return torch.cat([func(env, command_name, *args, **kwargs) * m, m], dim=-1)

    return wrapped


goal_channel.channels = {}  # {channel name: column index}
goal_channel.probs = []     # per-channel reveal probability, indexed by column
