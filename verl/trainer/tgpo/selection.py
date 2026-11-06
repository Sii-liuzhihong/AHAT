"""Selection policy, preserving the original experiment's grouping and tie rules."""

from collections import defaultdict

import numpy as np
import torch


def select_refinement_indices(instructions, rewards, repeat_num, select_num, threshold):
    """Select top-scoring rows only when every reward in their group is below threshold.

    The source groups by instruction, then chunks each group by rollout.n. This
    deliberately preserves that policy even after DP balancing reorders rows;
    switching to uid grouping is a separate algorithm change.
    """
    if repeat_num < 1 or select_num < 0:
        raise ValueError("repeat_num must be positive and select_num nonnegative")
    if isinstance(rewards, list):
        rewards = torch.tensor(rewards, dtype=torch.float32)
    elif isinstance(rewards, np.ndarray):
        rewards = torch.from_numpy(rewards).float()
    if not isinstance(rewards, torch.Tensor) or rewards.ndim != 1:
        raise TypeError("rewards must be a one-dimensional tensor, list, or ndarray")
    if len(instructions) != len(rewards):
        raise ValueError("instruction and reward lengths differ")
    groups = defaultdict(list)
    for i, instruction in enumerate(instructions):
        groups[instruction].append(i)
    selected = []
    for instruction, indices in groups.items():
        if len(indices) % repeat_num:
            raise ValueError(f"instruction {instruction!r} count is not divisible by rollout.n")
        for start in range(0, len(indices), repeat_num):
            group = indices[start:start + repeat_num]
            values = rewards[torch.tensor(group, device=rewards.device)]
            if torch.all(values < torch.tensor(threshold, device=rewards.device)):
                if select_num >= len(group):
                    selected.extend(group)
                else:
                    # sorted=False and torch.topk ties intentionally match source.
                    top = torch.topk(values, k=select_num, largest=True, sorted=False).indices
                    selected.extend(group[i] for i in top.tolist())
    return sorted(set(selected))


def best_candidate_indices(rewards, repeat_num):
    """First maximum wins a tie within each interleaved continuation group."""
    if repeat_num < 1:
        raise ValueError("repeat_num must be positive")
    rewards = torch.as_tensor(rewards)
    if rewards.ndim != 1 or len(rewards) % repeat_num:
        raise ValueError("candidate reward count must be divisible by continuation count")
    grouped = rewards.reshape(-1, repeat_num)
    return (torch.arange(len(grouped), device=rewards.device) * repeat_num + grouped.argmax(1)).tolist()
