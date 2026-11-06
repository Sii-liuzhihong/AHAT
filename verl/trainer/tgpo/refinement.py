"""TGPO orchestration independent of robot task prompts, clients, and PDDL."""

from collections.abc import Callable
from dataclasses import dataclass

from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.tgpo.batch_ops import (
    attach_reward_info, continuation_prompts, merge_continuations, replace_improved,
)
from verl.trainer.tgpo.selection import best_candidate_indices, select_refinement_indices


@dataclass(frozen=True)
class RefinementConfig:
    rollout_n: int
    select_num: int = 2
    continuations_per_sample: int = 1
    reward_threshold: float = 1.0
    position_policy: str = "recompute"


def refine_rollouts(batch, reward_tensor, reward_info, *, tokenizer, generate: Callable,
                    reward: Callable, repair: Callable, format_prefix: Callable,
                    config: RefinementConfig, generation_divisor: int = 1):
    """Return final batch/rewards before old-log-probability and GRPO advantage.

    repair accepts independent record dictionaries and returns one dictionary per
    row with fix_cot. Invalid/failed repairs retain their original training rows.
    """
    if "score" not in reward_info:
        raise ValueError("TGPO reward manager must return per-row score diagnostics")
    # Lists allow replacement with longer strings and nested feedback records.
    info = {k: list(v) for k, v in reward_info.items()}
    attach_reward_info(batch, info)
    selected = select_refinement_indices(batch.non_tensor_batch.get("instruction", []), info["score"],
                                         config.rollout_n, config.select_num, config.reward_threshold)
    metrics = {"tgpo/selected": len(selected), "tgpo/repaired": 0, "tgpo/improved": 0}
    if not selected:
        return batch, reward_tensor, info, metrics
    records = [{k: values[i] for k, values in batch.non_tensor_batch.items()} for i in selected]
    repairs = repair(records)
    if len(repairs) != len(selected):
        raise ValueError("repair result count differs from selected row count")
    # A failed repair is not converted into an empty prefix and regenerated.
    keep = [i for i, result in enumerate(repairs) if result.get("fix_cot") and result.get("status", "ok") not in {"error", "failed"}]
    if not keep:
        return batch, reward_tensor, info, metrics
    selected = [selected[i] for i in keep]
    repairs = [repairs[i] for i in keep]
    fixed = batch[selected]
    attach_reward_info(fixed, {"fix_cot": [r["fix_cot"] for r in repairs]})
    prefixes = [format_prefix(r["fix_cot"], phase="generation") for r in repairs]
    generation_batch = continuation_prompts(fixed, tokenizer, prefixes, position_policy=config.position_policy)
    repeats = config.continuations_per_sample
    if repeats < 1:
        raise ValueError("continuations_per_sample must be positive")
    generation_batch = generation_batch.repeat(repeats, interleave=True)
    # Selected rows rarely divide the worker count; source could fail dispatch.
    padded, pad_size = pad_dataproto_to_divisor(generation_batch, generation_divisor)
    generated = unpad_dataproto(generate(padded), pad_size)
    fixed = fixed.repeat(repeats, interleave=True)
    merged_prefixes = [format_prefix(r["fix_cot"], phase="response") for r in repairs for _ in range(repeats)]
    fixed = merge_continuations(fixed, generated, tokenizer, merged_prefixes, position_policy=config.position_policy)
    candidate_reward, candidate_info = reward(fixed)
    candidate_info = {k: list(v) for k, v in candidate_info.items()}
    attach_reward_info(fixed, candidate_info)
    best = best_candidate_indices(candidate_info["score"], repeats)
    improved = replace_improved(batch, fixed, selected, best, reward_tensor, candidate_reward, info, candidate_info)
    metrics.update({"tgpo/repaired": len(selected), "tgpo/improved": len(improved)})
    return batch, reward_tensor, info, metrics
