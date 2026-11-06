"""Tensor operations for refinement; task text formatting is supplied by the caller."""

from copy import copy, deepcopy

import numpy as np
import torch

from verl import DataProto
from verl.utils.model import compute_position_id_with_mask


def _encode_prefixes(tokenizer, texts, device):
    padding_side = tokenizer.padding_side
    try:
        tokenizer.padding_side = "right"
        result = tokenizer(texts, return_tensors="pt", padding=True, truncation=False, add_special_tokens=False)
    finally:
        tokenizer.padding_side = padding_side
    return result["input_ids"].to(device), result["attention_mask"].to(device)


def continuation_prompts(batch, tokenizer, formatted_prefixes, *, position_policy="recompute"):
    """Append prefixes to left-padded prompts and retain the original prompt capacity.

    Truncation from the left matches the development code. The generated prefix
    is not a complete response: the actor must generate the remaining tokens.
    """
    response_length = batch.batch["responses"].shape[-1]
    tensors = {k: batch.batch[k].clone() for k in ("input_ids", "attention_mask", "position_ids", "responses")}
    result = DataProto.from_dict(tensors=tensors,
                                 non_tensors={"fix_cot": batch.non_tensor_batch["fix_cot"].copy()},
                                 meta_info=copy(batch.meta_info))
    ids = tensors["input_ids"][:, :-response_length].clone()
    mask = tensors["attention_mask"][:, :-response_length].clone()
    prefix_ids, prefix_mask = _encode_prefixes(tokenizer, formatted_prefixes, ids.device)
    width = ids.shape[-1]
    if not width:
        raise ValueError("refinement requires nonempty prompts")
    for i in range(len(batch)):
        length = int(prefix_mask[i].sum())
        ids[i] = torch.cat((ids[i], prefix_ids[i, :length]))[-width:]
        mask[i] = torch.cat((mask[i], prefix_mask[i, :length]))[-width:]
    result.batch["input_ids"] = ids
    result.batch["attention_mask"] = mask
    # Source accidentally retained full prompt+response position ids here.
    # A compatibility mode is available for differential tests, never the recipe default.
    if position_policy == "legacy":
        pass
    elif position_policy == "recompute":
        result.batch["position_ids"] = compute_position_id_with_mask(mask)
    else:
        raise ValueError("position_policy must be recompute or legacy")
    return result


def merge_continuations(batch, generated, tokenizer, formatted_prefixes, *, position_policy="recompute"):
    """Train on repaired prefix plus continuation, cropped to original response width."""
    response = generated.batch["responses"].clone()
    width = response.shape[-1]
    mask = generated.batch["attention_mask"][:, -width:].clone()
    ids, prefix_mask = _encode_prefixes(tokenizer, formatted_prefixes, response.device)
    for i in range(len(batch)):
        length = int(prefix_mask[i].sum())
        response[i] = torch.cat((ids[i, :length], response[i]))[:width]
        mask[i] = torch.cat((prefix_mask[i, :length], mask[i]))[:width]
    prompt_ids = batch.batch["input_ids"][:, :-width].clone()
    prompt_mask = batch.batch["attention_mask"][:, :-width].clone()
    batch.batch["responses"] = response
    batch.batch["response_mask"] = mask
    batch.batch["input_ids"] = torch.cat((prompt_ids, response), dim=-1)
    batch.batch["attention_mask"] = torch.cat((prompt_mask, mask), dim=-1)
    if position_policy == "recompute":
        batch.batch["position_ids"] = compute_position_id_with_mask(batch.batch["attention_mask"])
    elif position_policy != "legacy":
        raise ValueError("position_policy must be recompute or legacy")
    return batch


def attach_reward_info(batch, info):
    """Use object arrays for strings and nested records to avoid fixed-width truncation."""
    for key, values in info.items():
        if len(values) != len(batch):
            raise ValueError(f"reward field {key!r} has wrong length")
        array = np.empty(len(values), dtype=object)
        array[:] = list(values)
        batch.non_tensor_batch[key] = array


def replace_improved(batch, candidates, selected, best, reward_tensor, candidate_reward, info, candidate_info):
    """Replace all row fields and reward diagnostics together; equal scores retain originals."""
    improved = []
    for row, candidate in zip(selected, best, strict=True):
        score = float(candidate_info["score"][candidate])
        if score <= float(info["score"][row]):
            continue
        for key in batch.batch.keys():
            batch.batch[key][row] = candidates.batch[key][candidate].clone()
        for key in batch.non_tensor_batch:
            batch.non_tensor_batch[key][row] = deepcopy(candidates.non_tensor_batch[key][candidate])
        reward_tensor[row] = candidate_reward[candidate].clone()
        # The old code updated score only; this would overwrite new reviewer
        # feedback in the shared loop. Keep all selected reward fields aligned.
        for key in info:
            info[key][row] = deepcopy(candidate_info[key][candidate])
        improved.append(row)
    batch.meta_info["global_token_num"] = batch.batch["attention_mask"].sum(-1).tolist()
    return improved
