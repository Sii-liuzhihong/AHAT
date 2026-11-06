"""Executable policy and tensor contracts; source differential tests are opt-in.

TGPO_SOURCE_ROOT points to the private development decision_making directory.
No legacy implementation is bundled into the release or needed for normal tests.
"""
import copy
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.trainer.tgpo.batch_ops import continuation_prompts, merge_continuations
from verl.trainer.tgpo.refinement import RefinementConfig, refine_rollouts
from verl.trainer.tgpo.selection import best_candidate_indices, select_refinement_indices
from verl.utils.model import compute_position_id_with_mask
from robot_planning.repair import format_repair_prefix




def make_batch(rows=4, prompt_width=128, response_width=128):
    ids = torch.zeros(rows, prompt_width + response_width, dtype=torch.long)
    mask = torch.zeros_like(ids)
    ids[:, prompt_width-8:prompt_width] = torch.tensor([80, 114, 111, 109, 112, 116, 58, 32])
    mask[:, prompt_width-8:prompt_width] = 1
    ids[:, prompt_width:prompt_width+5] = torch.tensor([104, 101, 108, 108, 111])
    mask[:, prompt_width:prompt_width+5] = 1
    return DataProto.from_dict(
        tensors={'input_ids': ids, 'attention_mask': mask, 'position_ids': compute_position_id_with_mask(mask),
                 'responses': ids[:, prompt_width:].clone(), 'response_mask': mask[:, prompt_width:].clone(),
                 'prompts': ids[:, :prompt_width].clone()},
        non_tensors={'instruction': np.array(['a'] * rows, dtype=object),
                     'uid': np.array(['u'] * rows, dtype=object)},
        meta_info={'global_token_num': mask.sum(-1).tolist()})


def source_module(name, relative):
    source_root = os.environ.get('TGPO_SOURCE_ROOT')
    if not source_root:
        pytest.skip('set TGPO_SOURCE_ROOT to run original/public differential tests')
    path = Path(source_root) / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selection_threshold_and_ties():
    assert select_refinement_indices(['a'] * 4, [0, .2, .5, .5], 4, 4, .51) == [0, 1, 2, 3]
    assert select_refinement_indices(['a'] * 4, [0, .2, .5, .5], 4, 4, .5) == []
    assert best_candidate_indices([.5, .5, .1, .2], 2) == [0, 3]
    assert best_candidate_indices([], 2) == []
    with pytest.raises(ValueError):
        select_refinement_indices(['a'] * 3, [0] * 3, 2, 1, 1)


def test_original_selection_randomized():
    old = source_module('source_selection', 'RL_framework/verl_fix_cot/custom/fix_manager/fix_cot_utils.py')
    generator = torch.Generator().manual_seed(20261024)
    for n in [2, 4, 8]:
        for select in [0, 1, 2, n, n + 1]:
            for _ in range(8):
                batch = make_batch(rows=n * 4)
                instructions = np.array(['same'] * (2*n) + ['other'] * (2*n), dtype=object)
                perm = torch.randperm(4*n, generator=generator)
                batch.non_tensor_batch['instruction'] = instructions[perm]
                rewards = (torch.randint(0, 5, (4*n,), generator=generator) / 4).numpy()
                _, expected = old.filter_fix_cot_batch(batch, rewards, n, select, .76)
                assert select_refinement_indices(batch.non_tensor_batch['instruction'], rewards, n, select, .76) == expected
                _, best = old.choose_best_fix_cot_batch(batch, rewards, n)
                assert best_candidate_indices(rewards, n) == best


def test_original_prefix_and_merge_exact_legacy_mode(tokenizer):
    old = source_module('source_batch_ops', 'RL_framework/verl_fix_cot/custom/fix_manager/concate_cot.py')
    for output_format in ['subgoal', 'e2e']:
        batch = make_batch()
        batch.non_tensor_batch['fix_cot'] = np.array(['a\nb', 'c', 'd\ne', 'f'], dtype=object)
        prefixes = [format_repair_prefix(c, output_format, 'generation') for c in batch.non_tensor_batch['fix_cot']]
        expected = old.concate_fix_cot(old.get_fix_gen_batch(copy.deepcopy(batch)), tokenizer, output_format)
        actual = continuation_prompts(copy.deepcopy(batch), tokenizer, prefixes, position_policy='legacy')
        assert expected.batch.keys() == actual.batch.keys()
        for key in actual.batch.keys():
            torch.testing.assert_close(expected.batch[key], actual.batch[key], rtol=0, atol=0)
        generated = DataProto.from_dict(tensors={'responses': batch.batch['responses'].clone(),
                                                'attention_mask': batch.batch['attention_mask'].clone()})
        expected_merge = old.merge_fix_cot_output(copy.deepcopy(batch), copy.deepcopy(generated), tokenizer)
        response_prefixes = [format_repair_prefix(c, output_format, 'response') for c in batch.non_tensor_batch['fix_cot']]
        actual_merge = merge_continuations(copy.deepcopy(batch), generated, tokenizer, response_prefixes, position_policy='legacy')
        for key in actual_merge.batch.keys():
            torch.testing.assert_close(expected_merge.batch[key], actual_merge.batch[key], rtol=0, atol=0)


def run_refinement(tokenizer, *, select=2, repairs=None):
    batch = make_batch()
    original = copy.deepcopy(batch)
    rewards = torch.zeros(4, 128)
    scores = [0, .2, .5, .5]
    rewards[:, 4] = torch.tensor(scores)
    info = {'score': scores, 'evaluate': ['original'] * 4, 'reply': ['original reply'] * 4,
            'reward_validate': [0] * 4}
    generated_sizes = []
    def generate(prompts):
        generated_sizes.append(len(prompts))
        assert prompts.batch['input_ids'].shape == prompts.batch['position_ids'].shape
        responses = torch.full((len(prompts), 128), 97, dtype=torch.long)
        return DataProto.from_dict(tensors={'responses': responses,
                     'attention_mask': torch.cat((prompts.batch['attention_mask'], torch.ones_like(responses)), -1)})
    def reward(candidates):
        assert len(candidates) == 4
        result = torch.zeros(4, 128)
        candidate_scores = [.5, .75, .5, .5]
        result[:, -1] = torch.tensor(candidate_scores)
        return result, {'score': candidate_scores, 'evaluate': ['short', 'longer improved feedback', 'x', 'y'],
                       'reply': ['candidate-0', 'candidate-1-long', 'candidate-2', 'candidate-3'],
                       'reward_validate': [1] * 4}
    output = refine_rollouts(batch, rewards, info, tokenizer=tokenizer, generate=generate, reward=reward,
        repair=lambda records: repairs if repairs is not None else [{'fix_cot': 'a\nb'} for _ in records],
        format_prefix=format_repair_prefix,
        config=RefinementConfig(rollout_n=4, select_num=select, continuations_per_sample=2, reward_threshold=.51),
        generation_divisor=3)
    return original, generated_sizes, output


def test_refinement_improvement_padding_and_metadata(tokenizer):
    original, sizes, (batch, reward, info, metrics) = run_refinement(tokenizer)
    assert sizes == [6]  # 4 candidates padded to 2 worker chunks of 3.
    assert metrics == {'tgpo/selected': 2, 'tgpo/repaired': 2, 'tgpo/improved': 1}
    changed = [i for i, score in enumerate(info['score']) if score == .75]
    assert len(changed) == 1
    i = changed[0]
    assert info['evaluate'][i] == batch.non_tensor_batch['evaluate'][i] == 'longer improved feedback'
    assert info['reply'][i] == batch.non_tensor_batch['reply'][i] == 'candidate-1-long'
    assert float(reward[i].sum()) == .75
    assert (batch.non_tensor_batch['uid'] == original.non_tensor_batch['uid']).all()
    torch.testing.assert_close(batch.batch['position_ids'], compute_position_id_with_mask(batch.batch['attention_mask']))
    assert batch.meta_info['global_token_num'] == batch.batch['attention_mask'].sum(-1).tolist()
    assert tokenizer.padding_side == 'left'
    # All unchanged rows retain original response tensors, including equal-score candidate.
    for j in set(range(4)) - {i}:
        torch.testing.assert_close(batch.batch['responses'][j], original.batch['responses'][j], rtol=0, atol=0)


@pytest.mark.parametrize('select,repairs', [(0, None), (2, [{'fix_cot': ''}, {'fix_cot': None}])])
def test_empty_and_failed_repairs_retain_rows(tokenizer, select, repairs):
    original, sizes, (batch, rewards, info, metrics) = run_refinement(tokenizer, select=select, repairs=repairs)
    assert sizes == []
    assert metrics['tgpo/improved'] == 0
    for key in original.batch.keys():
        torch.testing.assert_close(batch.batch[key], original.batch[key], rtol=0, atol=0)
