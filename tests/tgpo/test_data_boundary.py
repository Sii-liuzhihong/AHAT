"""Exercise the public converter and actual verl dataset at the trainer boundary."""
from pathlib import Path

import numpy as np
import pandas as pd

from verl import DataProto
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from robot_planning.data.io import read_records
from robot_planning.data.prepare_rl import convert_records
from test_training_loop_parity import build_trainer


def test_canonical_parquet_context_reaches_tgpo(tokenizer, tmp_path):
    path = Path(__file__).parents[2] / 'recipe/tgpo/examples/tasks.jsonl'
    tasks = read_records(path)
    converted = convert_records(tasks, data_path=path)
    assert 'instruction' not in converted[0] and 'scene_graph' not in converted[0]
    data_path = tmp_path / 'train.parquet'
    pd.DataFrame(converted).to_parquet(data_path, index=False)
    tokenizer.chat_template = "{% for m in messages %}{{ m['role'] + ': ' + m['content'] + '\n' }}{% endfor %}assistant: "
    trainer = build_trainer(tokenizer)
    trainer.config.data.max_prompt_length = 20000
    trainer.config.data.filter_overlong_prompts = False
    dataset = RLHFDataset([str(data_path)], tokenizer=tokenizer, config=trainer.config.data)
    raw = DataProto.from_single_dict(collate_fn([dataset[0]]))
    assert 'instruction' not in raw.non_tensor_batch
    generation = trainer._get_gen_batch(raw).repeat(4, interleave=True)
    generated = trainer.actor_rollout_wg.generate_sequences(generation)
    batch = raw.repeat(4, interleave=True).union(generated)
    assert 'instruction' not in batch.non_tensor_batch
    # Zero threshold deliberately tests context hydration without external API calls.
    trainer.config.tgpo.reward_threshold = 0
    reward = batch.batch['responses'].float() * 0
    info = {'score': [0] * len(batch), 'reply': [''] * len(batch), 'evaluate': [''] * len(batch)}
    final, _, _ = trainer.postprocess_rollouts(batch, reward, info, {})
    assert list(final.non_tensor_batch['instruction']) == [tasks[0]['instruction']] * 4
    assert all('furnitures' in str(scene) for scene in final.non_tensor_batch['scene_graph'])
    assert final.non_tensor_batch['instruction'].dtype == np.dtype(object)
    assert all('instruction' in row for row in final.non_tensor_batch['extra_info'])


def test_reward_loader_does_not_reuse_another_file(tmp_path, monkeypatch):
    import sys
    from omegaconf import OmegaConf
    from verl.trainer.ppo.reward import get_custom_reward_fn
    monkeypatch.delitem(sys.modules, 'custom_module', raising=False)
    first, second = tmp_path / 'first.py', tmp_path / 'second.py'
    first.write_text('def compute_score(**kwargs):\n    return 1\n')
    second.write_text('def compute_score(**kwargs):\n    return 2\n')
    def load(path):
        return get_custom_reward_fn(OmegaConf.create({'custom_reward_function': {
            'path': str(path), 'name': 'compute_score'}}))
    a, b = load(first), load(second)
    assert a() == 1 and b() == 2
    assert load(first)() == 1
