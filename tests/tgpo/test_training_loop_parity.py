"""Run both full training loops with identical deterministic rollouts and Torch updates.

This isolates loop/GRPO semantics from stochastic GPU generation and hosted LLMs.
It is not a substitute for the separate native worker/vLLM integration smoke run.
"""
import ast
import copy
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo import ray_trainer as shared
from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla
from verl.trainer.tgpo.trainer import TGPOTrainer
from verl.utils.model import compute_position_id_with_mask
from test_refinement import source_module


def repair_records(records, **kwargs):
    return [{'fix_cot': 'a\nb', 'response': '<fixed_cot>a\nb</fixed_cot>'} for _ in records]


class TorchActor:
    """Small real autoregressive Torch model and optimizer behind the worker boundary."""
    world_size = 1

    def __init__(self, config):
        torch.manual_seed(11)
        self.model = torch.nn.Sequential(torch.nn.Embedding(128, 8), torch.nn.Linear(8, 128))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=.002)
        self.initial = copy.deepcopy(self.model.state_dict())
        self.config = config
        self.events = []
        self.generations = 0
        self.saved_batch = None
        self.loss = None

    def generate_sequences(self, prompts):
        self.events.append('generate')
        self.generations += 1
        rows = len(prompts)
        responses = torch.zeros(rows, 128, dtype=torch.long)
        if "fix_cot" not in prompts.non_tensor_batch:
            for i in range(rows):
                responses[i, :5] = torch.tensor([97 + i, 101, 108, 108, 111])
            mask = torch.zeros_like(responses)
            mask[:, :5] = 1
        else:
            # All candidates have a valid continuation; reward breaks ties below.
            responses[:, :8] = torch.tensor([103, 114, 111, 117, 110, 100, 101, 100])
            mask = torch.zeros_like(responses)
            mask[:, :8] = 1
        ids = torch.cat((prompts.batch['input_ids'], responses), -1)
        attention = torch.cat((prompts.batch['attention_mask'], mask), -1)
        return DataProto.from_dict(tensors={
            'prompts': prompts.batch['input_ids'].clone(), 'responses': responses,
            'input_ids': ids, 'attention_mask': attention,
            'position_ids': compute_position_id_with_mask(attention)},
            non_tensors=copy.deepcopy(prompts.non_tensor_batch), meta_info={'timing': {}})

    def log_probs(self, batch):
        width = batch.batch['responses'].shape[-1]
        logits = self.model(batch.batch['input_ids'][:, :-1])[:, -width:]
        log_probs = logits.log_softmax(-1)
        selected = log_probs.gather(-1, batch.batch['responses'].unsqueeze(-1)).squeeze(-1)
        entropy = -(log_probs.exp() * log_probs).sum(-1)
        return selected, entropy

    def compute_log_prob(self, batch):
        self.events.append('log_prob')
        with torch.no_grad():
            probs, entropy = self.log_probs(batch)
        return DataProto.from_dict(tensors={'old_log_probs': probs, 'entropys': entropy})

    def update_actor(self, batch):
        self.events.append('update')
        self.saved_batch = copy.deepcopy(batch)
        probs, _ = self.log_probs(batch)
        loss, *_ = compute_policy_loss_vanilla(batch.batch['old_log_probs'], probs,
            batch.batch['advantages'], batch.batch['response_mask'], config=self.config.actor_rollout_ref.actor)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss = loss.detach()
        return DataProto(meta_info={'metrics': {'actor/pg_loss': [float(self.loss)]}})


class Reward:
    def __init__(self, candidate_improves=True):
        self.calls = 0
        self.candidate_improves = candidate_improves

    def __call__(self, batch, return_dict=False):
        self.calls += 1
        scores = ([.4, .75, .5, .5] if self.candidate_improves else [0, 0, 0, 0]) if "fix_cot" in batch.non_tensor_batch else [0, .2, .4, .5]
        assert len(batch) == len(scores)
        values = torch.zeros_like(batch.batch['responses'], dtype=torch.float32)
        lengths = batch.batch['response_mask'].sum(-1).long()
        for i, score in enumerate(scores):
            values[i, lengths[i]-1] = score
        info = {'score': scores, 'evaluate': ['original'] * len(batch),
                'reply': ['original reply'] * len(batch), 'reward_validate': [0] * len(batch)}
        return {'reward_tensor': values, 'reward_extra_info': info} if return_dict else values


def original_fit():
    src = os.environ.get('TGPO_SOURCE_ROOT')
    if not src:
        pytest.skip('set TGPO_SOURCE_ROOT for full original/public loop differential')
    old_select = source_module('source_fit_select', 'RL_framework/verl_fix_cot/custom/fix_manager/fix_cot_utils.py')
    old_batch = source_module('source_fit_batch', 'RL_framework/verl_fix_cot/custom/fix_manager/concate_cot.py')
    path = Path(src) / 'RL_framework/verl_fix_cot/custom/trainer/custom_ray_trainer.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RayPPOTrainer')
    fit = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'fit')
    namespace = dict(vars(shared))
    namespace.update({name: getattr(old_select, name) for name in ['filter_fix_cot_batch', 'choose_best_fix_cot_batch']})
    namespace.update({name: getattr(old_batch, name) for name in ['get_fix_gen_batch', 'concate_fix_cot', 'merge_fix_cot_output']})
    namespace['record_sample_information'] = lambda *args, **kwargs: None  # excluded debug logging only
    namespace['fix_tree_structure'] = lambda batch, **kwargs: {
        'fix_cot': ['a\nb'] * len(batch), 'response': ['<fixed_cot>a\nb</fixed_cot>'] * len(batch)}
    exec(compile(ast.Module(body=[fit], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['fit']


def build_trainer(tokenizer, select=2, candidate_improves=True, steps=1):
    config_dir = Path(shared.__file__).parents[1] / 'config'
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name='tgpo_trainer')
    OmegaConf.set_struct(config, False)
    config.trainer.logger = []
    config.trainer.total_epochs = 1
    config.trainer.balance_batch = True
    config.trainer.save_freq = -1
    config.trainer.test_freq = -1
    config.trainer.fix_cot_select_num = select
    config.trainer.fix_cot_repeat_times = 2
    config.trainer.fix_cot_threshold = .51
    config.trainer.model_format = 'subgoal'
    config.actor_rollout_ref.rollout.n = 4
    config.actor_rollout_ref.actor.tis_imp_ratio_cap = -1
    config.tgpo.select_num = select
    config.tgpo.continuations_per_sample = 2
    config.tgpo.reward_threshold = .51
    config.tgpo.position_policy = 'legacy'
    config.task.repair.callable = __name__ + '.repair_records'
    trainer = object.__new__(TGPOTrainer)
    trainer.config = config
    trainer.tokenizer = tokenizer
    trainer.async_rollout_mode = False
    trainer.use_rm = trainer.use_reference_policy = trainer.use_critic = False
    trainer.val_reward_fn = None
    trainer.reward_fn = Reward(candidate_improves)
    trainer.total_training_steps = steps
    trainer._load_checkpoint = lambda: None
    trainer._start_profiling = lambda *args: None
    trainer._stop_profiling = lambda *args: None
    trainer.resource_pool_manager = SimpleNamespace(get_n_gpus=lambda: 1)
    trainer.train_dataset = []
    ids = torch.zeros(1, 128, dtype=torch.long)
    ids[:, -8:] = torch.tensor([80, 114, 111, 109, 112, 116, 58, 32])
    mask = (ids != 0).long()
    batch_dict = {'input_ids': ids, 'attention_mask': mask, 'position_ids': compute_position_id_with_mask(mask),
                  'instruction': np.array(['instruction'], dtype=object),
                  'reward_model': np.array([{'ground_truth': ''}], dtype=object),
                  'data_source': np.array(['robot_planning'], dtype=object),
                  'extra_info': np.array([{'instruction': 'instruction', 'scene_graph': '{}'}], dtype=object)}
    class Loader:
        sampler = None
        def __iter__(self):
            for _ in range(steps):
                yield copy.deepcopy(batch_dict)
    trainer.train_dataloader = Loader()
    trainer.actor_rollout_wg = TorchActor(config)
    return trainer


@pytest.mark.parametrize("select,candidate_improves,steps", [(2, True, 1), (2, False, 1), (0, True, 1), (2, True, 2)])
def test_full_source_and_public_training_step(tokenizer, select, candidate_improves, steps):
    old = build_trainer(tokenizer, select, candidate_improves, steps)
    new = build_trainer(tokenizer, select, candidate_improves, steps)
    original_fit()(old)
    new.fit()
    a, b = old.actor_rollout_wg, new.actor_rollout_wg
    assert a.events == b.events == (['generate', 'generate', 'log_prob', 'update'] if select else ['generate', 'log_prob', 'update']) * steps
    assert old.reward_fn.calls == new.reward_fn.calls == (2 if select else 1) * steps
    for key in a.saved_batch.batch.keys():
        torch.testing.assert_close(a.saved_batch.batch[key], b.saved_batch.batch[key], rtol=0, atol=0)
    for key in a.model.state_dict():
        torch.testing.assert_close(a.model.state_dict()[key], b.model.state_dict()[key], rtol=0, atol=0)
    torch.testing.assert_close(a.loss, b.loss, rtol=0, atol=0)
    assert any(not torch.equal(value, a.initial[key]) for key, value in a.model.state_dict().items())
    assert old.global_steps == new.global_steps == steps + 1


def test_async_reward_is_resolved_before_refinement(tokenizer, monkeypatch):
    trainer = build_trainer(tokenizer)
    trainer.config.reward_model.launch_reward_fn_async = True
    events = trainer.actor_rollout_wg.events
    def start(data, reward_fn):
        events.append("reward_start")
        return SimpleNamespace(resolve=lambda: shared.compute_reward(data, reward_fn))
    def resolve(future):
        events.append("reward_resolve")
        return future.resolve()
    monkeypatch.setattr(shared, "compute_reward_async", SimpleNamespace(remote=start))
    monkeypatch.setattr(shared.ray, "get", resolve)
    trainer.fit()
    assert events == ["generate", "reward_start", "reward_resolve", "generate", "log_prob", "update"]
    assert trainer.reward_fn.calls == 2
    assert float(trainer.actor_rollout_wg.saved_batch.batch["token_level_scores"].sum(-1).max()) == .75
