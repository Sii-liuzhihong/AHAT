"""Native Ray/FSDP/vLLM smoke harness with explicit external task fixtures.

Run from repository root on one available GPU, after installing training deps:
  CUDA_VISIBLE_DEVICES=1 python -m scripts.validation.native_training \
    --model /tmp/tiny-qwen --output /tmp/tgpo-native --mode tgpo

mode source executes the original private fit and batch functions unchanged.
source-corrected additionally adapts malformed position IDs in memory; the
original files remain untouched. TGPO_SOURCE_ROOT locates that private source.
"""
import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import uuid

from verl.trainer.ppo import ray_trainer as shared
from verl.trainer.tgpo.trainer import TGPOTrainer
from verl.utils.model import compute_position_id_with_mask


def _source_fit(correct_positions=False):
    source_root = Path(os.environ['TGPO_SOURCE_ROOT'])
    source = source_root / 'RL_framework/verl_fix_cot'
    def module(name, relative):
        spec = importlib.util.spec_from_file_location(name, source / relative)
        result = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(result)
        return result
    selection = module('tgpo_original_selection', 'custom/fix_manager/fix_cot_utils.py')
    batch_ops = module('tgpo_original_batch_ops', 'custom/fix_manager/concate_cot.py')
    namespace = dict(vars(shared))
    for name in ('filter_fix_cot_batch', 'choose_best_fix_cot_batch'):
        namespace[name] = getattr(selection, name)
    for name in ('get_fix_gen_batch', 'concate_fix_cot', 'merge_fix_cot_output'):
        namespace[name] = getattr(batch_ops, name)
    if correct_positions:
        def concatenate(*args, **kwargs):
            batch = batch_ops.concate_fix_cot(*args, **kwargs)
            batch.batch['position_ids'] = compute_position_id_with_mask(batch.batch['attention_mask'])
            return batch
        def merge(*args, **kwargs):
            batch = batch_ops.merge_fix_cot_output(*args, **kwargs)
            batch.batch['position_ids'] = compute_position_id_with_mask(batch.batch['attention_mask'])
            return batch
        namespace.update(concate_fix_cot=concatenate, merge_fix_cot_output=merge)
    # Only external repair and excluded debug logging are replaced with fixtures.
    namespace['record_sample_information'] = lambda *args, **kwargs: None
    namespace['fix_tree_structure'] = lambda batch, **kwargs: {
        'fix_cot': ['Move to kitchen.'] * len(batch),
        'response': ['<fixed_cot>Move to kitchen.</fixed_cot>'] * len(batch)}
    path = source / 'custom/trainer/custom_ray_trainer.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RayPPOTrainer')
    fit = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'fit')
    exec(compile(ast.Module(body=[fit], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['fit']


class CaptureNativeBatch:
    """Persist actual dispatch inputs without replacing the native update worker."""
    def init_workers(self):
        super().init_workers()
        update = self.actor_rollout_wg.update_actor
        def capture(batch):
            import torch
            path = Path(self.config.validation_output) / f"native_batch_step_{self.global_steps}.pt"
            torch.save({key: value.detach().cpu().clone() for key, value in batch.batch.items()}, path)
            fields = batch.non_tensor_batch
            metadata = {"keys": sorted(fields), "rows": len(batch)}
            if "extra_info" in fields and "instruction" in fields:
                metadata["instruction_matches_extra_info"] = all(
                    info.get("instruction") == instruction
                    for info, instruction in zip(fields["extra_info"], fields["instruction"], strict=True))
            path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            return update(batch)
        self.actor_rollout_wg.update_actor = capture


class NativeTGPOTrainer(CaptureNativeBatch, TGPOTrainer):
    def postprocess_rollouts(self, batch, reward_tensor, reward_info, metrics):
        result = super().postprocess_rollouts(batch, reward_tensor, reward_info, metrics)
        path = Path(self.config.validation_output) / f"refinement_step_{self.global_steps}.json"
        path.write_text(json.dumps({key: value for key, value in metrics.items() if key.startswith("tgpo/")}, indent=2) + "\n")
        return result


class NativeGRPOTrainer(CaptureNativeBatch, shared.RayPPOTrainer):
    pass


class SourceTrainer(NativeTGPOTrainer):
    def fit(self):
        return _source_fit(False)(self)


class CorrectedSourceTrainer(NativeTGPOTrainer):
    def fit(self):
        return _source_fit(True)(self)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=['grpo', 'tgpo', 'source', 'source-corrected'], default='tgpo')
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--position-policy', choices=['recompute', 'legacy'], default='recompute')
    parser.add_argument('--canonical-tasks', help='Prepare this canonical task JSONL with the public RL converter')
    parser.add_argument('--real-task-reward', action='store_true', help='Use actual robot_planning reward in only_validate mode')
    args = parser.parse_args()
    if args.real_task_reward and not args.canonical_tasks:
        parser.error('--real-task-reward requires --canonical-tasks')
    import pandas as pd
    import ray
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from verl.trainer.main_ppo import run_ppo
    import verl.trainer.main_ppo as entry

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Short synthetic prompts isolate native training from long task prompts and
    # from private data. A real task eval is tested separately through its CLI.
    rows = []
    for index in range(4):
        rows.append({'prompt': [{'role': 'user', 'content': f'Move to kitchen {index}.'}],
                     'instruction': f'Move to kitchen {index}.',
                     'scene_graph': '{}', 'data_source': 'native_validation',
                     'reward_model': {'style': 'rule', 'ground_truth': ''},
                     'extra_info': {'index': index}})
    if args.canonical_tasks:
        from robot_planning.data.io import read_records
        from robot_planning.data.prepare_rl import convert_records
        tasks = read_records(args.canonical_tasks)
        rows = convert_records([tasks[i % len(tasks)] for i in range(4)], data_path=args.canonical_tasks)
        if any('instruction' in row or 'scene_graph' in row for row in rows):
            raise AssertionError('Canonical task metadata must be stored only in extra_info')
    data_path = output / 'train.parquet'
    pd.DataFrame(rows).to_parquet(data_path, index=False)
    config_dir = Path(entry.__file__).parent / 'config'
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name='tgpo_trainer' if args.mode != 'grpo' else 'ppo_trainer')
    OmegaConf.set_struct(config, False)
    config.algorithm.adv_estimator = 'grpo'
    config.data.train_files = config.data.val_files = [str(data_path)]
    config.data.train_batch_size = 2
    config.data.val_batch_size = 2
    config.data.max_prompt_length = 6144 if args.canonical_tasks else 128
    config.data.max_response_length = 96
    config.data.dataloader_num_workers = 0
    config.data.shuffle = False
    config.actor_rollout_ref.model.path = str(Path(args.model).resolve())
    config.actor_rollout_ref.model.use_remove_padding = False
    config.actor_rollout_ref.model.enable_gradient_checkpointing = False
    config.actor_rollout_ref.actor.ppo_mini_batch_size = 2
    config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu = 1
    config.actor_rollout_ref.actor.use_torch_compile = False
    config.actor_rollout_ref.actor.optim.lr = 1e-4
    config.actor_rollout_ref.actor.fsdp_config.param_offload = False
    config.actor_rollout_ref.actor.fsdp_config.optimizer_offload = False
    rollout = config.actor_rollout_ref.rollout
    rollout.name = 'vllm'
    rollout.mode = 'sync'
    rollout.n = 2
    rollout.tensor_model_parallel_size = 1
    rollout.gpu_memory_utilization = .15
    rollout.enforce_eager = True
    rollout.enable_chunked_prefill = False
    rollout.enable_prefix_caching = False
    rollout.max_model_len = config.data.max_prompt_length + config.data.max_response_length
    rollout.max_num_batched_tokens = 8192 if args.canonical_tasks else 512
    rollout.max_num_seqs = 8
    rollout.log_prob_micro_batch_size_per_gpu = 1
    rollout.log_prob_use_dynamic_bsz = False
    rollout.temperature = 1.0
    rollout.top_p = 1.0
    rollout.top_k = -1
    rollout.load_format = 'dummy_dtensor'
    rollout.free_cache_engine = True
    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu = 1
    config.reward_model.reward_manager = 'batch'
    callback_path = Path(__file__).with_name('native_callbacks.py').resolve()
    config.custom_reward_function.path = str(callback_path)
    config.custom_reward_function.name = 'compute_score_batch'
    if args.real_task_reward:
        import robot_planning.reward as task_reward
        config.custom_reward_function.path = task_reward.__file__
        config.custom_reward_function.reward_kwargs = {'eval_mode': 'only_validate', 'max_workers': 1}
    config.validation_output = str(output)
    trainer = config.trainer
    trainer.trainer_class = "scripts.validation.native_training.NativeGRPOTrainer" if args.mode == "grpo" else "scripts.validation.native_training.NativeTGPOTrainer"
    trainer.n_gpus_per_node = trainer.nnodes = 1
    trainer.total_epochs = 2
    trainer.total_training_steps = args.steps
    trainer.logger = ['console']
    trainer.project_name = 'native-tgpo-validation'
    trainer.experiment_name = args.mode
    trainer.default_local_dir = str(output / 'checkpoints')
    trainer.default_hdfs_dir = None
    trainer.val_before_train = False
    trainer.test_freq = -1
    trainer.save_freq = 1
    trainer.resume_mode = 'auto' if args.resume else 'disable'
    trainer.balance_batch = True
    if args.mode != 'grpo':
        config.tgpo.select_num = 1
        config.tgpo.continuations_per_sample = 2
        config.tgpo.reward_threshold = .51
        config.tgpo.position_policy = args.position_policy
        config.task.repair.callable = 'scripts.validation.native_callbacks.repair_batch_records'
        trainer.fix_cot_select_num = 1
        trainer.fix_cot_repeat_times = 2
        trainer.fix_cot_threshold = .51
        trainer.model_format = 'subgoal'
        if args.mode == 'source':
            trainer.trainer_class = 'scripts.validation.native_training.SourceTrainer'
        elif args.mode == 'source-corrected':
            trainer.trainer_class = 'scripts.validation.native_training.CorrectedSourceTrainer'
    # Each run owns a fresh local Ray cluster. Shutdown cannot affect another run.
    config.ray_kwargs.ray_init = {'address': 'local', 'num_cpus': 4,
        'include_dashboard': False, 'object_store_memory': 512 * 1024 * 1024,
        '_temp_dir': '/tmp/tgpo-ray-' + uuid.uuid4().hex[:8]}
    (output / 'resolved_config.yaml').write_text(OmegaConf.to_yaml(config, resolve=True))
    try:
        run_ppo(config)
        import torch
        from safetensors.torch import load_file
        checkpoint_dir = output / 'checkpoints' / f'global_step_{args.steps}' / 'actor'
        checkpoint = torch.load(checkpoint_dir / 'model_world_size_1_rank_0.pt', weights_only=False, map_location='cpu')
        initial = load_file(Path(args.model) / 'model.safetensors')
        max_change = max(float((tensor - initial[key]).abs().max()) for key, tensor in checkpoint.items() if key in initial)
        if max_change == 0:
            raise AssertionError('Native training completed without changing model parameters')
        batch = torch.load(output / f'native_batch_step_{args.steps}.pt', weights_only=True)
        scores = batch['token_level_scores'].sum(-1).tolist()
        expected_scores = ([0.0] * 4 if args.real_task_reward else
                           [0.0, 0.25, 0.0, 0.25] if args.mode == 'grpo' else [0.0, 1.0, 0.0, 1.0])
        if scores != expected_scores:
            raise AssertionError(f'Unexpected native refinement scores: {scores}')
        state = {'status': 'passed', 'mode': args.mode, 'steps': args.steps,
                 'resume': args.resume, 'position_policy': args.position_policy, 'checkpoint': str(output / 'checkpoints'),
                 'checkpoint_tensor_count': len(checkpoint), 'max_parameter_change': max_change,
                 'final_scores': scores, 'dispatch_tensor_count': len(batch),
                 'external_task_fixtures': str(callback_path),
                 'canonical_data': bool(args.canonical_tasks), 'real_task_reward': args.real_task_reward}
        (output / ('resume_result.json' if args.resume else 'result.json')).write_text(json.dumps(state, indent=2) + '\n')
    finally:
        ray.shutdown()


if __name__ == '__main__':
    main()
