"""Optional actual parquet→dataset→verl manager→task reward→planner test."""

import os
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(os.environ.get("TGPO_TOKENIZER") and os.environ.get("TGPO_SOLVER"),
                     "Set TGPO_TOKENIZER and TGPO_SOLVER for production reward integration")
class ProductionRewardBoundary(unittest.TestCase):
    def test_canonical_parquet_through_configured_manager(self):
        import torch
        from omegaconf import OmegaConf
        from transformers import AutoTokenizer
        from verl import DataProto
        from verl.trainer.ppo.reward import compute_reward, load_reward_manager
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
        from verl.utils.model import compute_position_id_with_mask
        from robot_planning.data.io import read_records, write_parquet
        from robot_planning.data.prepare_rl import convert_records
        from robot_planning.clients.llm import OpenAICompatibleClient
        from robot_planning.repair import extract_task_decomposition, repair_batch_records
        from verl.trainer.tgpo.trainer import TGPOTrainer
        from types import SimpleNamespace
        import robot_planning.reward

        examples = Path(__file__).resolve().parents[2] / "recipe/tgpo/examples"
        task = read_records(examples / "tasks.jsonl")[0]
        tokenizer = AutoTokenizer.from_pretrained(os.environ["TGPO_TOKENIZER"], local_files_only=True)
        config = OmegaConf.create({
            "data": {"max_prompt_length": 8192, "truncation": "error", "filter_overlong_prompts": False,
                     "reward_fn_key": "data_source"},
            "reward_model": {"reward_manager": "batch"},
            "custom_reward_function": {"path": robot_planning.reward.__file__,
                                       "name": "compute_total_score_subgoal_batch",
                                       "reward_kwargs": {"eval_mode": "only_validate", "max_workers": 1,
                                                         "solver_path": os.environ["TGPO_SOLVER"]}}})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rl.parquet"
            write_parquet(convert_records([task, task], data_path=examples / "tasks.jsonl"), path)
            dataset = RLHFDataset(str(path), tokenizer, config.data)
            batch = DataProto.from_single_dict(collate_fn([dataset[0], dataset[1]]))
            self.assertNotIn("instruction", batch.non_tensor_batch)
            self.assertNotIn("scene_graph", batch.non_tensor_batch)
            prompts, mask = batch.batch["input_ids"].clone(), batch.batch["attention_mask"].clone()
            replies = [task["output"], "malformed prediction"]
            ids = [tokenizer.encode(reply, add_special_tokens=False) + [tokenizer.eos_token_id] for reply in replies]
            width = max(map(len, ids)) + 8
            responses = torch.full((2, width), tokenizer.pad_token_id, dtype=torch.long)
            response_mask = torch.zeros_like(responses)
            for row, tokens in enumerate(ids):
                responses[row, :len(tokens)] = torch.tensor(tokens)
                response_mask[row, :len(tokens)] = 1
            batch.batch["prompts"], batch.batch["responses"] = prompts, responses
            batch.batch["input_ids"] = torch.cat((prompts, responses), dim=-1)
            batch.batch["attention_mask"] = torch.cat((mask, response_mask), dim=-1)
            batch.batch["position_ids"] = compute_position_id_with_mask(batch.batch["attention_mask"])
            manager = load_reward_manager(config, tokenizer, num_examine=0)
            reward, info = compute_reward(batch, manager)
            self.assertEqual(info["score"], [1, 0])
            self.assertEqual(reward.sum(-1).tolist(), [1.0, 0.0])
            self.assertEqual(info["reply"], replies)
            self.assertEqual(float(reward[0, len(ids[0]) - 1]), 1.0)
            self.assertEqual(float(reward[0, :len(ids[0]) - 1].sum()), 0.0)
            config.task = {"metadata_fields": ["instruction", "scene_graph"], "output_format": "subgoal",
                           "prefix_formatter": "robot_planning.repair.format_repair_prefix",
                           "repair": {"callable": "robot_planning.repair.repair_batch_records", "kwargs": {}}}
            config.tgpo = {"select_num": 1, "continuations_per_sample": 1, "reward_threshold": 0.51,
                           "position_policy": "recompute"}
            config.actor_rollout_ref = {"rollout": {"n": 2}}
            trainer = TGPOTrainer.__new__(TGPOTrainer)
            trainer.config, trainer.tokenizer, trainer.reward_fn = config, tokenizer, manager
            trainer.async_rollout_mode = False
            def unexpected_generation(_batch):
                raise AssertionError("Group with a score-1 response must not be refined at threshold 0.51")
            trainer.actor_rollout_wg = SimpleNamespace(generate_sequences=unexpected_generation, world_size=1)
            metrics = {}
            trainer.postprocess_rollouts(batch, reward, info, metrics)
            self.assertEqual(batch.non_tensor_batch["instruction"].tolist(), [task["instruction"]] * 2)
            self.assertEqual(metrics["tgpo/selected"], 0)
            records = [{key: values[index] for key, values in batch.non_tensor_batch.items()} for index in range(2)]
            # Exercise the production HTTP failure boundary against a closed
            # local port; source semantics retain usable original CoT prefixes.
            client = OpenAICompatibleClient(base_url="http://127.0.0.1:1/v1", model="boundary-test",
                                             api_key="", max_retries=1, timeout=0.5)
            repairs = repair_batch_records(records, client=client)
            self.assertEqual(repairs[0]["fix_cot"], extract_task_decomposition(task["output"]))
            self.assertEqual([row["status"] for row in repairs], ["fallback", "failed"])


if __name__ == "__main__":
    unittest.main()
