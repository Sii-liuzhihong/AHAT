"""Executable protocol checks; source parity is enabled with TGPO_SOURCE_ROOT."""

import ast
import json
import os
from pathlib import Path
import tempfile
import types
import unittest

from robot_planning.clients.llm import LLMResponse
from robot_planning.data.io import read_records, write_records, write_parquet
from robot_planning.data.legacy import build_e2e_training_record, build_two_stage_reply, normalize_legacy_rooms
from robot_planning.data.prepare_rl import convert_records as rl_records
from robot_planning.data.prepare_sft import convert_records as sft_records
from robot_planning.data.prompts import build_messages, build_sft_input, format_task
from robot_planning.prompts import load_prompt
from robot_planning.repair import (extract_task_decomposition, fix_cot, format_fix_input,
                                   format_repair_prefix, parse_fix_response, repair_batch_records)
from robot_planning.reward import compute_score_batch


EXAMPLES = Path(__file__).resolve().parents[2] / "recipe/tgpo/examples"


def old_functions(relative, names, namespace=None):
    """Execute source function bodies without importing private wrappers/services."""
    root = Path(os.environ["TGPO_SOURCE_ROOT"])
    tree = ast.parse((root / relative).read_text())
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
    module = types.ModuleType("source_reference")
    module.__dict__.update(namespace or {})
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(root / relative), "exec"), module.__dict__)
    return module


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.task = read_records(EXAMPLES / "tasks.jsonl")[0]

    def test_data_and_parquet_roundtrip(self):
        rows = rl_records([self.task], data_path=EXAMPLES / "tasks.jsonl")
        self.assertEqual(rows[0]["prompt"], build_messages(self.task))
        self.assertEqual(json.loads(rows[0]["extra_info"]["scene_graph"]), self.task["scene_graph"])
        self.assertTrue(Path(rows[0]["extra_info"]["domain_file_path"]).is_file())
        sft = sft_records([self.task])
        self.assertEqual(sft[0]["answer"]["answer"], self.task["output"])
        self.assertEqual(sft[0]["prompt"]["question"], build_sft_input(self.task))
        with tempfile.TemporaryDirectory() as temporary:
            import pyarrow.parquet as pq
            for name, records in (("rl", rows), ("sft", sft)):
                path = Path(temporary) / f"{name}.parquet"
                write_parquet(records, path)
                self.assertEqual(pq.read_table(path).to_pylist(), records)
            path = Path(temporary) / "data.jsonl"
            write_records([self.task], path)
            self.assertEqual(read_records(path), [self.task])

    def test_invalid_inputs_are_not_silently_dropped(self):
        with self.assertRaises(ValueError):
            sft_records([{**self.task, "output": None}])
        with self.assertRaises(ValueError):
            rl_records([])
        with self.assertRaises(ValueError):
            build_two_stage_reply(["a"], [], [])
        scene = {"rooms": ["kitchen"]}
        self.assertEqual(normalize_legacy_rooms(scene)["rooms"][0]["name"], "kitchen")
        self.assertEqual(scene, {"rooms": ["kitchen"]})

    def test_dynamic_prompt_requires_explicit_domain(self):
        prompt = "subgoal_system_prompt_with_predicate_dynamic_domain"
        domain = (EXAMPLES / "domain.pddl").read_text()
        with self.assertRaises(ValueError):
            build_messages(self.task, prompt)
        expected = load_prompt(prompt).format(dynamic_domain=domain)
        self.assertEqual(format_task(self.task, prompt, dynamic_domain=domain)["instruction"], expected)
        record = {**self.task, "dynamic_domain_path": "domain.pddl"}
        with self.assertRaises(ValueError):
            format_task(record, prompt)
        sft = sft_records([record], prompt=prompt, data_path=EXAMPLES / "tasks.jsonl")
        rl = rl_records([record], prompt=prompt, data_path=EXAMPLES / "tasks.jsonl")
        self.assertIn(domain, sft[0]["prompt"]["question"])
        self.assertNotIn("{dynamic_domain}", rl[0]["prompt"][1]["content"])

    def test_repair_response_fallback_and_batch_order(self):
        class Client:
            def call(self, messages, **kwargs):
                value = messages[1]["content"].split("**instruction**: ")[1].split("\n")[0]
                return LLMResponse(True, f"<fixed_cot>  {value} repaired  </fixed_cot>")
        records = [{"instruction": str(index), "scene_graph": self.task["scene_graph"],
                    "reply": self.task["output"], "evaluate": "feedback"} for index in range(5)]
        result = repair_batch_records(records, client=Client(), max_workers=3)
        self.assertEqual([item["fix_cot"] for item in result], [f"{index} repaired" for index in range(5)])
        class Failed:
            def call(self, *args, **kwargs):
                return LLMResponse(False, error="failed")
        self.assertEqual(fix_cot("i", "scene", "original", "bad", client=Failed()),
                         {"fix_cot": "original", "response": "no response"})
        fallback = repair_batch_records(records[:1], client=Failed())
        self.assertEqual(fallback[0]["status"], "fallback")
        self.assertEqual(fallback[0]["fix_cot"], extract_task_decomposition(records[0]["reply"]))
        self.assertIsNone(parse_fix_response("no tag"))
        self.assertEqual(format_repair_prefix("a\nb"), "### Task Decomposition ###\n   a\n   b")
        self.assertEqual(format_repair_prefix("a", phase="response"),
                         "<think>\n### Task Decomposition (Tree-Structured CoT)\na")

    def test_reward_min_ties_keep_full_second_result(self):
        calls = []
        def evaluator(**kwargs):
            calls.append(kwargs)
            return [[{"final_result": 0.5, "evaluate": str(len(calls)), "solvable_score": len(calls)}]]
        info = {"instruction": self.task["instruction"], "scene_graph": json.dumps(self.task["scene_graph"]),
                "domain_file_path": "domain.pddl"}
        result = compute_score_batch(["task"], ["reply"], [""], [info], evaluator=evaluator, repeats=2)
        self.assertEqual(result, [{"score": 0.5, "evaluate": "2", "reply": "reply", "reward_validate": 2}])
        with self.assertRaises(ValueError):
            compute_score_batch(["task"] * 2, ["a", "b"], [""] * 2,
                                [info, {**info, "domain_file_path": "other"}], evaluator=evaluator)


@unittest.skipUnless(os.environ.get("TGPO_SOURCE_ROOT"), "Set TGPO_SOURCE_ROOT for development-source parity")
class SourceParityTests(unittest.TestCase):
    def setUp(self):
        self.task = read_records(EXAMPLES / "tasks.jsonl")[0]

    @unittest.skipUnless(os.environ.get("TGPO_TOKENIZER"), "Set TGPO_TOKENIZER for real tokenizer parity")
    def test_original_sft_tensor_outputs_with_real_tokenizer(self):
        import torch
        from transformers import AutoTokenizer
        from omegaconf import OmegaConf
        from verl.utils.dataset.sft_dataset import SFTDataset, compute_position_id_with_mask
        tokenizer = AutoTokenizer.from_pretrained(os.environ["TGPO_TOKENIZER"], local_files_only=True)
        config = OmegaConf.create({"prompt_key": "prompt", "response_key": "answer", "prompt_dict_keys": ["question"],
                                  "response_dict_keys": ["answer"], "system_prompt": "You are a helpful assistant.", "max_length": 8192})
        source = Path(os.environ["TGPO_SOURCE_ROOT"]) / "RL_framework/verl_fix_cot/verl/utils/dataset/sft_dataset.py"
        tree = ast.parse(source.read_text())
        method = next(method for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SFTDataset"
                      for method in node.body if isinstance(method, ast.FunctionDef) and method.name == "__getitem__")
        namespace = {"torch": torch, "compute_position_id_with_mask": compute_position_id_with_mask}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sft.parquet"
            write_parquet(sft_records([self.task]), path)
            dataset = SFTDataset(str(path), tokenizer, config)
            old, new = namespace["__getitem__"](dataset, 0), dataset[0]
            self.assertEqual(old.keys(), new.keys())
            for key in old:
                self.assertTrue(torch.equal(old[key], new[key]), key)

    def test_source_prompt_bytes_and_formatting(self):
        source = Path(os.environ["TGPO_SOURCE_ROOT"])
        for relative, names in [("policy_iteration/iter/data_process/train_system_prompt/subgoal_prompt/subgoal_predicate_system_prompt.py",
                                  ["subgoal_system_prompt_with_predicate_0819", "subgoal_system_prompt_with_predicate_0112", "subgoal_system_prompt_with_predicate_dynamic_domain"]),
                                 ("policy_iteration/iter/fix_plan/fix_llm_tree_structure/constants.py",
                                  ["gpt_prompt_subgoal", "subgoal_fix_cot_0102_prompt", "subgoal_fix_cot_0111_prompt", "subgoal_fix_cot_0116_prompt", "subgoal_fix_total_cot_prompt"])]:
            tree = ast.parse((source / relative).read_text())
            for node in tree.body:
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names:
                    self.assertEqual(load_prompt(node.targets[0].id), ast.literal_eval(node.value))
        old = old_functions("policy_iteration/iter/data_process/subgoal_data_process/build_subgoal_data.py",
                            ["make_format_data", "convert_to_messages_format"],
                            {"Dict": dict, "Optional": __import__("typing").Optional,
                             "Union": __import__("typing").Union, "List": list})
        for prompt in ("subgoal_system_prompt_with_predicate_0819", "subgoal_system_prompt_with_predicate_0112"):
            expected = old.make_format_data("subgoal_two_stage_grounding", load_prompt(prompt), self.task)
            self.assertEqual(format_task(self.task, prompt), {key: expected[key] for key in ("instruction", "input")})
            self.assertEqual(build_messages(self.task, prompt)[1:], old.convert_to_messages_format(expected))
            self.assertEqual(build_sft_input(self.task, prompt), expected["instruction"] + "\n" + expected["input"])
        prompt = "subgoal_system_prompt_with_predicate_dynamic_domain"
        domain = (EXAMPLES / "domain.pddl").read_text()
        expected = old.make_format_data("subgoal_two_stage_grounding", load_prompt(prompt).format(dynamic_domain=domain), self.task)
        self.assertEqual(format_task(self.task, prompt, dynamic_domain=domain),
                         {key: expected[key] for key in ("instruction", "input")})
        e2e = old_functions("RL_framework/verl_fix_cot/custom/data_process/e2e_data_process/process_e2e_domain_description_data.py",
                            ["make_map_fn"], {"qwen_instruction": "You are a helpful assistant."})
        record = {"instruction": "task", "scene_graph": json.dumps(self.task["scene_graph"]), "domain_file": "domain",
                  "system_prompt": "task instructions", "plan": ["wipe(robot, table)"], "tree_structure_cot": "cot"}
        self.assertEqual(build_e2e_training_record(record), e2e.make_map_fn("train", "robot_planning")(record, 0))
        reply = old_functions("policy_iteration/iter/data_process/subgoal_data_process/make_subgoal_reply.py",
                              ["_make_reply_subgoal_two_stage_grounding"], {"List": list, "Any": __import__("typing").Any})
        args = (["wipe table", "return"], [["(is_clean table)"], ["(agent_at robot kitchen)"]], [["table"], ["kitchen"]])
        self.assertEqual(build_two_stage_reply(*args), reply._make_reply_subgoal_two_stage_grounding(*args))

    def test_source_repair_and_reward(self):
        import re
        import typing
        from robot_planning.scene_graph.summary import sum_sg_content
        def get_content(text, name):
            match = re.search(f"<{name}>(.*?)</{name}>", text, re.DOTALL)
            return match.group(1) if match else None
        old = old_functions("policy_iteration/iter/fix_plan/fix_llm_tree_structure/fix_cot.py",
                            ["parse_fix_response", "format_fix_input", "fix_cot_cpu_server"],
                            {"Optional": typing.Optional, "Dict": dict, "List": list, "Union": typing.Union,
                             "json": json, "sum_sg_content": sum_sg_content, "get_content": get_content,
                             "ACTION_LIST_MAP": {"base_pddl_action": "unused"},
                             "SYSTEM_PROMPT": {"subgoal_fix_cot_0102_prompt": load_prompt("subgoal_fix_cot_0102_prompt")}})
        for response in ("", "no tag", "<fixed_cot> abc\n</fixed_cot>", "<fixed_cot></fixed_cot>"):
            self.assertEqual(parse_fix_response(response), old.parse_fix_response(response))
        for scene in ("scene", self.task["scene_graph"]):
            self.assertEqual(format_fix_input("instruction", scene, [], "cot", "eval"),
                             old.format_fix_input("instruction", scene, [], "cot", "eval"))
        for response in ("<fixed_cot>updated</fixed_cot>", "malformed"):
            class Client:
                def call(self, messages, **kwargs):
                    return LLMResponse(True, response)
            old.call_LLM = lambda **kwargs: response
            self.assertEqual(fix_cot("i", "s", "original", "e", client=Client()),
                             old.fix_cot_cpu_server("i", "s", "original", "e", system_prompt="subgoal_fix_cot_0102_prompt"))
        extraction = old_functions("policy_iteration/iter/utils/data_related_utils.py", ["extract_task_decomposition"])
        for reply in (self.task["output"], "not formatted", "### Task Decomposition ###\n\n a\n\n### Subgoal Grounding ###"):
            self.assertEqual(extract_task_decomposition(reply), extraction.extract_task_decomposition(reply, "subgoal"))
        source_reward = old_functions("RL_framework/verl_fix_cot/custom/reward_manager/__init__.py",
                                      ["compute_total_score_subgoal_batch", "compute_total_score_double_subgoal_batch"],
                                      {"json": json, "LLM_REVIEW_SYSTEM_PROMPT": "eval_plan_summary_1006_testing_2", "MAX_WORKERS": 128})
        infos = [{"scene_graph": json.dumps(self.task["scene_graph"]), "instruction": "i", "domain_file_path": "domain"}] * 3
        for repeats in (1, 2):
            def evaluator(**kwargs):
                return [[{"final_result": score, "evaluate": f"eval-{index}", "solvable_score": index}]
                        for index, score in enumerate([0, 0.5, 1])]
            source_reward.final_eval_batch_subgoal = evaluator
            old_fn = source_reward.compute_total_score_subgoal_batch if repeats == 1 else source_reward.compute_total_score_double_subgoal_batch
            args = (["task"] * 3, ["r0", "r1", "r2"], [""] * 3, infos)
            self.assertEqual(compute_score_batch(*args, repeats=repeats, evaluator=evaluator), old_fn(*args))


if __name__ == "__main__":
    unittest.main()
