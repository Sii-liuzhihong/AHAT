"""Optional real GPU test against the original vLLM wrapper (no mock engine).

Set TGPO_SOURCE_ROOT and TGPO_INFERENCE_MODEL to local paths. Restrict the GPU
with CUDA_VISIBLE_DEVICES before invoking this test. A tiny random model tests
execution and parity; it does not test model quality.
"""

import ast
import importlib.util
import os
from pathlib import Path
import typing
import unittest


@unittest.skipUnless(os.environ.get("TGPO_SOURCE_ROOT") and os.environ.get("TGPO_INFERENCE_MODEL"),
                     "Set TGPO_SOURCE_ROOT and TGPO_INFERENCE_MODEL for real vLLM parity")
class InferenceIntegration(unittest.TestCase):
    def test_original_and_public_generation(self):
        from vllm import SamplingParams
        from robot_planning.data.io import read_records
        from robot_planning.data.prompts import build_inference_messages
        from robot_planning.inference import GenerationConfig, generate_predictions
        from robot_planning.prompts import load_prompt
        source = Path(os.environ["TGPO_SOURCE_ROOT"])

        def load(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

        original = load("original_vllm", source / "policy_iteration/iter/vllm_inference/vllm_inference.py")
        prompt_utils = load("original_prompt_utils", source / "policy_iteration/iter/vllm_inference/prompt_utils.py")
        tree = ast.parse((source / "policy_iteration/iter/data_process/subgoal_data_process/build_subgoal_data.py").read_text())
        namespace = {"Dict": typing.Dict, "List": typing.List, "Optional": typing.Optional, "Union": typing.Union}
        selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "make_format_data"]
        exec(compile(ast.Module(body=selected, type_ignores=[]), "<source make_format_data>", "exec"), namespace)
        examples = Path(__file__).resolve().parents[2] / "recipe/tgpo/examples"
        task = read_records(examples / "tasks.jsonl")[0]
        engine = original.VLLMInference(model_name_or_path=os.environ["TGPO_INFERENCE_MODEL"], tensor_parallel_size=1,
                                        data_parallel_size=1, max_model_len=8192, gpu_memory_utilization=0.12,
                                        enforce_eager=True)
        tokenizer = engine.model.get_tokenizer()
        formatted = namespace["make_format_data"]("subgoal_two_stage_grounding",
                                                   load_prompt("subgoal_system_prompt_with_predicate_0112"), task)
        prompts = prompt_utils.build_prompts_from_dataset([formatted], tokenizer=tokenizer, use_chat_template=True)
        self.assertEqual(prompts, [tokenizer.apply_chat_template(build_inference_messages(task), tokenize=False,
                                                                 add_generation_prompt=True)])
        expected = engine.inference(input_text=prompts, sampling_params=SamplingParams(temperature=0, top_p=0.9,
                                                                                    max_tokens=16, n=1, seed=7))
        actual = generate_predictions([task], engine.model, config=GenerationConfig(temperature=0, max_tokens=16, seed=7),
                                      data_path=examples / "tasks.jsonl")
        self.assertEqual(actual[0]["reply_list"], expected)


if __name__ == "__main__":
    unittest.main()
