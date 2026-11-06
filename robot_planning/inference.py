"""Canonical tasks to vLLM predictions, optionally followed by shared evaluation."""

import argparse
import hashlib
from importlib.metadata import version
from pathlib import Path
import platform
from dataclasses import asdict, dataclass
from robot_planning.data.io import read_records, write_records
from robot_planning.data.prompts import DEFAULT_PROMPT, build_inference_messages
from robot_planning.data.schema import resolve_domain, validate_task
from robot_planning.prompts import load_prompt


@dataclass
class GenerationConfig:
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 3096
    n: int = 1
    seed: int = 1


def generate_predictions(records, engine, *, prompt=DEFAULT_PROMPT, config=None,
                         data_path=None, domain=None):
    """Engine exposes get_tokenizer() and generate(prompts, sampling_params,...).

    All alternatives are kept in reply_list; no output is silently dropped.
    """
    config = config or GenerationConfig()
    tokenizer = engine.get_tokenizer()
    prompts, results = [], []
    for index, record in enumerate(records):
        validate_task(record)
        messages = build_inference_messages(record, prompt, data_path=data_path)
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        item = dict(record)
        item["domain_file_path"] = resolve_domain(record, data_path=data_path, domain=domain)
        item["generation"] = {"prompt": prompt, **asdict(config)}
        item.setdefault("id", str(index))
        results.append(item)
    from vllm import SamplingParams
    outputs = engine.generate(prompts, sampling_params=SamplingParams(**asdict(config)), use_tqdm=False)
    if len(outputs) != len(records):
        raise ValueError("Generation returned the wrong number of tasks")
    for record, output in zip(results, outputs):
        record["reply_list"] = [candidate.text for candidate in output.outputs]
        if len(record["reply_list"]) != config.n:
            raise ValueError("Generation returned the wrong number of alternatives")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--domain")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=3096)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--score-output", help="Also score predictions and write task-aligned eval_result")
    parser.add_argument("--solver-path")
    parser.add_argument("--eval-mode", default="validate_first", choices=["validate_first", "only_validate", "always_eval"])
    args = parser.parse_args()
    from vllm import LLM
    engine = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size,
                 max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization,
                 enforce_eager=args.enforce_eager)
    config = GenerationConfig(args.temperature, args.top_p, args.max_tokens, args.n, args.seed)
    predictions = generate_predictions(read_records(args.input), engine, prompt=args.prompt, config=config,
                                       data_path=args.input, domain=args.domain)
    run = {"model": args.model, "tensor_parallel_size": args.tensor_parallel_size,
           "max_model_len": args.max_model_len, "gpu_memory_utilization": args.gpu_memory_utilization,
           "enforce_eager": args.enforce_eager, "python_version": platform.python_version(),
           "vllm_version": version("vllm"),
           "input_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
           "prompt_sha256": hashlib.sha256(load_prompt(args.prompt).encode()).hexdigest()}
    for record in predictions:
        record["generation"].update(run)
    write_records(predictions, args.output)
    if args.score_output:
        from robot_planning.evaluation.evaluator import final_eval_batch
        scores = final_eval_batch(predictions, eval_mode=args.eval_mode, solver_path=args.solver_path)
        for record, score in zip(predictions, scores):
            record["eval_result"] = score
        write_records(predictions, args.score_output)


if __name__ == "__main__":
    main()
