"""Evaluate JSON/JSONL predictions and save results with their run provenance."""

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import random

from robot_planning.evaluation.evaluator import evaluate_repeated
from robot_planning.evaluation.metrics import aggregate_metrics
from robot_planning.evaluation.reviewer import resolve_prompt


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _package_version():
    try:
        return version("verl")
    except PackageNotFoundError:
        return "source-tree"


def _select_predictions(record, policy):
    if policy == "all":
        return record
    record = dict(record)
    for plural, singular in (
        ("reply_list", "reply"),
        ("subgoal_related_obj_list_pair_list", "subgoal_related_obj_list_pair"),
    ):
        if record.get(plural) is not None:
            values = record.pop(plural)
            record[singular] = values[0] if values else None
            break
    return record


def _result_records(records, results):
    for source_index, (record, result) in enumerate(zip(records, results)):
        values = result if isinstance(result, list) else [result]
        predictions = record.get(
            "reply_list", record.get("subgoal_related_obj_list_pair_list")
        )
        if predictions is None:
            predictions = [
                record.get("reply", record.get("subgoal_related_obj_list_pair"))
            ]
        for candidate_index, value in enumerate(values):
            yield dict(
                id=record.get("id", record.get("task_id", source_index)),
                source_index=source_index,
                candidate_index=candidate_index,
                prediction=(
                    predictions[candidate_index]
                    if candidate_index < len(predictions)
                    else None
                ),
                metadata=record.get("metadata", {}),
                evaluation=value,
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="JSON array or JSONL task records with predictions",
    )
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--output", help="Combined JSON result file")
    output.add_argument(
        "--output-dir",
        help="Directory for results.jsonl, metrics.json and run_config.json",
    )
    parser.add_argument("--domain", help="Global PDDL domain override")
    parser.add_argument(
        "--solver", help="fast-downward.py or a plan/domain/problem shell adapter"
    )
    parser.add_argument(
        "--mode",
        choices=("validate_first", "always_eval", "only_validate"),
        default="validate_first",
    )
    parser.add_argument(
        "--model", default=os.environ.get("TGPO_LLM_MODEL", "deepseek-v3")
    )
    parser.add_argument("--prompt", default="eval_plan_summary_1006_2")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--candidates",
        choices=("first", "all"),
        default="all",
        help="Explicit prediction policy; source final eval supports all candidates",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional explicit row limit; default evaluates every row",
    )
    args = parser.parse_args(argv)
    path = Path(args.input).resolve()
    content = path.read_text(encoding="utf-8")
    records = (
        [json.loads(line) for line in content.splitlines() if line.strip()]
        if path.suffix == ".jsonl"
        else json.loads(content)
    )
    if not isinstance(records, list):
        parser.error("input must contain a JSON array or JSONL records")
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        records = records[: args.limit]
    records = [_select_predictions(row, args.candidates) for row in records]
    domains = {}
    if args.domain:
        args.domain = str(Path(args.domain).resolve())
    for record in records:
        domain = args.domain or record.get("domain_file_path")
        if not domain:
            parser.error("provide --domain or domain_file_path on every input row")
        resolved = Path(domain) if Path(domain).is_absolute() else path.parent / domain
        resolved = resolved.resolve()
        if not resolved.is_file():
            parser.error(f"domain file does not exist: {resolved}")
        record["domain_file_path"] = str(resolved)
        domains[str(resolved)] = _sha256(resolved)
    random.seed(args.seed)
    results = evaluate_repeated(
        records,
        repeats=args.repeats,
        domain_file_path=args.domain,
        eval_mode=args.mode,
        system_prompt=args.prompt,
        exp_model=args.model,
        max_workers=args.workers,
        solver_path=args.solver,
        diagnostics=True,
    )
    config = dict(
        vars(args),
        input_sha256=_sha256(path),
        domain_sha256=domains,
        prompt_sha256=hashlib.sha256(
            resolve_prompt(args.prompt, batch=True).encode()
        ).hexdigest(),
        python_version=platform.python_version(),
        verl_version=_package_version(),
        seed_scope="parent process; worker RNG follows the original multiprocessing protocol",
        llm_request_seed=(int(os.environ["TGPO_LLM_SEED"]) if os.environ.get("TGPO_LLM_SEED") else None),
        generation=[
            row.get("generation", row.get("generation_config"))
            for row in records
            if "generation" in row or "generation_config" in row
        ],
    )
    metrics = aggregate_metrics(results)
    rows = list(_result_records(records, results))
    if args.output_dir:
        directory = Path(args.output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "results.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        )
        for name, value in (("metrics", metrics), ("run_config", config)):
            (directory / f"{name}.json").write_text(
                json.dumps(value, ensure_ascii=False, indent=2) + "\n"
            )
    else:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(config=config, metrics=metrics, results=results, records=rows)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
