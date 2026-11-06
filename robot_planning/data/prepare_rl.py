"""Convert canonical tasks to subgoal TGPO parquet without private extraction."""

import argparse
import json
from robot_planning.data.io import read_records, write_parquet
from robot_planning.data.prompts import DEFAULT_PROMPT, build_messages
from robot_planning.data.schema import resolve_domain, validate_task


def convert_records(records, *, prompt=DEFAULT_PROMPT, data_source="robot_planning",
                    split="train", data_path=None, domain=None):
    rows = []
    for index, record in enumerate(records):
        scene = validate_task(record)
        domain_path = resolve_domain(record, data_path=data_path, domain=domain)
        rows.append({"data_source": data_source, "prompt": build_messages(record, prompt, data_path=data_path),
                     "ability": "decision_making", "reward_model": {"style": "rule", "ground_truth": ""},
                     "extra_info": {"split": split, "index": index, "domain_file_path": domain_path,
                                    "scene_graph": json.dumps(scene), "instruction": record["instruction"]}})
    if not rows:
        raise ValueError("No RL records")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--domain")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--data-source", default="robot_planning")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    source = read_records(args.input)
    rows = convert_records(source, prompt=args.prompt, data_source=args.data_source, split=args.split,
                           data_path=args.input, domain=args.domain)
    write_parquet(rows, args.output)
    print(json.dumps({"input": len(source), "written": len(rows), "filtered": 0, "output": args.output}))


if __name__ == "__main__":
    main()
