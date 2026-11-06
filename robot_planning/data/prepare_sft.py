"""Convert canonical supervised tasks to the original nested verl SFT schema."""

import argparse
import json
from robot_planning.data.io import read_records, write_parquet
from robot_planning.data.prompts import DEFAULT_PROMPT, build_sft_input
from robot_planning.data.schema import reference_reply, validate_task


def convert_records(records, prompt=DEFAULT_PROMPT, data_source="sft_data", *, data_path=None):
    rows = []
    for record in records:
        validate_task(record, supervised=True)
        rows.append({"data_source": data_source,
                     "prompt": {"question": build_sft_input(record, prompt, data_path=data_path)},
                     "answer": {"answer": reference_reply(record)}})
    if not rows:
        raise ValueError("No SFT records")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args()
    source = read_records(args.input)
    rows = convert_records(source, args.prompt, data_path=args.input)
    write_parquet(rows, args.output)
    print(json.dumps({"input": len(source), "written": len(rows), "filtered": 0, "output": args.output}))


if __name__ == "__main__":
    main()
