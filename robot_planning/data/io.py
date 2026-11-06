"""Strict JSON/JSONL input and atomic output; malformed tasks are never skipped."""

import json
import os
import tempfile
from pathlib import Path


def read_records(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Empty dataset: {path}")
    if path.suffix.lower() == ".jsonl":
        records = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
    else:
        records = json.loads(text)
        records = records if isinstance(records, list) else [records]
    if not records or any(not isinstance(row, dict) for row in records):
        raise ValueError("Dataset must contain one or more JSON objects")
    return records


def write_records(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False) as output:
        temporary = output.name
        try:
            if path.suffix.lower() == ".jsonl":
                for row in records:
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
            else:
                json.dump(records, output, ensure_ascii=False, indent=2)
            output.flush()
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def write_parquet(records, path):
    """Keep Arrow optional for users who only score existing predictions."""
    if not records:
        raise ValueError("Cannot write an empty training dataset")
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records), path)
