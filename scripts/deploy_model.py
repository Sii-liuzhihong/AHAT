# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Sii-liuzhihong
"""Serve an exported AHAT/TGPO model with the pinned vLLM backend."""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/AHAT-TGPO")
    parser.add_argument("--served-model-name", default="AHAT-TGPO")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    args = parser.parse_args()
    model = Path(args.model).expanduser()
    if not model.is_absolute():
        model = Path(__file__).resolve().parents[1] / model
    if not model.is_dir():
        parser.error(f"local model directory does not exist: {model}")
    subprocess.run([
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(model), "--served-model-name", args.served_model_name,
        "--host", args.host, "--port", str(args.port),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
    ], check=True)


if __name__ == "__main__":
    main()
