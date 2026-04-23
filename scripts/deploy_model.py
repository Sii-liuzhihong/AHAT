"""Deploy the AHAT model with vLLM (single GPU).

This script launches the vLLM OpenAI-compatible server on port 8000
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

from ahat.configs.constants import (
		AHAT_MODEL_DEFAULT_PATH, 
		DEPLOYMENT_DEFAULT_PORT, 
		DEPLOYMENT_DEFAULT_HOST
	)





def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy AHAT model using vLLM.")
    parser.add_argument(
        "--model",
        default=AHAT_MODEL_DEFAULT_PATH,
        help=f"Local model path (default: {AHAT_MODEL_DEFAULT_PATH})",
    )
    parser.add_argument(
        "--host",
        default=DEPLOYMENT_DEFAULT_HOST,
        help=f"Host address to bind (default: {DEPLOYMENT_DEFAULT_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEPLOYMENT_DEFAULT_PORT,
        help=f"Port to serve (default: {DEPLOYMENT_DEFAULT_PORT})",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size (single GPU default: 1)",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    project_root = Path(__file__).resolve().parents[1]
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = project_root / model_path

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(model_path),
        "--served-model-name",
        AHAT_MODEL_DEFAULT_PATH,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
    ]

    print("Executing:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
