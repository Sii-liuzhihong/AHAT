# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Sii-liuzhihong
"""Download the AHAT dataset from Hugging Face."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="SII-liyang2024/AHAT-dataset")
    parser.add_argument("--output-dir", default="data/AHAT-dataset")
    parser.add_argument("--revision", help="Optional pinned Hugging Face revision")
    args = parser.parse_args()
    # Keep --help usable without installing optional download dependencies.
    from huggingface_hub import snapshot_download

    destination = Path(args.output_dir).expanduser()
    if not destination.is_absolute():
        destination = Path(__file__).resolve().parents[1] / destination
    snapshot_download(repo_id=args.repo_id, repo_type="dataset",
                      local_dir=str(destination), revision=args.revision)
    print(f"Downloaded {args.repo_id} to {destination}")


if __name__ == "__main__":
    main()
