"""Download AHAT dataset from Hugging Face.

By default, this script downloads the dataset snapshot from
`AHAT_DATASET_REPO_ID` and stores it under
`AHAT_DATASET_DEFAULT_PATH` in the current project.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from ahat.configs.constants import AHAT_DATASET_DEFAULT_PATH, AHAT_DATASET_REPO_ID


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Download AHAT dataset from Hugging Face.")
	parser.add_argument(
		"--repo-id",
		default=AHAT_DATASET_REPO_ID,
		help=f"Hugging Face dataset repository ID (default: {AHAT_DATASET_REPO_ID})",
	)
	parser.add_argument(
		"--output-dir",
		default=AHAT_DATASET_DEFAULT_PATH,
		help=f"Local directory to save dataset (default: {AHAT_DATASET_DEFAULT_PATH})",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	project_root = Path(__file__).resolve().parents[1]
	local_dir = Path(args.output_dir)
	if not local_dir.is_absolute():
		local_dir = project_root / local_dir

	local_dir.mkdir(parents=True, exist_ok=True)

	print(f"Downloading dataset {args.repo_id} to {local_dir}...")
	snapshot_download(
		repo_id=args.repo_id,
		repo_type="dataset",
		local_dir=str(local_dir),
		local_dir_use_symlinks=False,
	)
	print("Dataset download completed.")


if __name__ == "__main__":
	main()