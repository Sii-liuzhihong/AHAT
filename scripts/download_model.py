"""Download the AHAT-TGPO model from Hugging Face.

This script downloads the repository snapshot into the project's
`models/` directory. The target directory is created automatically if it
does not already exist.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download
from ahat.configs.constants import AHAT_MODEL_DEFAULT_PATH, AHAT_MODEL_REPO_ID

def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Download the AHAT-TGPO model from Hugging Face.")
	parser.add_argument(
		"--repo-id",
		default=AHAT_MODEL_REPO_ID,
		help=f"Hugging Face repository ID to download (default: {AHAT_MODEL_REPO_ID})",
	)
	parser.add_argument(
		"--output-dir",
		default=AHAT_MODEL_DEFAULT_PATH,
		help=f"Target directory for the downloaded model. Defaults to {AHAT_MODEL_DEFAULT_PATH}.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	project_root = Path(__file__).resolve().parents[1]
	local_dir = Path(args.output_dir)
	if not local_dir.is_absolute():
		local_dir = project_root / local_dir

	local_dir.mkdir(parents=True, exist_ok=True)

	print(f"Downloading {args.repo_id} to {local_dir}...")
	snapshot_download(
		repo_id=args.repo_id,
		local_dir=str(local_dir),
		local_dir_use_symlinks=False,
	)
	print("Download completed.")


if __name__ == "__main__":
	main()
