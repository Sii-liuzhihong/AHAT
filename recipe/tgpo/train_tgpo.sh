#!/usr/bin/env bash
set -euo pipefail
: "${TGPO_MODEL:?Set TGPO_MODEL to an exported SFT Hugging Face model}"
: "${TGPO_TRAIN_DATA:?Set TGPO_TRAIN_DATA to train.parquet}"
: "${TGPO_VAL_DATA:?Set TGPO_VAL_DATA to validation.parquet}"
: "${TGPO_LLM_MODEL:?Set TGPO_LLM_MODEL for the repair/review endpoint}"
recipe_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export TGPO_REWARD_PATH="$(python -c 'import robot_planning.reward; print(robot_planning.reward.__file__)')"
exec python -m verl.trainer.main_tgpo \
  --config-path "$recipe_dir/configs" --config-name tgpo \
  "hydra.searchpath=[pkg://verl.trainer.config]" "trainer.n_gpus_per_node=${N_GPUS:-8}" "$@"
