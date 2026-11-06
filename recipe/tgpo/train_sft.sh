#!/usr/bin/env bash
set -euo pipefail
: "${TGPO_MODEL:?Set TGPO_MODEL to a Hugging Face model directory or ID}"
: "${TGPO_TRAIN_DATA:?Set TGPO_TRAIN_DATA to train.parquet}"
: "${TGPO_VAL_DATA:?Set TGPO_VAL_DATA to validation.parquet}"
recipe_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="${N_GPUS:-8}" \
  -m verl.trainer.fsdp_sft_trainer \
  --config-path "$recipe_dir/configs" --config-name sft \
  "hydra.searchpath=[pkg://verl.trainer.config]" "trainer.n_gpus_per_node=${N_GPUS:-8}" "$@"
