"""Train source and release SFT on one GPU, then compare checkpoint weights.

Requires the private source checkout only for the reference run. Both runs use
the same real model, Arrow input, CUDA/FSDP worker and optimizer. No model or
training function is replaced. Keep outputs outside the release source tree.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", default="2")
    args = parser.parse_args()
    release = Path(__file__).resolve().parents[2]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    model, data = Path(args.model).resolve(), Path(args.data).resolve()
    if not model.is_dir() or not data.is_file():
        parser.error("model and data must exist")
    roots = {"source": Path(args.source_root).resolve() / "RL_framework/verl_fix_cot", "release": release}
    checkpoints = {}
    for label, root in roots.items():
        destination = output / label
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, PYTHONPATH=str(root),
                   TOKENIZERS_PARALLELISM="false", TGPO_MODEL=str(model), TGPO_TRAIN_DATA=str(data),
                   TGPO_VAL_DATA=str(data), TGPO_OUTPUT=str(destination), WANDB_MODE="disabled")
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc_per_node=1",
                   "-m", "verl.trainer.fsdp_sft_trainer", "--config-path", str(release / "recipe/tgpo/configs"),
                   "--config-name", "sft", "hydra.searchpath=[pkg://verl.trainer.config]",
                   "data.train_batch_size=2", "data.micro_batch_size_per_gpu=1", "data.max_length=8192",
                   "model.fsdp_config.model_dtype=bf16", "model.strategy=fsdp",
                   "model.enable_gradient_checkpointing=false", "trainer.total_training_steps=1",
                   "trainer.n_gpus_per_node=1", "trainer.save_freq=1", "trainer.test_freq=1", "trainer.resume_mode=disable"]
        (output / f"{label}_command.json").write_text(json.dumps(command, indent=2) + "\n")
        with (output / f"{label}.log").open("w") as log:
            completed = subprocess.run(command, cwd=output, env=env, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise SystemExit(f"{label} SFT failed ({completed.returncode}); see {output / (label + '.log')}")
        files = list(destination.rglob("model_world_size_1_rank_0.pt"))
        if len(files) != 1:
            raise SystemExit(f"Expected one saved SFT checkpoint, found {len(files)} for {label}")
        checkpoints[label] = files[0]
    import torch
    before = torch.load(checkpoints["source"], map_location="cpu", weights_only=False)
    after = torch.load(checkpoints["release"], map_location="cpu", weights_only=False)
    if before.keys() != after.keys():
        raise AssertionError("Checkpoint parameter names differ")
    differences = {name: float((before[name].float() - after[name].float()).abs().max()) for name in before}
    if any(differences.values()):
        raise AssertionError(f"Source/release SFT checkpoint tensors differ: {differences}")
    result = dict(source_exit_code=0, release_exit_code=0, parameter_tensors=len(before),
                  max_abs_difference=max(differences.values(), default=0), exact_weight_parity=True,
                  device="CUDA", steps=1, samples=2)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
