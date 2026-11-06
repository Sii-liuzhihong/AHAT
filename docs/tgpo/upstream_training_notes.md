# Framework baseline and training migration

The public framework is based on upstream verl commit
`a7b8675f9674999e2c6b75a402d95837eca40a3d` (2025-09-15), version
`0.5.0.dev` (public fork version `0.5.0.dev0+tgpo.1`). This is an explicitly selected, immutable upstream baseline.
It is **not** a claim that the original development directory was checked out
from that exact commit: that directory has no independent Git metadata.

The neighboring repository's HEAD (`18f1a7af...`, version `0.8.0.dev`) is not
used as the baseline. Git object/history comparisons found that the original
`ray_trainer.py` matches a blob introduced on 2025-09-03 and replaced on
2025-09-16, while `protocol.py` matches a blob introduced on 2025-09-09 and
replaced on 2025-09-16. Comparing all available commits in the surrounding
2025-09-01 through 2025-09-18 history found the selected commit had the
highest exact-file overlap: 315 of 325 original framework files.

All 322 files under upstream `verl/` at the selected commit form the initial
public tree. No local debug trainer or later untracked rollout module was
copied. `framework_manifest.json` records upstream Git blobs and SHA-256 hashes
of the original and released files, including exact-match flags. New TGPO
modules are intentionally outside the upstream file list.

## Original framework differences

| Original file | Difference from selected upstream | Public treatment |
| --- | --- | --- |
| `trainer/main_ppo_debug.py` | Local debug entry | Excluded |
| `workers/rollout/replica.py`, `workers/rollout/utils.py` | Local modules absent from baseline, unused by selected training path | Excluded |
| `trainer/main_ppo.py`, `workers/reward_manager/naive.py`, `trainer/config/reward_model/dp_reward_model.yaml` | Whitespace/comment differences | Use upstream |
| `trainer/config/ppo_trainer.yaml` | Machine checkpoint root and token budget overrides | Use upstream, move experiment overrides into recipe |
| `workers/config/rollout.py`, `workers/rollout/vllm_rollout/vllm_rollout_spmd.py` | Hardcoded larger default token budget | Use upstream, configure budget explicitly |
| `utils/dataset/sft_dataset.py` | Adds a system message, `You are a helpful assistant.` | Optional `data.system_prompt`; SFT recipe explicitly enables the original message |

SFT therefore reuses the upstream trainer, but the development dataset was
not completely unmodified. The public optional message makes this previously
implicit difference reproducible without changing the ordinary upstream default.

## Shared training loop

`main_ppo.TaskRunner` retains upstream worker construction, dataset/sampler
creation, checkpointing, and execution. Its only extension is a configured
`trainer_class`, checked to subclass `RayPPOTrainer`. `main_tgpo` is a thin
Hydra entry selecting `tgpo_trainer.yaml`.

`RayPPOTrainer.fit` is the sole training loop. The `postprocess_rollouts` hook
runs after initial reward resolution and before recomputing old log probability.
The default implementation returns the input unchanged. `TGPOTrainer` only
adapts configured task callables into `refine_rollouts`; the algorithm package
never imports robot task prompts, PDDL, or LLM clients.

The original custom trainer, backup trainer, duplicate configs, per-step memory
profiling, sample-debug logging, and checkpoint-manager utilities are excluded.
The original GRPO advantage and PPO update implementation are reused unchanged.

## Preserved experiment semantics

- Group rows by **instruction string**, preserving encountered order, then chunk
  each instruction's rows by `rollout.n`. This deliberately preserves the source
  policy even though `uid` grouping would avoid ambiguity for duplicate tasks.
- Trigger only when every group reward is **strictly less** than threshold.
- Select the highest scores with `torch.topk(..., sorted=False)` and preserve its
  tie behavior; sort/deduplicate selected original row indices afterward.
- Repeat each selected prefix with `interleave=True`; use first `argmax` for
  equally scoring continuation candidates.
- Replace only on **strict improvement**. Train on the repaired prefix plus
  continuation, cropped to original response capacity.
- Generation prompt truncation retains the original prompt capacity and removes
  tokens from the left, as in the source.
- Retain the historical text asymmetry: subgoal continuation prompts use the
  indented `### Task Decomposition ###` heading; reconstructed responses use the
  `<think>`/`Task Decomposition (Tree-Structured CoT)` wrapper. This is documented
  rather than silently changed during cleanup.
- Recompute old log probability only after final row replacement. Identity,
  context, and GRPO group UID remain associated with their original task.

## Explicit fixes, separate from structural equivalence

The public default corrects the following defects. Differential tests do not
claim these fixes are identical to defective source behavior.

| Fix | Source behavior | Public behavior / evidence |
| --- | --- | --- |
| Position IDs | Continuation input retains full prompt+old-response position width; reconstructed response retains stale position IDs | Recompute from final masks; `tgpo.position_policy=legacy` exists only to compare original tensors |
| Async reward order | Original TGPO reads reward diagnostics before resolving the future (and has a local `ray` import) | Resolve initial reward before refinement; never overwrite refined rewards with the original future later |
| Selected metadata | Initial fixed-width string arrays may truncate text; candidate reward feedback not attached; original diagnostics can overwrite feedback | Store row metadata in object arrays, attach newly evaluated candidate diagnostics, replace all reward fields together |
| Global token counts | Counts are calculated before repaired responses are inserted | Recompute final attention-mask token counts after refinement |
| Worker divisibility | Selected continuation count may not divide rollout worker count | Pad before generation and remove padded candidates before selection/scoring |
| Empty repairs | Can create a heading-only replacement and unnecessarily regenerate | Keep the original row when no usable repair prefix exists |
| Canonical task metadata | Data converter stores context in `extra_info`, while custom TGPO expects top-level instruction/scene fields | Promote only configured `task.metadata_fields` at the trainer boundary; parquet remains normalized |
| Reward loader cache | The global `custom_module` alias can reuse a different reward file in the same interpreter | Reload when the requested resolved path differs; regression test loads two functions with identical names from different files |
| uvloop compatibility | Recent uvloop does not create an event loop implicitly in a fresh Ray worker | Create and register a loop when synchronous FSDP rollout initialization has none; discovered by native worker testing |
| SFT pandas compatibility | Series positional `ls[0]` fails on pandas 3, generic `squeeze()` collapses one-row datasets | Use `.iloc[0]` for Series and `squeeze(axis='columns')`; ndarray behavior retained |

The default recipe uses corrected positions. The `legacy` switch does not
reintroduce obsolete diagnostics, async faults, or machine-specific paths, and
is not a promise that the entire old trainer could execute with every native
backend. The parity tests isolate and document those distinctions.

## Executed comparison tests

Run from the installed release checkout:

```bash
python -m pytest tests/tgpo -q
TGPO_SOURCE_ROOT=/path/to/private/decision_making \
  CUDA_VISIBLE_DEVICES='' python -m pytest tests/tgpo -q
python -m verl.trainer.main_tgpo --cfg job
```

The source path is read only. Public tests do not bundle copies of the excluded
custom trainer. Tests dynamically execute the original selection and batch
functions from that directory; the full-loop test extracts the original `fit`
AST and executes it unchanged with excluded logging disabled and deterministic
repair/rollout/reward boundaries. The model and optimizer are real Torch code,
using upstream GRPO advantages and the upstream PPO clipped policy objective.

The completed checks include 120 randomized source/public batches (240 selection/candidate-index comparisons)
covering reordered rows, duplicate instruction groups, ties, and varying group
and selection sizes; exact original/public prefix-generation and response-merge
tensors in legacy position mode; empty/failed repair fallbacks; selected-batch
padding; metadata replacement and position-mask invariants; and the complete
original/public full loops with successful refinement, no improvement, no selection, and two consecutive steps. Both loops produce exactly equal final token
batches, rewards, GRPO advantages, old log probabilities, policy loss, and model
weights after real AdamW updates. An additional async-order regression test verifies that the initial future is resolved exactly once before refinement and never overwrites selected rewards. The weights demonstrably change from their
initial values. Hosted LLM quality and native GPU worker initialization are
outside that deterministic loop comparison; consult the full validation report
for native end-to-end smoke runs and their limitations.

## Native GPU execution and canonical data wiring

`native_training_validation.json` records actual native executions on one GPU,
using a 377,856 parameter random Qwen2 model after its verified SFT update.
The runtime uses actual Ray workers, FSDP, FlashAttention, vLLM sampling,
probability recomputation, upstream GRPO/PPO loss, AdamW, checkpoint save and
restore. At world size one, PyTorch selects `NO_SHARD`; these tests do not
establish multi-GPU sharding or performance.

The original private `fit` and original selection/batch functions executed with
that same native backend. All **12 actor-update input tensors and 27 saved model
tensors exactly match** the public default TGPO run (`torch.equal`, zero
tolerance). The native public and original runs both use explicit deterministic
external repair/scoring fixtures, which force actual continuation generation
and strict improvement from `[0, .25, 0, .25]` to `[0, 1, 0, 1]`. Nonzero AdamW
first moments and changed parameters establish real policy-gradient updates.

A separate native run starts with the actual `prepare_rl.convert_records`
output. Its parquet stores `instruction` and `scene_graph` only in `extra_info`.
The actual `RLHFDataset`, common loop and configured metadata boundary feed
`robot_planning.reward` in `only_validate` mode. The random model's invalid
subgoals correctly receive zero scores. Two rows are selected and repaired,
actor continuations run, and zero improvement retains the originals. This run
checks production data/reward wiring; its zero policy gradient is expected and
its small parameter change is AdamW weight decay, not task learning.

The resumed native TGPO run loads checkpoint 1, restores optimizer and dataloader
state, and saves checkpoint 2. Optimizer step counters advance from 1 to 2 and
parameters change by a further `0.00010114908218383789`. VLLM random-state
reproducibility across restarts is not established by this restore check.

Reproduce from the repository root with a local small compatible model directory
and an available GPU selected in `CUDA_VISIBLE_DEVICES`:

```bash
export VLLM_USE_V1=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m scripts.validation.native_training --model "$SMOKE_MODEL" --output /tmp/tgpo-grpo --mode grpo
python -m scripts.validation.native_training --model "$SMOKE_MODEL" --output /tmp/tgpo-public --mode tgpo
TGPO_SOURCE_ROOT="$PRIVATE_SOURCE" python -m scripts.validation.native_training \
  --model "$SMOKE_MODEL" --output /tmp/tgpo-original --mode source
python -m scripts.validation.compare_native_training /tmp/tgpo-original /tmp/tgpo-public \
  --output /tmp/tgpo-parity.json --require-equal
python -m scripts.validation.native_training --model "$SMOKE_MODEL" --output /tmp/tgpo-public \
  --mode tgpo --steps 2 --resume
python -m scripts.validation.native_training --model "$SMOKE_MODEL" --output /tmp/tgpo-canonical \
  --mode tgpo --canonical-tasks recipe/tgpo/examples/tasks.jsonl --real-task-reward
```

The optional `source-corrected` harness mode adapts original position IDs in
memory for diagnosing configurations where they differ. It was unnecessary for
the demonstrated native equality and is not used to claim raw-source parity.
Original source files are never edited by the harness. All output checkpoints,
Ray sessions and logs belong in external validation directories, not the release.
