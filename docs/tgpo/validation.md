# Executed release validation

Validation date: 2026-10-24. These checks establish executable interfaces and
source correspondence on synthetic tasks. They do not reproduce private-data
research results. Original development sources were read as references and
were not modified. Checkpoints, private source, and large runtime logs are not
part of the release.

## Scope audit

The release follows the inclusion and exclusion decisions in
[migration.md](migration.md). No custom checkpoint/debug package, backup trainer,
duplicate custom config tree, unrelated e2e/final-goal/two-stage/ranking evaluator,
private data extraction, or compiled planner is included. Useful approved data
creation and `fix_plan_pddl` functionality is split into task data, scene state
and planning modules. The shared evaluator implements only the specified
`final_eval_subgoal` protocol.

The selected immutable verl base contains 322 framework files; 316 remain
unchanged. Original snapshot hashes and public hashes are checked separately by
`scripts/validation/verify_release.py`. The selected base is reproducible;
it is not asserted to recover the snapshot's missing checkout history.

## Results

| Check | Executed result | Boundary and limits |
| --- | --- | --- |
| Complete source/integration suite | **78 passed, 3 skipped**, 29.99 s | Original checkout, real Qwen tokenizer, real Fast Downward and optional local PPDDL backend supplied; live-model tests run separately below |
| Final wheel in a clean training environment | **77 passed, 4 skipped**, 29.28 s | Tests run outside the checkout against installed packages; optional PPDDL backend absent, plus 3 live-model tests skipped |
| Final wheel native CUDA training | **Passed**, 27 saved parameter tensors | Clean environment, torch 2.7.0+cu126; external harness imports installed packages only; canonical data, production reward and actual continuation/backward/save, with fixture repair |
| Standalone CPU CI dependencies | **25 passed, 43 skipped**, 2.67 s | Installed only `[data,test]`, no torch/Ray/vLLM or private checkout; reference/model-dependent cases explicitly skipped |
| SFT source/public CUDA comparison | **27/27 checkpoint tensors exactly equal**, maximum difference 0 | Two separate real CUDA/FSDP training processes; one step, two synthetic records; same small Qwen2 model and Arrow input |
| Native TGPO source/public comparison | **12/12 dispatched tensors and 27/27 checkpoint tensors exactly equal** | Original `fit` and original selection/batch function bodies; real Ray/FSDP/vLLM, old log probabilities, GRPO advantages, backward pass, AdamW and save; deterministic external repair/scoring fixtures |
| Native GRPO control | Real generation, update and checkpoint succeeded | Shared unrefined training path; optimizer first moments and parameter changes verified |
| Native TGPO resume | Step and optimizer counters advanced **1 → 2** | Restored optimizer state; maximum parameter change after resume 0.000101149 |
| Canonical data/native TGPO | Real converter → RLHFDataset → TGPO → actual task reward → continuation → update/checkpoint succeeded | Context stored solely in `extra_info`, hydrated correctly at the trainer boundary; selected 2, repaired 2, improved 0; repair fixture, actual planner scoring |
| Production reward boundary | Actual configured manager produced **[1, 0]** | Real Qwen tokenizer, canonical parquet, dynamic reward loading, Fast Downward, EOS reward placement and repair HTTP-failure fallback |
| Full pipeline | SFT export → TGPO → export → vLLM generation → evaluation succeeded | Uses verl's existing model merger; random small model's invalid output correctly scores 0 |
| Real inference source/public | Prompt and generated output exactly equal | Original VLLMInference wrapper versus public inference; 4,562-token prompt, tiny random model, greedy 16-token generation, seed 7 |
| Live reviewer source/public | Complete messages and result dictionaries exactly equal in both `validate_first` and `always_eval` | Qwen2.5-7B-Instruct served through real HTTP, real two-action Fast Downward plan; temperature 0, seed 7, maximum 2,048 tokens; 2 separately executed tests passed |
| Live repair source/public | Request messages, raw response and parsed repaired CoT exactly equal | Original repair function body versus public repair; real Qwen2.5-7B-Instruct HTTP endpoint, temperature 0, seed 7, maximum 1,024 tokens; missing-tag fallback also checked |
| Packaging and installation | Wheel/sdist built; independent CPU-only and training environments exercised | Evaluation requires only PyYAML; training environment does not inherit system site-packages; see installation checks below |

The native model has 377,856 parameters and runs on one RTX 4090. FSDP has
world size 1 and uses NO_SHARD. Fixture-based TGPO tests deliberately produce
scores `[0, 1, 0, 1]` to exercise successful candidate replacement and nonzero
policy gradients. The canonical-data test uses the production evaluator;
random generations all score zero. Its small weight change comes from AdamW
weight decay, and is not evidence of a useful policy gradient. These tests
cover different boundaries and are not interchangeable. A compact native
record is in [native_training_validation.json](native_training_validation.json).

Temperature 0 alone did **not** make the first live reviewer/repair responses
byte-identical: the unseeded reviewer kept the same score, grade and plan but
changed its natural-language feedback. Those differences were retained in the
local evidence. The subsequent explicitly seeded comparisons passed. This
does not guarantee deterministic hosted models across servers or versions.

## Test coverage and reference handling

Training tests cover 120 randomized source/public batches (240 selection and
candidate-index comparisons), duplicate/reordered instruction groups, threshold
and tie behavior, prefix masks, tensor replacement, metadata alignment, worker
padding, failed repair fallback, asynchronous rewards and canonical task context.
The CPU full-loop comparison executes the original `fit` AST with controlled
external boundaries and real Torch/AdamW updates, including two consecutive
steps and no-selection/no-improvement cases. The native comparison then checks
actual worker dispatch tensors and saved model weights.

Evaluation tests cover all three modes, nested batches, parser tolerance,
invalid/unknown predicates, partial and empty plans, sequential scene effects,
seeded problem bytes, original SummaryEvaluator request/response protocol,
whole-evaluator repetition/minimum selection, and feedback selection on ties.
Real planner tests cover concurrency, arbitrary working directories, subprocess
timeouts, process-group cleanup, solver statuses and infrastructure failures.
Data tests compare source function bodies, prompt bytes and tokenizer tensors,
and exercise parquet conversion, dynamic domains, task trees and effect helpers.
The shared HTTP client is tested through an actual local server.

Private-source comparisons use `TGPO_SOURCE_ROOT` to read original functions;
minimal import scaffolding isolates them from excluded packages. They do not
replace reference algorithms with public implementations. The SFT comparison
runs the entire original trainer in a separate process. The native TGPO
comparison runs the original loop and batch functions with the same public
native worker backend; it is not a test of every original environment dependency.
Intentional fixes have separate regressions and are documented in
[upstream_training_notes.md](upstream_training_notes.md).

## Installation checks

The first native environment used Python 3.11.14, torch 2.7.0+cu128,
transformers 4.51.3, vLLM 0.9.1, FlashAttention 2.7.4.post1, Ray 2.52.1,
tensordict 0.8.3, torchdata 0.11.0, datasets 3.6.0, pandas 2.2.3,
NumPy 1.26.0, pyarrow 19.0.1 and uvloop 0.23.0.

A second Python 3.11 virtual environment was created **without** system
site-packages. Normal installation of `.[train,inference,test]` resolved all
dependencies; a matching upstream FlashAttention wheel was installed separately.
`pip check` reported no broken requirements. Sixteen key modules imported from
that environment's site-packages. SFT/PPO/TGPO installed configuration entry
points all exited successfully from an unrelated working directory. This clean
environment uses torch 2.7.0+cu126, NumPy 1.26.4 and pandas 2.3.3; both tested
CUDA builds share the declared torch 2.7.0 version.

The final wheel was reinstalled into both independent environments. The clean
training environment then ran the complete test suite from outside the source
checkout: 77 passed, 4 explicitly skipped (one optional PPDDL dependency and
three separate live-model cases). No production module was imported from the
workspace. The source-backed environment's 78-pass run additionally executed
the optional PPDDL comparison.

The clean environment also completed the native canonical-data TGPO run from
an external harness containing no framework/task source. Installed origin paths
were checked before launching Ray. This verifies actual CUDA execution with the
normally resolved torch 2.7.0+cu126 build, rather than import success alone.

A separate evaluation-only environment installed the wheel with PyYAML and no
torch, Ray or vLLM. From outside the checkout it loaded installed prompt resources,
ran the real planner and scored the supplied example 1. Fast Downward
release-24.06.1 was downloaded, SHA-256 checked and built from source. Optional
PPDDL parity was exercised with a separately supplied compatible local backend;
that library is not distributed or silently equated to a same-named PyPI package.

## Re-run the checks

From the release checkout, with the relevant dependencies installed:

```bash
export TGPO_SOURCE_ROOT=/path/to/original/decision_making
export TGPO_TOKENIZER=/path/to/Qwen2.5-7B-Instruct
export FAST_DOWNWARD_PATH=/path/to/fast-downward/fast-downward.py
export TGPO_SOLVER="$FAST_DOWNWARD_PATH"
python -m pytest tests -q --junitxml=/tmp/tgpo-tests.xml
python scripts/validation/verify_release.py --source-root "$TGPO_SOURCE_ROOT"
python -m ruff check robot_planning verl/trainer/tgpo tests scripts/validation
python -m build
```

Live tests are opt-in rather than silently making network/model requests:

```bash
TGPO_LIVE_LLM_BASE_URL=http://localhost:8000/v1 \
TGPO_LIVE_LLM_MODEL=your-served-model \
  python -m pytest tests/robot_planning/test_evaluation_parity.py -k live -q
CUDA_VISIBLE_DEVICES=0 TGPO_INFERENCE_MODEL=/path/to/tiny-model \
  python -m pytest tests/robot_planning/test_inference_integration.py -q
```

For native training, create the small model with
`scripts/validation/create_tiny_model.py`. Prepare a two-record SFT parquet and
use `run_sft_parity.py --source-root ... --model ... --data ... --output ...`.
Export the checkpoint with `python -m verl.model_merger` before RL training.

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.validation.native_training \
  --model /path/to/sft-exported-model --output /tmp/tgpo-native \
  --mode tgpo --canonical-tasks recipe/tgpo/examples/tasks.jsonl --real-task-reward
```

For controlled numerical parity, omit the last two options, run both `--mode
source` and `--mode tgpo` into different directories, then use
`scripts/validation/compare_native_training.py`. The harness explicitly names
its fixtures; it is a validation tool, not an alternate production trainer.

## Limits

No multi-node, multi-GPU sharding/throughput, long-run numerical stability,
large-model training, async native rollout, or private benchmark performance is
claimed. Public recipe hyperparameters are candidate historical settings, not
an author-confirmed main-result preset. Only the synchronous single-GPU native
path was exercised. Successful smoke tests do not certify every optional
backend retained in the pinned upstream framework.
