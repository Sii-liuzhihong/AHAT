# Subgoal evaluation

Training reward and offline scoring call the same `robot_planning.evaluation`
implementation. This package has no Ray, torch, vLLM or training imports.
Only the final subgoal protocol is included; e2e, final-goal, rank and two-stage
reviewers are excluded.

## Protocol

1. Parse `Subtask goal: [...]` and `Implicitly related object: [...]` lines.
   Related objects may be quoted, unquoted or mixed. Parsing follows the
   development implementation, including its accepted structured-pair format.
2. Check predicate name, argument count and object existence. The reference
   does not enforce argument types, so the migrated checker does not add a
   type gate.
3. Solve each subtask's cumulative goal prefixes with Fast Downward
   `lama-first`. After **each** plan, update the scene before generating the
   next problem. These dependent goals must not be parallelized.
4. Scene context construction randomly samples up to three additional pieces
   of furniture from the relevant room. This is part of the existing protocol,
   rather than an LLM-only source of randomness.
5. A valid prediction satisfies `parsable and subgoal_correct and
   solvable_score > 0`. A Summary review maps Good/Normal/Bad to **1/0.5/0**.
   `final_result = int(validate_result) * llm_score`.

| Mode | LLM review | Final score |
|---|---|---|
| `validate_first` | Valid prediction with a nonempty solved plan | Validity × review |
| `always_eval` | Any nonempty solved plan, including partial plans | Validity × review |
| `only_validate` | None | Integer validity |

A goal that already holds can be valid with an empty action plan. Like the
reference, the first two modes cannot review that plan and score zero;
`only_validate` scores one. A global domain path overrides per-row paths, as
in the actual source implementation. These conventions are preserved rather
than silently changed during restructuring.

`evaluate_repeated(..., repeats=2)` reruns **all validation and review**, then
selects the smaller final score with its complete result. Equal scores select
the later run. It does not repeat only the reviewer or combine reviews from
one run with a plan from another.

## Usage

Build Fast Downward with `scripts/setup_fast_downward.sh` and set
`FAST_DOWNWARD_PATH` to its `fast-downward.py`, or pass `--solver`. The default
outer timeout remains five seconds. A legacy shell adapter accepting
`plan_path domain_path problem_path` also works. Planner calls own isolated
temporary directories and process groups; a timeout terminates descendants.

```bash
python -m robot_planning.evaluation.cli \
  --input recipe/tgpo/examples/predictions.jsonl \
  --output-dir runs/example-eval \
  --mode only_validate --workers 1 --candidates all
```

For review, set `OPENAI_BASE_URL`, `OPENAI_API_KEY` when needed, and
`TGPO_LLM_MODEL`, or pass `--model`. `TGPO_LLM_SEED` optionally pins the
review request seed; the evaluation CLI row-sampling seed is separate. The API functions preserve the source's
`deepseek-v3` default; pass `exp_model` explicitly to use another model.
The endpoint must implement OpenAI chat completions. No endpoint credentials
are embedded in prompts, examples or manifests.

`--candidates all` preserves all nested predictions; `first` selects the first
prediction explicitly. There is no implicit 500-row limit; `--limit` opts into
a limit. Per-row relative domain paths resolve against the input file's
parent, independently of the shell working directory.

The output directory contains:

- `results.jsonl`: task ID, source index, candidate index, prediction, metadata,
  complete evaluation and per-planner-call statuses.
- `metrics.json`: task/candidate counts, all-candidate means, reviewed-candidate
  mean, denominators and infrastructure/API failure counts.
- `run_config.json`: evaluation parameters, input/domain/prompt SHA256 hashes,
  Python/verl versions and generation metadata supplied by input records.

`--output file.json` instead writes a combined JSON result for simple callers.
Only explicit generation metadata is copied into run provenance; credentials
are not CLI parameters. `--seed` seeds the parent process. The existing
multiprocessing validation protocol does not guarantee identical worker random
states; use direct single-item calls with a seeded RNG for exact sampled
problem comparisons. No deterministic LLM-quality guarantee is implied.

## API and diagnostics

```python
from robot_planning.evaluation import final_eval_single, final_eval_batch

result = final_eval_single(
    instruction, scene_graph, reply=reply, domain_file_path=domain_path,
    eval_mode="only_validate", solver_path=planner_path,
)
```

`llm_client=` accepts an injectable client returning `LLMResponse`. This
boundary is shared with repair. Default result fields and nesting preserve
legacy behavior. `diagnostics=True` additionally includes `solver_results`:
`success`, `no_solution`, `timeout`, `planner_error`, or `input_error`. Missing planner/domain files raise actionable configuration
errors rather than being counted as unsolvable tasks. Reviewer API failures
retain `llm_error` and zero score.

`robot_planning.pddl.solver.solve_pddl` returns `(plan, success)`;
`solve_pddl_detailed` also returns status and return code. Goal/action problem
construction shares one context builder. `generate_problem_from_actions`
preserves its actual legacy **empty-goal** output; the older comment claiming
it derives the last action's effects was inaccurate.

## Deliberate corrections from the development tree

- Missing scene rows retain their own index; they cannot shift another row's
  validation into the wrong task.
- A solver that exits without a plan consistently returns `([], False)`,
  instead of the reference's accidental bare `[]` in that branch.
- Unknown evaluation modes are rejected.
- Missing domain/planner configuration raises an error, rather than silently
  turning a broken run into zero task rewards.

These corrections have explicit regression tests. Canonical valid inputs are
compared against the actual development function bodies and real planners.

## Optional PPDDL functions

`robot_planning.pddl.probabilistic` retains numeric-function initialization,
metric conversion, ground-action formatting and seeded UCT-MCTS planning.
They are independent of the required subgoal evaluator and import `pyppddl`
only when `solve_ppddl` is called.

The development workspace's backend identified itself as `pyppddl 0.1.0` with
a generic author and no license/source URL. A same-named PyPI package cannot
be assumed equivalent. This repository therefore does not copy that backend
or claim a verified public installer for it. Install an independently sourced,
compatible backend in the Python environment to use this optional API. Missing
backend imports are explicit. The migration validation used an isolated
installation of the supplied development backend to run original and migrated
wrappers against identical deterministic tasks and seeds.

## Validation

Standalone protocol/process tests run with `pytest tests/robot_planning`.
Differential tests additionally accept:

```bash
TGPO_SOURCE_ROOT=/path/to/original/decision_making \
FAST_DOWNWARD_PATH=/path/to/fast-downward.py \
pytest tests/robot_planning/test_evaluation_parity.py \
       tests/robot_planning/test_pddl_runtime.py
```

The reference loader executes original function/class bodies while replacing
only workspace import scaffolding. It does not import the development tree
from production code or substitute the migrated algorithm into the reference.
Tests cover parsing variants, exact seeded problem text, state/action effects,
review messages and normalization, all three modes, partial/empty plans,
nested batches, concurrent real planner calls, timeout cleanup, CLI behavior,
metrics and the optional PPDDL backend when installed. Recorded response
fixtures prove protocol parity, not the behavior of a live model. The separate
opt-in live test uses `TGPO_LIVE_LLM_BASE_URL` and `TGPO_LIVE_LLM_MODEL` to compare
original and migrated end-to-end results on an actual model endpoint.

Live migration validation also used a locally served Qwen2.5-7B-Instruct model
with `temperature=0`, `seed=7`, and `max_tokens=2048`. For both `validate_first`
and `always_eval`, independent original/migrated calls produced identical full
result dictionaries and request messages on the synthetic planning task.
Before explicitly setting a request seed, one pair had identical scores,
grades and plans but different review prose. Temperature zero alone was not
a sufficient exact-text reproducibility setting for that server.
