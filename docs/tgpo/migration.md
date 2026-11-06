# Migration scope

This release follows the author's scope instructions and release-plan v2.
It is a selective implementation in a fixed verl fork, not a copy of the
development workspace. Source paths below identify provenance; public code
does not import them. Tests can compare a separately supplied private checkout
using `TGPO_SOURCE_ROOT`.

| Development component | Release implementation |
| --- | --- |
| `custom/trainer/custom_ray_trainer.py` | Sole public PPO loop + `verl/trainer/tgpo/trainer.py` hook |
| `custom/trainer/custom_ppo.py` | Existing TaskRunner + thin `main_tgpo.py` |
| `custom/fix_manager/fix_cot_utils.py` | `tgpo/selection.py` and complete row replacement in `batch_ops.py` |
| `custom/fix_manager/concate_cot.py` | `batch_ops.py`; task text wrappers in `robot_planning/repair.py` |
| `custom/fix_manager/fix_cot.py`, selected subgoal repair logic | Tensor-free task repair + shared client |
| Main subgoal reward functions | Thin `robot_planning/reward.py`, shared evaluator; repeat/min as configuration |
| `evaluate/final_eval/final_eval_subgoal.py` | `evaluation/evaluator.py`, single and batch APIs |
| Active subgoal parsing, correctness, solvability | `evaluation/parsing.py`, `validation.py` |
| Active SummaryEvaluator, batch message builder | `evaluation/reviewer.py`, versioned text prompts |
| `fix_plan_pddl` state/parser/goal and action generation | `scene_graph/`, `pddl/`; one implementation per capability |
| Probabilistic planning wrapper | `pddl/probabilistic.py`, optional external `pyppddl` backend |
| Approved data creation utilities | `data/legacy.py`, canonical SFT/RL converters, prompt builders |
| `fix_plan_pddl/process_data` task tree utilities | `data/task_tree.py`: hierarchical IDs, action grouping, numeric serialization and effect rendering |
| `fix_plan_pddl/process_data` effect/goal utilities | `data/effects.py`: net effects, related objects, annotation/goal conversion and history filtering; shared PDDL parser |
| Selected checkpoint eval loader/generation | `inference.py`, existing-prediction CLI; one metrics implementation |

## Explicit exclusions

- All `custom/checkpoint_manager` and `custom/debug`, including renamed copies.
- `custom/original_trainer`, debug trainers, duplicate upstream config files,
  manual experiment launchers, checkpoint scans and dated/copy backup versions.
- Old e2e/final-goal reward functions; unrelated e2e/final-goal/two-stage/ranking
  evaluation, eval web viewers and legacy package exports importing those paths.
- Private source-data extraction, raw datasets, historical experiment outputs,
  credentials, file queues, machine paths, compiled binaries and model weights.

The useful data-creation functions in a directory named `e2e_data_process`
do not imply inclusion of an e2e evaluation method. Legacy data helpers are
pure transforms of caller-provided records; they do not read the raw dataset.
Useful PDDL functionality is retained and consolidated, including capabilities
outside the main evaluator. The optional probabilistic backend's dependency
provenance is documented in evaluation.md; its library is not silently vendored.

## Deliberate behavior changes

Structural migration preserves thresholds, top-k/argmax ties, strict candidate
improvement, prefix training, GRPO advantage, score normalization, full double
evaluation and sequential scene updates. Known source defects are corrected
explicitly: stale positions, reward/metadata alignment, async resolution,
worker divisibility, failed/empty repairs, missing-scene indexing, and planner
return/configuration errors. Mixed-domain reward batches are rejected rather
than silently scored against the first domain.

The older `data_structure.extract_goal_from_problem_file` appended an extra
closing parenthesis for a single goal: `(p a)` became `(p a))`. Its public name
now aliases the canonical `extract_pddl_goals`, matching the newer source
goal extractor and sharing the framework-independent S-expression parser.
This data-utility correction is separate from final-evaluation behavior. The
historical task-tree renderer still prints overflow action groups as Python
list literals; that legacy format is not a lossless tree roundtrip. The loose
data-check `compare_plan` retains its source placement-action equivalences
and is never used as the final evaluator.

Consult upstream_training_notes.md and evaluation.md for each change. Tests
compare untouched source function bodies in the applicable domain; corrected
defects have separate regression cases. Native source failures and runtime
adapters are recorded as differences, not reported as source equivalence.
