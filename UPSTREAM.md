# Pinned verl base

- Repository: https://github.com/verl-project/verl
- Commit: `a7b8675f9674999e2c6b75a402d95837eca40a3d` (2025-09-15).
- Upstream version: `0.5.0.dev`; fork version: `0.5.0.dev0+tgpo.1`.
- License: Apache-2.0. Original copyright headers and NOTICE retained; license text is preserved in
  `LICENSES/Apache-2.0.txt`.

The historical development snapshot did not include its own Git metadata.
This commit is the **selected reproducible base**, established by comparing
available upstream history, rather than a claim of recovered checkout history.
315 of the snapshot's 325 framework files exactly match that base. The adjacent
newer verl checkout is not the base. The public framework starts from the
selected commit's 322 `verl/` files, with the following explicit changes:

1. A checked `trainer_class` factory in the existing TaskRunner.
2. A `postprocess_rollouts` hook in the sole PPO/GRPO training loop, resolving
   initial reward before refinement and recomputing log probabilities afterward.
3. TGPO modules and its thin Hydra entry/configuration.
4. An optional SFT system message, enabled by the recipe to preserve the source
   training template, plus positional Series and single-row pandas compatibility.
5. Fork version metadata.
6. A reward-module cache path check, avoiding reuse of another configured file
   in the same interpreter.
7. A minimal event-loop initialization fallback for synchronous FSDP rollout
   workers with recent uvloop.

Canonical task context remains normalized in parquet `extra_info`; the TGPO
trainer promotes only configured metadata fields at its input boundary. Of the
322 baseline framework files, 316 remain byte-for-byte unchanged. New TGPO
modules and task resources are additions, not copies of a second framework.

`docs/tgpo/framework_manifest.json` maps upstream Git blobs, original snapshot
SHA-256 hashes and released framework hashes. New task/algorithm modules are
listed by purpose in `docs/tgpo/migration.md`; they are not upstream files.
See `docs/tgpo/upstream_training_notes.md` for the original snapshot differences
and the algorithm invariants and fixes.

No custom checkpoint/debug package, duplicate trainer/config tree, unrelated
evaluation method, private dataset, compiled planner or model weights are
vendored. Existing upstream save/restore and model merger remain available.
