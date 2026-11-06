# AHAT repository integration

The integration starts from AHAT commit
`0990e759babb7eccacf05e2a79904116bd49608b`. The prepared TGPO release is
placed directly at the repository root, preserving the tested `verl` and
`robot_planning` module paths and the fixed framework manifest.

- Remove `src/ahat` and its `ahat` console entry; keep one task implementation.
- Preserve AHAT paper links, citation, figures, public model/dataset links and
  `examples/ahat_domain.pddl`.
- Replace installation metadata with the TGPO dependency groups and add a small
  optional `download` group. Keep the distribution name `verl` so fork version
  reporting remains accurate; the project title is TGPO.
- Anchor downloaded-data and planner ignore rules at the repository root, so
  `robot_planning/data` and required `verl/third_party` source remain tracked.
- Adapt download/deployment scripts to run without importing deleted modules.
  Use the shared runtime environment variable names shown in `.env.example`.
- Local generation uses vLLM; old `ahat pipeline api` generation is removed.
  Review and repair continue using the shared OpenAI-compatible HTTP client.
- Retain original MIT notices separately from Apache-2.0 code and upstream NOTICE.
- Carry over the earlier release exclusions; no private data, model weights,
  compiled solver, custom debug/checkpoint tools or extra evaluator is added.

The existing AHAT domain is a full household domain. The recipe's minimal
synthetic domain remains independent: replacing it would invalidate its smoke
fixtures. Domain compatibility with each dataset is the caller's responsibility.

The October validation records describe the TGPO release before repository
integration. They are historical evidence, not claims that all GPU comparisons
were rerun for a documentation/packaging migration. Integration validation checks
packaging, module/resource preservation, standalone CPU tests, utility entry
points and a real Fast Downward example.

## Integration checks (2026-11-06)

- 396 files across `verl/`, `robot_planning/`, `recipe/` and `tests/` match the
  prepared TGPO source byte for byte; all 322 framework fingerprints pass.
- Standalone installed-package CPU suite: 25 passed, 43 explicitly skipped
  because private reference source, GPU/model services or optional backends
  were not supplied. Real Fast Downward integration cases executed.
- Clean Python 3.11 environment, `[data,test]` dependencies, `pip check` passed.
- Actual planner evaluation of the synthetic example returned score 1.
- Utility `--help` entries and deployment command argument assembly verified;
  large model/dataset downloads and a new GPU server were not launched.
- Ruff passed for task code, TGPO modules, tests and scripts; Python/shell syntax
  and documentation links checked. Both wheel and source distribution built.
- Earlier AHAT assets/domain/MIT license preserved byte for byte; deleted
  package imports and release exclusions audited before commit.

GPU training numerical parity is documented in the earlier validation report;
this integration does not claim a new full GPU training run.
