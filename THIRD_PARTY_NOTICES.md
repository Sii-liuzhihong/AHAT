# License scope

The existing AHAT `assets/`, `examples/ahat_domain.pddl`, and adapted
`scripts/download_model.py`, `scripts/download_data.py`, and
`scripts/deploy_model.py` retain the MIT license in the root LICENSE.
The imported TGPO implementation (`verl/`, `robot_planning/`, `recipe/tgpo/`,
tests and validation/setup tools) is distributed under Apache-2.0, whose text is
in LICENSES/Apache-2.0.txt. Existing copyright headers and NOTICE are retained.
The root MIT license does not relicense the Apache-licensed code.

# Third-party code and dependencies

The `verl/` package is derived from [verl](https://github.com/verl-project/verl),
licensed under Apache-2.0. Original copyright headers and NOTICE are retained; the upstream license text
is in LICENSES/Apache-2.0.txt. See UPSTREAM.md for the exact base and local changes.

Fast Downward is an external executable, not embedded in the Python wheel.
`scripts/setup_fast_downward.sh` downloads release-24.06.1 from
https://github.com/aibasel/downward and verifies its SHA-256:
`994374021146f5d587a3559af016a24d56c404cbbed3d6d431f28c58b5e8b802`.
Fast Downward is GPL-3.0-or-later; its own LICENSE.md and component notices
remain in the downloaded source directory. Review those terms when distributing
a built solver or container. The build is kept in the ignored `third_party/` tree.

PyTorch, transformers, Ray, vLLM and other dependencies are installed separately
under their respective licenses; none of their model weights are distributed.
The example domain and tasks are synthetic and do not contain the private
research dataset.
