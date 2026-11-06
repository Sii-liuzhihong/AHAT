#!/usr/bin/env bash
set -euo pipefail
exec python -m robot_planning.evaluation.cli --candidates first "$@"
