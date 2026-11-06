"""Planner process protocol tests in addition to real-planner differential tests."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from robot_planning.pddl.solver import solve_pddl, solve_pddl_detailed
from robot_planning.evaluation.metrics import aggregate_metrics


@pytest.mark.parametrize(
    "exitcode,status",
    [
        (0, "no_solution"),
        (10, "no_solution"),
        (12, "no_solution"),
        (7, "planner_error"),
    ],
)
def test_solver_statuses(tmp_path, exitcode, status):
    script = tmp_path / "adapter.sh"
    script.write_text(f"exit {exitcode}\n")
    result = solve_pddl_detailed("(domain)", "(problem)", str(script))
    assert result.status == status and not result.success and not result.plan
    assert solve_pddl("(domain)", "(problem)", str(script)) == ([], False)


def test_timeout_stops_process_group(tmp_path):
    script, pidfile = tmp_path / "adapter.sh", tmp_path / "pid"
    script.write_text('echo $$ > "' + str(pidfile) + '"\nsleep 30\n')
    start = time.monotonic()
    result = solve_pddl_detailed("(domain)", "(problem)", str(script), timeout=0.1)
    assert result.status == "timeout" and not result.success
    assert time.monotonic() - start < 3
    assert not Path("/proc").joinpath(pidfile.read_text().strip()).exists()


def test_solver_reads_only_action_lines(tmp_path):
    script = tmp_path / "adapter.sh"
    script.write_text('printf "; cost=1\\n\\n(clean cup_1)\\n" > "$1"\n')
    assert solve_pddl("domain", "problem", str(script)) == (["(clean cup_1)"], True)


def test_metrics_denominators_and_infrastructure_failures():
    result = aggregate_metrics(
        [
            [
                dict(final_result=1, llm_evaluated=True, llm_score=1),
                dict(
                    final_result=0,
                    llm_evaluated=False,
                    llm_score=0,
                    solver_results=[dict(status="timeout")],
                ),
            ],
            dict(
                final_result=0,
                llm_evaluated=True,
                llm_score=0,
                llm_error="endpoint failed",
            ),
        ]
    )
    assert (
        result["n_tasks"] == 2
        and result["n_candidates"] == result["metric_denominator"] == 3
    )
    assert result["n_llm_evaluated"] == 2
    assert result["mean_llm_score_on_evaluated"] == 0.5
    assert result["mean_llm_score_all"] == 1 / 3
    assert result["n_api_failures"] == result["n_candidates_with_planner_errors"] == 1


def test_cli_arbitrary_cwd_and_manifest(tmp_path):
    planner = os.environ.get("FAST_DOWNWARD_PATH")
    if not planner:
        pytest.skip("Set FAST_DOWNWARD_PATH for real planner CLI integration")
    package_root = Path(__file__).resolve().parents[2]
    example = package_root / "recipe/tgpo/examples/predictions.jsonl"
    output = tmp_path / "result"
    env = dict(os.environ)
    # A source-tree run uses an absolute PYTHONPATH; an installed package needs none.
    env["PYTHONPATH"] = str(package_root)
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "robot_planning.evaluation.cli",
            "--input",
            str(example),
            "--output-dir",
            str(output),
            "--mode",
            "only_validate",
            "--workers",
            "1",
            "--candidates",
            "all",
            "--solver",
            planner,
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert run.returncode == 0, run.stderr
    metrics = json.loads((output / "metrics.json").read_text())
    manifest = json.loads((output / "run_config.json").read_text())
    row = json.loads((output / "results.jsonl").read_text())
    assert metrics["final_result"] == 1 and metrics["n_solver_calls"] == 1
    assert len(manifest["input_sha256"]) == len(manifest["prompt_sha256"]) == 64
    assert row["id"] == "synthetic-clean-table" and row["candidate_index"] == 0
    assert row["evaluation"]["plan"] == ["(wipe robot table kitchen)"]
