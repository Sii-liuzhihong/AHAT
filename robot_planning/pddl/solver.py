"""Isolated Fast Downward calls with bounded execution and process cleanup."""

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile


@dataclass(frozen=True)
class SolveResult:
    plan: list[str]
    success: bool
    status: str
    returncode: int | None = None


def _stop(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=1)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def solve_pddl_detailed(domain_pddl, problem_pddl, solver_path=None, timeout=5):
    """Run lama-first and distinguish no solution, timeout and planner failure.

    A shell adapter receives plan/domain/problem arguments. A Python driver
    receives the Fast Downward CLI. Each call owns its temporary directory and
    process group; timeouts clean up descendants as well as the parent.
    """
    configured = solver_path or os.environ.get("FAST_DOWNWARD_PATH")
    if not configured:
        raise RuntimeError(
            "Set FAST_DOWNWARD_PATH to fast-downward.py or pass solver_path."
        )
    executable = Path(configured).expanduser().resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"Planner not found: {executable}")
    with tempfile.TemporaryDirectory(prefix="tgpo-pddl-") as directory:
        directory = Path(directory)
        domain, problem, plan = [
            directory / name for name in ("domain.pddl", "problem.pddl", "plan.txt")
        ]
        domain.write_text(domain_pddl, encoding="utf-8")
        problem.write_text(problem_pddl, encoding="utf-8")
        if executable.suffix == ".py":
            command = [
                sys.executable,
                str(executable),
                "--alias",
                "lama-first",
                "--plan-file",
                str(plan),
                str(domain),
                str(problem),
            ]
        else:
            command = ["bash", str(executable), str(plan), str(domain), str(problem)]
        process = subprocess.Popen(
            command,
            cwd=directory,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            process.communicate(timeout=timeout)
            # Fast Downward: translator/search unsolvable (10/11), incomplete
            # search (12), and time limits (21/23/24). Other nonzero codes
            # remain infrastructure failures under the reference protocol.
            if process.returncode in (10, 11, 12):
                return SolveResult([], False, "no_solution", process.returncode)
            if process.returncode in (21, 23, 24):
                return SolveResult([], False, "timeout", process.returncode)
            if process.returncode != 0:
                return SolveResult([], False, "planner_error", process.returncode)
            if not plan.is_file():
                return SolveResult([], False, "no_solution", process.returncode)
            actions = [
                line.strip()
                for line in plan.read_text().splitlines()
                if line.strip() and not line.strip().startswith(";")
            ]
            return SolveResult(actions, True, "success", process.returncode)
        except subprocess.TimeoutExpired:
            return SolveResult([], False, "timeout")
        finally:
            _stop(process)


def solve_pddl(
    domain_pddl, problem_pddl, solver_path=None, timeout=5, diagnostics=None
):
    """Return the reference ``(plan, success)`` pair, with optional diagnostics.

    The missing-plan-file branch consistently returns ``([], False)``; the
    development implementation accidentally returned a bare list there.
    """
    result = solve_pddl_detailed(domain_pddl, problem_pddl, solver_path, timeout)
    if diagnostics is not None:
        diagnostics.append(asdict(result))
    return result.plan, result.success
