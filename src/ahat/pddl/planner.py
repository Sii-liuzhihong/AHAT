"""
PDDL Planner Interface
======================

This module provides a wrapper around the Fast Downward planner (or compatible
PDDL solvers). It handles process execution, timeout management, parallel
planning, and result parsing.

Architecture Position:
    [Tools / Planner] -> Used by task_planner.planning and atomic_domain modules
    to solve PDDL problems.
"""

import json
import os
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv

from ahat.configs.constants import EnvVar, JSONKey
from ahat.schemas.typings import Paths
from ahat.utils.runtime import setup_path

load_dotenv()

__all__ = ["PDDLPlanner", "parse_plans"]


class _PDDLRecorder:
    """Internal helper to track the total running time of the planner."""

    def __init__(self):
        self.planner_time = 0

    def record(self, planner_time: float) -> None:
        self.planner_time += planner_time

    def clear_record(self) -> None:
        self.planner_time = 0

    def write_record(self, save_path: Paths) -> None:
        save_path = setup_path(save_path, is_file=True)

        metrics = {}
        if save_path.exists():
            try:
                with save_path.open("r", encoding="utf-8") as f:
                    metrics = json.load(f)
            except json.JSONDecodeError:
                pass

        # Update metrics
        metrics[JSONKey.TOTAL_PLANNER_TIME] = self.planner_time

        with save_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=4)
        
        self.clear_record()

class PDDLPlanner:
    """Wrapper for executing PDDL planning tasks."""

    @staticmethod
    def _get_fast_downward_path() -> str:
        """Get the path to the Fast Downward executable from the environment variable.
        
        Raises:
            EnvironmentError: If the environment variable is not set or does not point to a valid file.
        """

        fast_downward_path = os.environ.get(EnvVar.FAST_DOWNWARD_PATH)
        if not fast_downward_path:
            raise EnvironmentError(
                f"Environment variable '{EnvVar.FAST_DOWNWARD_PATH}' is not set. "
                "Please configure it in your .env file."
            )
            
        fast_downward_path = Path(fast_downward_path)
        if not fast_downward_path.is_file():
            raise EnvironmentError(
                f"Fast Downward path '{fast_downward_path}' does not exist or is not a file."
            )

        return str(fast_downward_path)

    def __init__(self):
        self._recorder = _PDDLRecorder()
        self._fast_downward_path = self._get_fast_downward_path()

    def solve_pddl(
        self,
        domain_file: Paths,
        problem_file: Paths,
        output_path: Paths,
        timer: bool = True,
        timeout: float = 300,
    ) -> tuple[list, bool]:
        """Invoke the external Fast Downward planner to solve a PDDL problem.

        If the planner succeeds, the plan is written to `output_path`.
        If it fails, the error log (stdout/stderr) is written to `output_path` instead.

        Args:
            domain_file (Paths): Path to the domain.pddl.
            problem_file (Paths): Path to the problem.pddl.
            output_path (Paths): Path where the plan (or error log) will be saved.
            timer (bool): Whether to include this run in the total time metrics.
            timeout (float): Max execution time in seconds. Defaults to 300.

        Raises:
            RuntimeError: If EnvVar.FAST_DOWNWARD_PATH does not point to a valid Fast Downward executable.
        """

        # Standard Fast Downward configuration (A* with lmcut heuristic)
        # Note: Adjust search strategy if needed for specific domains
        command = [
            "python", self._fast_downward_path,
            "--alias", "lama-first",
            "--plan-file", output_path,
            str(domain_file),
            str(problem_file)
        ]

        start_time = time.perf_counter_ns()

        # Use a temporary directory as CWD to keep the workspace clean
        # Fast Downward creates intermediate files like output.sas
        # Avoid polluting the workspace with intermediate files
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                subprocess.run(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True,
                    timeout=timeout,
                    cwd=temp_dir
                )
                # On success, Fast Downward writes to --plan-file (output_path) automatically.

                with open(output_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                plan = [
                    line.strip()
                    for line in lines
                    if line.strip() and not line.strip().startswith(";")
                ]
                solvable = True
            
            except subprocess.CalledProcessError as e:
                # Environment variable does not point to a valid Fast Downward executable
                if e.stderr:
                    raise RuntimeError(
                        f"Run Fast Downward failed: {e.stderr}. "
                        "This is likely the path to Fast Downward is incorrect. "
                        f"Please check {self._fast_downward_path}"
                    )
                
                # On failure, write the error log to the output path for debugging analysis
                planner_error = e.stdout
                output_path = setup_path(output_path, is_file=True)
                output_path.write_text(planner_error, encoding="utf-8")
                solvable = False
                

            except subprocess.TimeoutExpired:
                output_path = setup_path(output_path, is_file=True)
                output_path.write_text(f"Planner timed out after {timeout} seconds.", encoding="utf-8")
                solvable = False

        if timer:
            planner_time = (time.perf_counter_ns() - start_time) / 1e9
            self._recorder.record(planner_time)
        
        return plan, solvable

    def parallel_solve_pddl(
        self,
        files: List[Tuple[Paths, Paths, Paths]],
        num_workers: Optional[int] = 5
    ) -> None:
        """Execute multiple PDDL planning tasks in parallel.

        Args:
            files (List[Tuple]): List of (domain_path, problem_path, output_path).
            num_workers (Optional[int]): Number of parallel threads. Defaults to 5.

        Returns:
            None
        """

        start_time = time.perf_counter_ns()
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Map future to input arguments for error reporting
            future_to_paths = {
                executor.submit(
                    self.solve_pddl,
                    domain_path,
                    problem_path,
                    output_path,
                    timer=False # Don't double count time inside individual threads
                ): (domain_path, problem_path)
                for domain_path, problem_path, output_path in files
            }

            for future in as_completed(future_to_paths):
                paths = future_to_paths[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing domain '{paths[0]}' and problem '{paths[1]}': {e}")

        # Record total wall-clock time for the batch
        total_time = (time.perf_counter_ns() - start_time) / 1e9
        self._recorder.record(total_time)

    def write_record(self, save_path: Paths) -> None:
        self._recorder.write_record(save_path)

def parse_plans(plan_path: Paths) -> List[str]:
    """Parse a Fast Downward plan file into a list of action strings.

    Robustly handles cases where the plan file might contain error logs
    or be empty. Standard plans contain actions like `(pick-up block_a)`
    and metadata lines starting with `;`.

    Args:
        plan_path (Paths): Path to the planner output file.

    Returns:
        List[str]: A list of cleaned action strings (e.g., "pick-up block_a").
                   Returns empty list if no valid plan is found.
    """
    plan_path = setup_path(plan_path, is_file=True, mkdir=False)

    if not plan_path.exists():
        return []

    with plan_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    parsed_plans = []
    for line in lines:
        line = line.strip()
        # Skip empty lines and comments (Fast Downward uses ; for cost info)
        if not line or line.startswith(";"):
            continue
        
        # Valid actions typically start with '('
        if line.startswith("("):
            # Remove enclosing parentheses: "(action arg)" -> "action arg"
            clean_action = line.strip("()")
            parsed_plans.append(clean_action)
        else:
            # If encountering lines that are not comments and don't look like actions
            # (e.g., "Planner failed..."), this is likely an error log, not a plan.
            # In this case, we treat it as no plan found.
            pass

    return parsed_plans
