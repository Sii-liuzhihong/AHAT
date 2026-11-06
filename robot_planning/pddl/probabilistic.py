"""Optional PPDDL conversion and UCT-MCTS planning (requires pyppddl)."""

import os
import re
import tempfile
from typing import List, Optional, Tuple


def _get_initialized_functions(problem_pddl: str) -> set:
    pattern = re.compile(
        "\\(=\\s*\\(\\s*([a-zA-Z][a-zA-Z0-9\\-_]*)\\s*\\)", re.IGNORECASE
    )
    return set(pattern.findall(problem_pddl))


def _convert_pddl_to_ppddl_problem(problem_pddl: str, domain) -> str:
    func_names = [f.name for f in getattr(domain, "functions", [])]
    if not func_names:
        return problem_pddl
    initialized = _get_initialized_functions(problem_pddl)
    missing = [fn for fn in func_names if fn not in initialized]
    result = problem_pddl
    if missing:
        init_injections = "\n    ".join((f"(= ({fn}) 0)" for fn in missing))
        result, n_replaced = re.subn(
            "(\\(:init\\b)", "\\1\\n    " + init_injections, result, flags=re.IGNORECASE
        )
        if n_replaced == 0:
            init_block = f"  (:init\n    {init_injections}\n  )\n"
            result = re.sub(
                "(\\(:goal\\b)", init_block + "\\1", result, flags=re.IGNORECASE
            )
    if (
        not re.search(":\\s*metric", result, re.IGNORECASE)
        and "total-reward" in func_names
    ):
        result = result.rstrip()
        if result.endswith(")"):
            result = result[:-1].rstrip() + "\n  (:metric maximize (total-reward))\n)"
    return result


def _ground_action_to_pddl_str(action) -> str:
    raw = repr(action)
    m = re.match("([^(]+)\\((.*)\\)$", raw, re.DOTALL)
    if not m:
        return f"({raw})"
    name = m.group(1).strip()
    args_str = m.group(2).strip()
    if args_str:
        args = [a.strip() for a in args_str.split(",")]
        return f"({name} {' '.join(args)})"
    return f"({name})"


def solve_ppddl(
    domain_ppddl: str,
    problem_pddl: str,
    num_iterations: int = 1000,
    max_steps: int = 50,
    discount: float = 0.99,
    rollout_depth: int = 50,
    seed: Optional[int] = None,
) -> Tuple[List[str], bool]:
    from pyppddl.ppddl.parser import load_domain, load_problem
    from pyppddl.mdp.ppddl_mdp import PPDDLMDP
    from pyppddl.solvers.mcts import run_mcts_episode

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            domain_path = os.path.join(temp_dir, "domain.ppddl")
            problem_path = os.path.join(temp_dir, "problem.ppddl")
            with open(domain_path, "w", encoding="utf-8") as f:
                f.write(domain_ppddl)
            domain = load_domain(domain_path)
            ppddl_problem_str = _convert_pddl_to_ppddl_problem(problem_pddl, domain)
            with open(problem_path, "w", encoding="utf-8") as f:
                f.write(ppddl_problem_str)
            problem = load_problem(problem_path)
            mdp = PPDDLMDP(domain, problem, discount=discount, max_steps=max_steps)
            trajectory, _ = run_mcts_episode(
                mdp,
                num_iterations=num_iterations,
                max_rollout_depth=rollout_depth,
                seed=seed,
                verbose=False,
            )
        plan = [_ground_action_to_pddl_str(action) for _, action, _, _ in trajectory]
        success = False
        if trajectory:
            last_next_state = trajectory[-1][2]
            success = mdp._check_goal(last_next_state)
        return (plan, success)
    except Exception as e:
        print(f"PPDDL 求解错误: {e}")
        import traceback

        traceback.print_exc()
        return ([], False)
