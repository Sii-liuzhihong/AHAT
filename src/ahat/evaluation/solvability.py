"""
Solvability Checker
-------------------
Checks whether each subgoal in a list can be reached (solved) by a PDDL
planner, starting from the current scene-graph state.

For each subgoal the checker iteratively tests all prefix sub-lists of the
goal predicates (e.g. for goals [A, B, C] it tests [A], [A,B], [A,B,C]).
After a plan is found for one prefix it is applied to the scene graph so
that the next subgoal check starts from the updated state.
"""

from pathlib import Path
import re
from typing import Any, Dict, List, Set, Tuple

from ahat.pddl.problem_generator import PDDLProblemGenerator
from ahat.pddl.planner import PDDLPlanner
from ahat.utils.runtime import setup_path
from ahat.scene_graph import SceneGraphManager
from ahat.schemas.typings import Paths


class PDDLSolvabilityChecker:
    """
    Checks whether each subgoal in a list can be reached (solved) by a PDDL planner.
    
    Provides methods for:
    - Building prefix sub-lists of goals
    - Expanding goal-related objects
    - Checking subgoal solvability with progressive scene-graph updates
    """

    @staticmethod
    def extend_goal_list(goal_list: List[Any]) -> List[List[Any]]:
        """
        Build all prefix sub-lists of *goal_list*.

        Example::

            extend_goal_list(["A", "B", "C"])
            -> [["A"], ["A", "B"], ["A", "B", "C"]]

        An empty input returns ``[[]]``.
        """
        if not goal_list:
            return [[]]
        extended = [[] for _ in range(len(goal_list))]
        for i, goal in enumerate(goal_list):
            for j in range(i, len(goal_list)):
                extended[j].append(goal)
        return extended

    @staticmethod
    def extend_goal_related_objects(
        goal_list: List[str],
        related_obj: List[str],
        problem_generator: PDDLProblemGenerator,
    ) -> List[str]:
        """
        Expand *related_obj* with all objects referenced by predicates in
        *goal_list* (i.e. union of direct participants).

        Args:
            goal_list: List of goal predicate strings.
            related_obj: Initial seed list of related object names.
            problem_generator: An initialised problem generator (used to parse
                goal predicates).

        Returns:
            Expanded list of related object names (no duplicates).
        """
        all_objs: Set[str] = set(related_obj)
        for goal in goal_list:
            structured = problem_generator.parse_parentheses(goal)
            _, objects = problem_generator._parse_goal_and_extract_objects(structured)
            all_objs.update(objects)
        return list(all_objs)

    @staticmethod
    def check(
        subgoal_pair_list: List[Tuple[List[str], List[str]]],
        sg_manager: SceneGraphManager,
        domain_file: Paths,
        output_dir: Paths
    ) -> Dict[str, Any]:
        """
        Test whether each subgoal in *subgoal_pair_list* is reachable from the
        current (and progressively updated) scene-graph state.

        For each ``(goal_list, related_obj)`` pair the function:
            1. Builds all prefix sub-lists of *goal_list*.
            2. Generates a PDDL problem file for each prefix.
            3. Calls the external solver.
            4. Applies the resulting plan to the scene graph so subsequent
               checks start from the new state.

        Args:
            subgoal_pair_list: List of ``(goal_list, related_obj)`` tuples.
            sg_manager: Scene-graph manager (mutated in place as plans are applied).
            domain_file_path: Path to the PDDL domain file.
            pddl_solver_path: Path to the solver shell script.

        Returns:
            ``{"score": bool, "dense_score": float, "plan": List[str]}``

            - *score*: ``True`` only when all subgoals are solvable.
            - *dense_score*: Fraction of solvable subgoals.
            - *plan*: Concatenated plan actions from all solved subgoals.
        """
        problem_generator = PDDLProblemGenerator(sg_manager)
        solvable_flags: List[int] = []
        plan_list: List[str] = []

        planner = PDDLPlanner()

        for i, (goal_list, related_obj) in enumerate(subgoal_pair_list):
            subtask_dir = Path(output_dir) / f'subtask_{i}'
            problem_file_path = subtask_dir / 'problem.pddl'
            setup_path(problem_file_path, is_file=True)  
            output_path = subtask_dir / 'plan.txt'
            goal_prefixes = PDDLSolvabilityChecker.extend_goal_list(goal_list)
            extended_obj_list = PDDLSolvabilityChecker.extend_goal_related_objects(
                goal_list, related_obj, problem_generator
            )

            for prefix in goal_prefixes:
                pddl_problem, _ = problem_generator.generate_problem_file(
                    [prefix, extended_obj_list]
                )
                with open(problem_file_path, "w", encoding="utf-8") as f:
                    f.write(pddl_problem)
                plan, solvable = planner.solve_pddl(domain_file=domain_file, problem_file=problem_file_path, output_path=output_path)

                solvable_flags.append(int(solvable))
                plan_list.extend(plan)

                for action in plan:
                    tokens = re.findall(r"\b[a-zA-Z0-9_-]+\b", action)
                    sg_manager.apply_action(tokens[0], tokens[1:])

        if not solvable_flags:
            return {"score": False, "dense_score": 0.0, "plan": []}

        return {
            "score": bool(all(solvable_flags)),
            "dense_score": sum(solvable_flags) / len(solvable_flags),
            "plan": plan_list,
        }


# Backward compatibility: module-level functions wrapping class methods
def extend_goal_list(goal_list: List[Any]) -> List[List[Any]]:
    """Deprecated: Use PDDLSolvabilityChecker.extend_goal_list() instead."""
    return PDDLSolvabilityChecker.extend_goal_list(goal_list)


def extend_goal_related_objects(
    goal_list: List[str],
    related_obj: List[str],
    problem_generator: PDDLProblemGenerator,
) -> List[str]:
    """Deprecated: Use PDDLSolvabilityChecker.extend_goal_related_objects() instead."""
    return PDDLSolvabilityChecker.extend_goal_related_objects(
        goal_list, related_obj, problem_generator
    )


