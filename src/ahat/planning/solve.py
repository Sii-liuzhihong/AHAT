from pathlib import Path
from typing import Any, Dict, List, Optional
from ahat.scene_graph import SceneGraphManager
from ahat.evaluation import (
    parse_subgoal_and_related_obj_from_cot,
    PDDLSolvabilityChecker
)
from ahat.schemas.typings import Paths


class SolveCoordinator:
    """
    Coordinates PDDL plan solving from LLM replies.
    
    Encapsulates the workflow of:
    1. Parsing LLM replies to extract subgoals
    2. Building scene graphs
    3. Solving PDDL plans sequentially
    """

    @staticmethod
    def _failure_result(
        parsable: bool = False,
        subgoal_list: Optional[List] = None,
    ) -> Dict[str, Any]:
        """Generate a failure result dict with empty plan and false success flag."""
        return {
            "plan": [],
            "success": False,
            "parsable": parsable,
            "subgoal_list": subgoal_list or [],
            "solvable_score_dense": 0.0,
        }

    @staticmethod
    def from_llm_reply(
        reply: str,
        scene_graph: Dict[str, Any],
        domain_file_path: Paths,
        output_dir: Paths,
    ) -> Dict[str, Any]:
        """
        Parse a model reply and solve the PDDL plan for each embedded subgoal.

        Pipeline
        --------
        1. Parse ``reply`` → ``(subgoal_list, related_obj_list)`` via
           :func:`~evaluator.subgoal._parsability.parse_subgoal_and_related_obj_from_cot`.
        2. Build a :class:`~utils.pddl.scene_graph.SceneGraphManager` from the
           current scene graph.
        3. Call :func:`PDDLSolvabilityChecker.check`
           to find a plan for every subgoal in sequence.

        Args:
            reply: Raw model output string in CoT format containing
                ``Subtask goal:`` and ``Implicitly related object:`` lines.
            scene_graph: Current environment state as a scene-graph dict.
            domain_file_path: Path to the PDDL domain file, or raw domain
                content as a string.
            pddl_solver_path: Path to the solver shell script (e.g.
                ``path/to/run_pddl.sh``).

        Returns:
            A dict with the following keys:

            ``"plan"`` (``List[str]``)
                Concatenated list of all grounded action strings from every
                solved subgoal.  Empty when the reply cannot be parsed or no
                plan is found.

            ``"success"`` (``bool``)
                ``True`` only when plans were found for **all** subgoals.

            ``"parsable"`` (``bool``)
                ``True`` when ``reply`` was successfully parsed into subgoals.

            ``"subgoal_list"`` (``List[List[str]]``)
                Parsed subgoal predicates, one inner list per subtask step.
                Empty when the reply is not parsable.

            ``"solvable_score_dense"`` (``float``)
                Fraction of subgoals for which a plan was found (0.0 – 1.0).
        """
        try:
            subgoal_list, related_obj_list = parse_subgoal_and_related_obj_from_cot(reply)
        except Exception:
            return SolveCoordinator._failure_result(parsable=False)

        if not subgoal_list or not related_obj_list:
            return SolveCoordinator._failure_result(parsable=False)

        try:
            sg_manager = SceneGraphManager(scene_graph, domain_file_path)
            subgoal_pair_list = list(zip(subgoal_list, related_obj_list))
            result = PDDLSolvabilityChecker.check(
                subgoal_pair_list, sg_manager, domain_file_path, output_dir
            )
        except Exception as exc:
            print(f"[SolveCoordinator.from_llm_reply] Solver error: {exc}")
            return SolveCoordinator._failure_result(parsable=True, subgoal_list=subgoal_list)

        return {
            "plan": result["plan"],
            "success": bool(result["score"]),
            "parsable": True,
            "subgoal_list": subgoal_list,
            "solvable_score_dense": result["dense_score"],
        }

    @staticmethod
    def solve(subgoal_reply: str, scene_graph: dict, domain_file_path: Paths, output_dir: Paths) -> dict:
        """
        Parse LLM subgoal output and generate PDDL plans for each subgoal.

        Main entry point that coordinates the full solving pipeline: parses the
        LLM reply into structured subgoals, generates PDDL problems, and solves
        them using FastDownward. Results are saved to the output directory.

        Args:
            subgoal_reply: Raw string output from the LLM containing subgoal definitions.
            scene_graph: Dictionary representation of the environment state,
                        containing objects, attributes, and relationships.
            domain_file_path: Path to the PDDL domain file defining predicates/actions.
            output_dir: Directory where PDDL subproblems and plans are written.

        Returns:
            A dictionary containing:
                - plan: List of action tuples from the solved PDDL problems
                - success: Boolean indicating if all subgoals were solvable
                - parsable: Boolean indicating if the LLM reply was successfully parsed
                - subgoal_list: List of extracted subgoal dictionaries
                - solvable_score_dense: Float score (0.0-1.0) of solvable ratio

        Raises:
            FileNotFoundError: If domain_file_path or output_dir does not exist.

        Example:
            >>> result = SolveCoordinator.solve(
            ...     "subgoal: move cup from table to shelf",
            ...     {"objects": ["cup", "table", "shelf"]},
            ...     Path("domain.pddl"),
            ...     Path("./output")
            ... )
            >>> print(result["success"])
            True
        """
        domain_file_path = Path(domain_file_path).resolve()
        output_dir = Path(output_dir).resolve()
        result = SolveCoordinator.from_llm_reply(
            reply=subgoal_reply,
            scene_graph=scene_graph,
            domain_file_path=domain_file_path,
            output_dir=output_dir
        )
        
        return result


