"""Subgoal correctness and sequential planning validation.

Each subtask solves cumulative prefixes, then applies the resulting actions
before constructing the next problem. Parallelizing these dependent goals
would change the reward protocol. Object type checking deliberately matches
the reference: name, arity and object presence, without an extra type gate.
"""

import pathlib
from typing import List, Dict, Union, Optional
import re
from multiprocessing import Pool
from robot_planning.evaluation.parsing import check_parsability
from robot_planning.pddl.problem import PDDLProblemGenerator
from robot_planning.pddl.solver import solve_pddl
from robot_planning.scene_graph.manager import SceneGraphManager


def parse_predicate_str(predicate_str):
    predicate_tokens = (
        predicate_str.replace("(", " ( ").replace(")", " ) ").split()[1:-1]
    )
    if predicate_tokens[0] == "not":
        predicate_tokens = predicate_tokens[2:-1]
    parsed_predicate = {"name": predicate_tokens[0], "params": predicate_tokens[1:]}
    return parsed_predicate


def check_predicate_correctness(predicate_str, sg_manager):
    try:
        predicate = parse_predicate_str(predicate_str)
        definition = sg_manager.pddl_parser.predicates.get(predicate["name"])
        return int(
            definition is not None
            and len(definition) == len(predicate["params"])
            and all((name in sg_manager.object_map for name in predicate["params"]))
        )
    except Exception:
        return 0


def check_related_obj_correctness(related_objs, sg_manager):
    return all((name in sg_manager.object_map for name in related_objs))


def extent_goal_list(goal_list):
    return [goal_list[: index + 1] for index in range(len(goal_list))] or [[]]


def extent_goal_related_obj(goal_list, related_obj, problem_generater):
    all_obj_list = set(related_obj)
    for goal in goal_list:
        structured_g = problem_generater.parse_parentheses(goal)
        _, objects = problem_generater._parse_goal_and_extract_objects(structured_g)
        all_obj_list.update(objects)
    return list(all_obj_list)


def check_subgoal_solvable(
    subgoal_pair_list, sg_manager, domain_file_path, pddl_solver_path
):
    problem_generator = PDDLProblemGenerator(sg_manager)
    domain_file_str = pathlib.Path(domain_file_path).read_text("utf-8")
    solvable_list = []
    plan_list = []
    solver_results = []
    for i, goal_pair in enumerate(subgoal_pair_list):
        goal_list, related_obj = (goal_pair[0], goal_pair[1])
        extented_goal_list = extent_goal_list(goal_list)
        extented_obj_list = extent_goal_related_obj(
            goal_list, related_obj, problem_generator
        )
        for goal in extented_goal_list:
            pddl_problem_file_content, _ = problem_generator.generate_problem_file(
                [goal, extented_obj_list]
            )
            plan, solvable = solve_pddl(
                domain_file_str,
                pddl_problem_file_content,
                pddl_solver_path,
                diagnostics=solver_results,
            )
            solvable_list.append(int(solvable))
            plan_list.extend(plan)
            for action in plan:
                action_param_list = re.findall("\\b[a-zA-Z0-9_-]+\\b", action)
                sg_manager.apply_action(action_param_list[0], action_param_list[1:])
    return {
        "score": all(solvable_list),
        "dense_score": sum(solvable_list) / len(solvable_list),
        "plan": plan_list,
        "solver_results": solver_results,
    }


def eval_subgoal_validate(
    scene_graph,
    domain_file_path,
    subgoal_related_obj_list_pair=None,
    solver_path=None,
    diagnostics=False,
):
    result = dict(
        parsable=False,
        subgoal_correct=0,
        predicate_correctness_list=[],
        related_obj_correctness_list=[],
        solvable_score=0,
        solvable_score_dense=0,
        plan=[],
    )
    try:
        subgoals, related = (
            subgoal_related_obj_list_pair[0],
            subgoal_related_obj_list_pair[1],
        )
        result["parsable"] = type(subgoals) is list and bool(subgoals)
        if not result["parsable"]:
            return result
        manager = SceneGraphManager(scene_graph, domain_file_path)
        for goals, objects in zip(subgoals, related):
            for predicate in goals:
                result["predicate_correctness_list"].append(
                    check_predicate_correctness(predicate, manager)
                )
                # Keep one related-object flag per predicate, aligned for repair.
                result["related_obj_correctness_list"].append(
                    check_related_obj_correctness(objects, manager)
                )
        result["subgoal_correct"] = int(
            all(result["predicate_correctness_list"])
        ) * int(all(result["related_obj_correctness_list"]))
        if result["subgoal_correct"]:
            solved = check_subgoal_solvable(
                list(zip(subgoals, related)), manager, domain_file_path, solver_path
            )
            result.update(
                solvable_score=solved["score"],
                solvable_score_dense=solved["dense_score"],
                plan=solved["plan"],
            )
            if diagnostics:
                result["solver_results"] = solved["solver_results"]
        elif diagnostics:
            result["solver_results"] = []
        return result
    except (FileNotFoundError, RuntimeError):
        # Broken run configuration must not become a zero task reward.
        raise
    except Exception as error:
        result = dict(
            parsable=False,
            subgoal_correct=0,
            predicate_correctness_list=[],
            related_obj_correctness_list=[],
            solvable_score=0,
            solvable_score_dense=0,
            plan=[],
        )
        if diagnostics:
            result.update(
                error_info=str(error),
                solver_results=[dict(status="input_error", success=False, plan=[])],
            )
        return result


def eval_validate(
    scene_graph: Union[Dict, str],
    reply: Optional[str] = None,
    subgoal_related_obj_list_pair: Optional[List[str]] = None,
    domain_file_path: Optional[str] = None,
    solver_path=None,
    diagnostics=False,
) -> Dict:
    if subgoal_related_obj_list_pair is not None:
        processed_subgoal = subgoal_related_obj_list_pair
    else:
        parsable, subgoal_list, related_obj_list = check_parsability(reply)
        processed_subgoal = [subgoal_list, related_obj_list]
    verify_result_dict = eval_subgoal_validate(
        scene_graph=scene_graph,
        domain_file_path=domain_file_path,
        subgoal_related_obj_list_pair=processed_subgoal,
        solver_path=solver_path,
        diagnostics=diagnostics,
    )
    return verify_result_dict


def _eval_validate_single_item(
    data: Dict,
    global_domain_file_path: Optional[str],
    return_all_results: bool,
    solver_path=None,
    diagnostics=False,
) -> Union[Dict, List[Dict]]:
    reply = data.get("reply")
    subgoal_pair = data.get("subgoal_related_obj_list_pair")
    reply_list = data.get("reply_list")
    subgoal_pair_list = data.get("subgoal_related_obj_list_pair_list")
    scene_graph = data.get("searched_sg") or data.get("scene_graph")
    if scene_graph is None:
        raise ValueError(f"Data item missing 'searched_sg' or 'scene_graph': {data}")
    item_domain_file_path = global_domain_file_path or data.get("domain_file_path")
    if return_all_results:
        if reply_list is not None:
            item_results = []
            for r in reply_list:
                result = eval_validate(
                    scene_graph=scene_graph,
                    reply=r,
                    subgoal_related_obj_list_pair=None,
                    domain_file_path=item_domain_file_path,
                    solver_path=solver_path,
                    diagnostics=diagnostics,
                )
                item_results.append(result)
            return item_results
        elif subgoal_pair_list is not None:
            item_results = []
            for p in subgoal_pair_list:
                result = eval_validate(
                    scene_graph=scene_graph,
                    reply=None,
                    subgoal_related_obj_list_pair=p,
                    domain_file_path=item_domain_file_path,
                    solver_path=solver_path,
                    diagnostics=diagnostics,
                )
                item_results.append(result)
            return item_results
    if reply is None and reply_list:
        reply = reply_list[0] if len(reply_list) > 0 else None
    if subgoal_pair is None and subgoal_pair_list:
        subgoal_pair = subgoal_pair_list[0] if len(subgoal_pair_list) > 0 else None
    result = eval_validate(
        scene_graph=scene_graph,
        reply=reply,
        subgoal_related_obj_list_pair=subgoal_pair,
        domain_file_path=item_domain_file_path,
        solver_path=solver_path,
        diagnostics=diagnostics,
    )
    return result


def eval_validate_batch(
    data_list: List[Dict],
    domain_file_path: Optional[str] = None,
    return_all_results: bool = False,
    max_workers=32,
    solver_path=None,
    diagnostics=False,
) -> List[Union[Dict, List[Dict]]]:
    if not data_list:
        return []
    with Pool(processes=max_workers) as pool:
        results = pool.starmap(
            _eval_validate_single_item,
            [
                (data, domain_file_path, return_all_results, solver_path, diagnostics)
                for data in data_list
            ],
        )
    return results
