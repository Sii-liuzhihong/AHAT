"""Typed task and scoring records; runtime APIs also accept ordinary dictionaries."""

from typing import Any, Literal, TypedDict

EvaluationMode = Literal["validate_first", "always_eval", "only_validate"]


class TaskRecord(TypedDict, total=False):
    instruction: str
    scene_graph: dict[str, Any]
    searched_sg: dict[str, Any]
    domain_file_path: str
    reply: str
    reply_list: list[str]
    subgoal_related_obj_list_pair: list
    subgoal_related_obj_list_pair_list: list


class EvaluationResult(TypedDict, total=False):
    parsable: bool
    subgoal_correct: int
    solvable_score: float
    solvable_score_dense: float
    predicate_correctness_list: list[int]
    related_obj_correctness_list: list[bool]
    plan: list[str]
    validate_result: bool
    llm_evaluated: bool
    llm_score: float
    llm_summary: str
    evaluate: str
    llm_result: str
    llm_response: str
    llm_error: str
    final_result: float
    error_info: str
