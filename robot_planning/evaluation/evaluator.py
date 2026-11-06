"""The subgoal evaluator used by both training reward and offline scoring."""

from robot_planning.evaluation.parsing import (
    extract_related_obj_list_from_reply_or_pair,
)
from robot_planning.evaluation.validation import eval_validate, eval_validate_batch
from robot_planning.evaluation.reviewer import (
    SummaryEvaluator,
    resolve_prompt,
    batch_evaluate_list,
)
from robot_planning.clients.llm import create_default_llm_client

MODES = ("validate_first", "always_eval", "only_validate")


def _validated(result):
    return (
        result.get("parsable", False)
        and result.get("subgoal_correct", False)
        and result.get("solvable_score", 0) > 0
    )


def _compose(validation, review, mode):
    result = {
        key: validation.get(key, default)
        for key, default in (
            ("parsable", False),
            ("subgoal_correct", False),
            ("solvable_score", 0),
            ("solvable_score_dense", 0),
            ("plan", []),
            ("predicate_correctness_list", []),
            ("related_obj_correctness_list", []),
        )
    }
    valid = validation.get("validate_result", _validated(validation))
    result.update(
        validate_result=valid,
        llm_evaluated=review is not None,
        llm_score=0,
        llm_summary="no llm evaluation",
        evaluate="no llm evaluation",
        llm_result="no llm evaluation",
        llm_response="",
        final_result=0,
    )
    if "solver_results" in validation:
        result["solver_results"] = validation["solver_results"]
    if validation.get("error_info"):
        result["error_info"] = validation["error_info"]
    if review is not None:
        result.update(
            llm_score=review.get("score", 0),
            llm_summary=review.get("summary", "no parsed summary"),
            evaluate=review.get("evaluate", "no parsed evaluate"),
            llm_result=review.get("result", "no parsed result"),
            llm_response=review.get("response", ""),
        )
        if not review.get("llm_success", True):
            result["llm_error"] = review.get("llm_error", "Unknown LLM error")
    result["final_result"] = (
        int(valid) if mode == "only_validate" else int(valid) * result["llm_score"]
    )
    return result


def _should_review(validation, mode):
    return bool(validation.get("plan")) and (
        mode == "always_eval" or (mode == "validate_first" and _validated(validation))
    )


def final_eval_single(
    instruction,
    scene_graph,
    reply=None,
    subgoal_related_obj_list_pair=None,
    domain_file_path=None,
    eval_mode="validate_first",
    system_prompt="eval_plan_summary_1006_2",
    exp_model="deepseek-v3",
    api_key=None,
    use_cpu_server=True,
    *,
    llm_client=None,
    solver_path=None,
    diagnostics=False,
):
    """Validate, solve and optionally review one prediction.

    Set only_validate for CPU-only PDDL scoring without a model endpoint.
    The returned keys and score range match the original final_eval_subgoal API.
    """
    if eval_mode not in MODES:
        raise ValueError(f"Unknown evaluation mode: {eval_mode}")
    if (reply is None) == (subgoal_related_obj_list_pair is None):
        raise ValueError(
            "Provide exactly one of reply and subgoal_related_obj_list_pair."
        )
    validation = eval_validate(
        scene_graph,
        reply,
        subgoal_related_obj_list_pair,
        domain_file_path,
        solver_path,
        diagnostics,
    )
    review = None
    error = None
    should_eval = eval_mode == "always_eval" or (
        eval_mode == "validate_first" and _validated(validation)
    )
    if should_eval:
        if not validation.get("plan"):
            error = "No plan available for LLM evaluation (subgoal not solvable)"
        else:
            try:
                client = llm_client or create_default_llm_client(
                    use_cpu_server=use_cpu_server
                )
                evaluator = SummaryEvaluator(
                    client, resolve_prompt(system_prompt), exp_model, api_key
                )
                objects = extract_related_obj_list_from_reply_or_pair(
                    reply, subgoal_related_obj_list_pair
                )
                review = evaluator.evaluate(
                    instruction,
                    scene_graph,
                    validation["plan"],
                    model=exp_model,
                    api_key=api_key,
                    related_obj_list=objects or None,
                )
            except Exception as exception:
                error = str(exception)
    result = _compose(validation, review, eval_mode)
    if error:
        result["llm_error"] = error
    return result


def final_eval_batch(
    data_list,
    domain_file_path=None,
    eval_mode="validate_first",
    system_prompt="eval_plan_summary_1006_2",
    exp_model="deepseek-v3",
    api_key=None,
    use_cpu_server=True,
    max_workers=32,
    *,
    llm_client=None,
    solver_path=None,
    diagnostics=False,
):
    """Score rows or per-row prediction lists, preserving their nested order.

    Missing-scene rows keep their own error slot. This explicitly fixes the old
    index shift where a missing scene could select a later row's validation.
    As in the reference protocol, a global domain overrides per-row domains.
    """
    if eval_mode not in MODES:
        raise ValueError(f"Unknown evaluation mode: {eval_mode}")
    if not data_list:
        return []
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    validation_inputs, row_map = [], []
    for row in data_list:
        scene = row.get("searched_sg") or row.get("scene_graph")
        if scene is None:
            row_map.append(None)
            continue
        row_map.append(len(validation_inputs))
        validation_inputs.append(dict(row, searched_sg=scene))
    validations = eval_validate_batch(
        validation_inputs,
        domain_file_path,
        return_all_results=True,
        max_workers=max_workers,
        solver_path=solver_path,
        diagnostics=diagnostics,
    )
    tasks, task_map = [], {}
    for row_index, (row, validation_index) in enumerate(zip(data_list, row_map)):
        if validation_index is None:
            continue
        values = validations[validation_index]
        values = values if isinstance(values, list) else [values]
        for item_index, validation in enumerate(values):
            if not _should_review(validation, eval_mode):
                continue
            reply = row.get("reply")
            pair = row.get("subgoal_related_obj_list_pair")
            if row.get("reply_list") is not None:
                reply, pair = row["reply_list"][item_index], None
            elif row.get("subgoal_related_obj_list_pair_list") is not None:
                pair, reply = (
                    row["subgoal_related_obj_list_pair_list"][item_index],
                    None,
                )
            task_map[row_index, item_index] = len(tasks)
            tasks.append(
                dict(
                    instruction=row.get("instruction", ""),
                    scene_graph=row.get("searched_sg") or row.get("scene_graph"),
                    plan=validation.get("plan", []),
                    system_prompt=system_prompt,
                    exp_model=row.get("exp_model") or exp_model,
                    api_key=row.get("api_key") or api_key,
                    related_obj_list=extract_related_obj_list_from_reply_or_pair(
                        reply, pair
                    )
                    or None,
                )
            )
    reviews = (
        batch_evaluate_list(tasks, use_cpu_server, max_workers, llm_client=llm_client)
        if tasks
        else []
    )
    results = []
    for row_index, validation_index in enumerate(row_map):
        if validation_index is None:
            results.append(
                dict(
                    error="Data item missing 'searched_sg' or 'scene_graph'",
                    final_result=0,
                    validate_result=False,
                    parsable=False,
                    subgoal_correct=False,
                    solvable_score=0,
                    llm_evaluated=False,
                    llm_score=0,
                )
            )
            continue
        validation = validations[validation_index]
        values = validation if isinstance(validation, list) else [validation]
        row_results = [
            _compose(
                value,
                reviews[task_map[row_index, i]] if (row_index, i) in task_map else None,
                eval_mode,
            )
            for i, value in enumerate(values)
        ]
        results.append(row_results if isinstance(validation, list) else row_results[0])
    return results


def evaluate_repeated(data_list, repeats=1, **kwargs):
    """Repeat the entire evaluation; keep the minimum score and its full record.

    A tie chooses the later run, matching the original double-evaluation reward.
    Repeating only the reviewer would change PDDL sampling and the protocol.
    """
    if repeats < 1:
        raise ValueError("repeats must be positive")
    result = final_eval_batch(data_list, **kwargs)

    def minimum(first, second):
        if isinstance(first, list):
            return [minimum(a, b) for a, b in zip(first, second)]
        return first if first["final_result"] < second["final_result"] else second

    for _ in range(repeats - 1):
        result = [
            minimum(a, b) for a, b in zip(result, final_eval_batch(data_list, **kwargs))
        ]
    return result
