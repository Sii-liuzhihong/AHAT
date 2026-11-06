"""Aggregate task and candidate metrics with explicit denominators and failures."""

from collections import Counter


def aggregate_metrics(results):
    flat = [
        item for row in results for item in (row if isinstance(row, list) else [row])
    ]
    count = len(flat)
    fields = (
        "parsable",
        "subgoal_correct",
        "solvable_score",
        "solvable_score_dense",
        "validate_result",
        "llm_evaluated",
        "llm_score",
        "final_result",
    )
    means = {
        field: sum(float(row.get(field, 0)) for row in flat) / count if count else 0.0
        for field in fields
    }
    reviewed = [row for row in flat if row.get("llm_evaluated")]
    solver_counts = Counter(
        item["status"] for row in flat for item in row.get("solver_results", [])
    )
    return dict(
        count=count,
        n_tasks=len(results),
        n_candidates=count,
        n_llm_evaluated=len(reviewed),
        metric_denominator=count,
        mean_llm_score_all=means["llm_score"],
        mean_llm_score_on_evaluated=(
            sum(row.get("llm_score", 0) for row in reviewed) / len(reviewed)
            if reviewed
            else None
        ),
        n_llm_errors=sum(bool(row.get("llm_error")) for row in flat),
        n_api_failures=sum(bool(row.get("llm_error")) for row in reviewed),
        n_input_errors=sum(
            bool(row.get("error") or row.get("error_info")) for row in flat
        ),
        n_solver_calls=sum(solver_counts.values()),
        solver_status_counts=dict(solver_counts),
        n_candidates_with_planner_errors=sum(
            any(
                item["status"] in ("timeout", "planner_error", "configuration_error")
                for item in row.get("solver_results", [])
            )
            for row in flat
        ),
        **means
    )
