"""The sole verl reward adapter, delegating subgoal scoring to offline evaluation."""

import json
import os

LLM_REVIEW_SYSTEM_PROMPT = "eval_plan_summary_1006_testing_2"
MAX_WORKERS = 128


def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos=None, *,
                        repeats=1, max_workers=MAX_WORKERS, llm_client=None,
                        system_prompt=LLM_REVIEW_SYSTEM_PROMPT, evaluator=None,
                        sandbox_fusion_url=None, concurrent_semaphore=None,
                        memory_limit_mb=None, **kwargs):
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if extra_infos is None or len(extra_infos) != len(solution_strs):
        raise ValueError("extra_infos must align with solution_strs")
    if not solution_strs:
        return []
    if evaluator is None:
        from robot_planning.evaluation.evaluator import final_eval_batch
        evaluator = final_eval_batch
    if "exp_model" not in kwargs and os.environ.get("TGPO_LLM_MODEL"):
        kwargs["exp_model"] = os.environ["TGPO_LLM_MODEL"]
    inputs = []
    for reply, info in zip(solution_strs, extra_infos):
        scene = info["scene_graph"]
        if isinstance(scene, str):
            scene = json.loads(scene)
        inputs.append({"scene_graph": scene, "reply_list": [reply],
                       "instruction": info["instruction"], "domain_file_path": info["domain_file_path"]})
    domains = {row["domain_file_path"] for row in inputs}
    if len(domains) > 1:
        raise ValueError("A reward batch must use one PDDL domain; mixed domains were silently mis-scored in the source")
    selected = None
    for _ in range(repeats):
        options = {"data_list": inputs, "eval_mode": "validate_first", "system_prompt": system_prompt,
                   "domain_file_path": inputs[0]["domain_file_path"], "max_workers": max_workers, **kwargs}
        if llm_client is not None:
            options["llm_client"] = llm_client
        results = evaluator(**options)
        if len(results) != len(inputs) or any(len(row) != 1 for row in results):
            raise ValueError("Evaluator results must align one-to-one with reward inputs")
        # The whole selected result follows the smaller score; ties choose last.
        selected = results if selected is None else [old if old[0]["final_result"] < new[0]["final_result"] else new
                                                   for old, new in zip(selected, results)]
    return [{"score": row[0]["final_result"], "evaluate": row[0]["evaluate"], "reply": reply,
             "reward_validate": row[0]["solvable_score"]} for row, reply in zip(selected, solution_strs)]


def compute_total_score_subgoal_batch(*args, **kwargs):
    return compute_score_batch(*args, **kwargs)


def compute_total_score_double_subgoal_batch(*args, **kwargs):
    kwargs["repeats"] = 2
    return compute_score_batch(*args, **kwargs)
