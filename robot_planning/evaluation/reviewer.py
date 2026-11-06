"""Single-stage Summary review and its original normalized score protocol."""

from concurrent.futures import ThreadPoolExecutor
from importlib.resources import files
import json

from robot_planning.clients.llm import create_default_llm_client
from robot_planning.evaluation.parsing import get_content, convert_to_parenthesis_style
from robot_planning.scene_graph.summary import sum_sg_content, get_related_objs_info

# Single-item and batch entry points historically resolve these prompt names.
PROMPT_NAMES = ("1006", "1006_2", "1006_testing", "1006_testing_2", "0104", "test_1011")


def resolve_prompt(name, *, batch=False):
    prefix = "eval_plan_summary_"
    version = name.removeprefix(prefix) if name.startswith(prefix) else None
    allowed = (
        ("1006", "1006_2", "1006_testing_2", "0104", "test_1011")
        if batch
        else ("1006", "1006_2", "1006_testing")
    )
    if version in allowed:
        return (
            files("robot_planning")
            .joinpath("prompts", f"review_{version}.txt")
            .read_text()
        )
    return name


def normalize_plan(plan):
    if not plan:
        return []
    try:
        if plan[0][0] == "(" and plan[0][-1] == ")":
            return plan
        return [convert_to_parenthesis_style(action) for action in plan]
    except (IndexError, TypeError):
        return plan if isinstance(plan, list) else []


class SummaryEvaluator:
    """Evaluate an action plan with an injectable, explicitly configured client."""

    def __init__(
        self,
        llm_client,
        system_prompt,
        default_model="deepseek-v3",
        default_api_key=None,
    ):
        self.llm_client = llm_client
        self.system_prompt = system_prompt
        self.default_model = default_model
        self.default_api_key = default_api_key

    def _prepare_input(self, instruction, scene_graph, plan, related_obj_list=None):
        if isinstance(scene_graph, str):
            scene_graph = json.loads(scene_graph)
        content = f"House_content: {sum_sg_content(scene_graph)}"
        if related_obj_list:
            content += get_related_objs_info(related_obj_list, scene_graph)
        plan_text = "\n".join(plan) if isinstance(plan, list) else str(plan)
        return content + f"\nAction_sequence: {plan_text}\nInstruction: {instruction}"

    def _build_messages(self, user_content):
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _get_default_result(self):
        return dict(
            score=0,
            summary="no parsed summary",
            evaluate="no parsed evaluate",
            result="no parsed result",
            response="",
        )

    def parse_response(self, response):
        summary = get_content(response, "Summary")
        evaluate = get_content(response, "Evaluate")
        result = get_content(response, "Result")
        normalized = (
            result.replace("\n", "").replace("\r", "").replace(" ", "").lower()
            if result
            else ""
        )
        # The actual training reward is normalized, despite the old 2/1/0 comments.
        score = {"good": 1.0, "normal": 0.5, "bad": 0.0}.get(normalized, 0)
        return dict(
            summary=summary or "no parsed summary",
            evaluate=evaluate or "no parsed evaluate",
            score=score,
            result=result or "no parsed result",
            response=response,
        )

    def evaluate(
        self,
        instruction,
        scene_graph,
        plan,
        model=None,
        api_key=None,
        related_obj_list=None,
        **kwargs,
    ):
        messages = self._build_messages(
            self._prepare_input(instruction, scene_graph, plan, related_obj_list)
        )
        response = self.llm_client.call(
            messages=messages,
            model_name=model or self.default_model,
            api_key=api_key or self.default_api_key,
            **kwargs,
        )
        if response.success:
            result = self.parse_response(response.content)
            return dict(
                result, llm_response=response.content, llm_success=True, llm_error=None
            )
        return dict(
            self._get_default_result(),
            llm_response="",
            llm_success=False,
            llm_error=response.error,
            llm_retry_count=response.retry_count,
        )


def batch_evaluate_list(
    data_list, use_cpu_server=True, max_workers=32, llm_client=None
):
    """Review independent plans concurrently and preserve input order."""
    if not data_list:
        return []
    client = llm_client or create_default_llm_client(use_cpu_server=use_cpu_server)

    def evaluate(data):
        plan = normalize_plan(data.get("plan"))
        if (
            not all(
                [data.get("instruction"), data.get("scene_graph"), data.get("plan")]
            )
            or not plan
        ):
            error = (
                "Missing required fields: instruction, scene_graph, or plan"
                if not all(
                    [data.get("instruction"), data.get("scene_graph"), data.get("plan")]
                )
                else f"Plan processing failed: plan is empty or invalid. Original plan: {data.get('plan')}"
            )
            return dict(
                SummaryEvaluator(client, "")._get_default_result(),
                llm_success=False,
                llm_error=error,
                llm_response="",
                plan_processing_error=True,
                original_plan=data.get("plan"),
            )
        reviewer = SummaryEvaluator(
            client,
            resolve_prompt(
                data.get("system_prompt", "eval_plan_summary_1006_2"), batch=True
            ),
            data.get("exp_model", "deepseek-v3"),
            data.get("api_key"),
        )
        return reviewer.evaluate(
            data["instruction"],
            data["scene_graph"],
            plan,
            related_obj_list=data.get("related_obj_list"),
        )

    tasks = [data for data in data_list for _ in range(data.get("repeat_num", 1))]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(evaluate, tasks))
