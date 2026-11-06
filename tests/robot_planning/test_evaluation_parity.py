"""Behavioral parity against caller-supplied development source.

The public test suite runs standalone. Set TGPO_REFERENCE_ROOT to the original
workspace to execute the differential cases too. Only import/path scaffolding
is replaced in the reference loader; source function bodies run unchanged.
"""

import ast
from abc import ABC, abstractmethod
import copy
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import runpy
import sys
import tempfile
import types
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from robot_planning.evaluation import evaluator as public
from robot_planning.evaluation.parsing import check_parsability
from robot_planning.evaluation.reviewer import (
    SummaryEvaluator,
    resolve_prompt,
)
from robot_planning.pddl.problem import PDDLProblemGenerator
from robot_planning.pddl.solver import solve_pddl
from robot_planning.scene_graph.manager import SceneGraphManager
from robot_planning.scene_graph import summary
from robot_planning.clients.llm import LLMResponse

DOMAIN = """(define (domain habitat_world_domain_final)
 (:requirements :strips :typing :negative-preconditions)
 (:types room furniture agent)
 (:predicates (is_clean ?obj - object) (is_open ?obj - object) (is_filled ?obj - object)
  (item_on_surface ?obj - object ?surface - furniture)
  (furniture_in_room ?furniture - furniture ?room - room)
  (agent_at ?agent - agent ?location - furniture) (handempty ?agent - agent))
 (:action clean :parameters (?obj - object) :precondition (not (is_clean ?obj)) :effect (is_clean ?obj))
 (:action open :parameters (?obj - object) :precondition (is_clean ?obj) :effect (is_open ?obj)))"""
SCENE = {
    "rooms": [{"name": "room_1", "states": {}}],
    "furnitures": [{"name": "table_1", "states": {}}]
    + [{"name": f"chair_{i}", "states": {"is_clean": True}} for i in range(5)],
    "objects": [{"name": "cup_1", "states": {"is_clean": False, "is_open": False}}],
    "agent": [{"name": "agent_0", "location": "table_1", "holding": None}],
    "links": [["room_1", "table_1", "in"], ["table_1", "cup_1", "on"]]
    + [["room_1", f"chair_{i}", "in"] for i in range(5)],
}
REPLY = "Subtask goal: ['(is_clean cup_1)', '(is_open cup_1)']\nImplicitly related object: [cup_1, table_1]\n"


class FixedClient:
    """Recorded protocol response; deliberately does not stand in for real-LLM validation."""

    def __init__(self, result="Good", fail=False):
        self.messages = []
        self.result = result
        self.fail = fail
        self.call_fn = self._legacy_call

    def _legacy_call(self, messages, model_name=None, api_key=None):
        return ""

    def call(self, messages, model_name=None, api_key=None, **kwargs):
        self.messages.append(messages)
        return LLMResponse(
            content=f"<Summary>summary</Summary><Evaluate>review</Evaluate><Result>{self.result}</Result>",
            success=not self.fail,
            error="recorded failure" if self.fail else None,
            retry_count=0,
        )


def _functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names
    ]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


@pytest.fixture(scope="module")
def legacy():
    root = os.environ.get("TGPO_SOURCE_ROOT") or os.environ.get("TGPO_REFERENCE_ROOT")
    if not root:
        pytest.skip(
            "Set TGPO_REFERENCE_ROOT for original-source differential verification"
        )
    base = Path(root) / "policy_iteration/iter"
    module = types.ModuleType("_tgpo_reference_evaluation")
    sys.modules[module.__name__] = module
    ns = module.__dict__
    ns.update(
        dict(
            re=re,
            ast=ast,
            json=json,
            copy=copy,
            random=random,
            List=List,
            Dict=Dict,
            Any=Any,
            Union=Union,
            Set=Set,
            Tuple=Tuple,
            Optional=Optional,
            Literal=Literal,
            Pool=Pool,
            ThreadPoolExecutor=ThreadPoolExecutor,
            threading=threading,
        )
    )
    import pathlib

    ns["pathlib"] = pathlib
    for file, names in (
        (
            "fix_plan_pddl/update_sg/dynamic_sg.py",
            ["PDDLDomainParser", "SceneGraphManager"],
        ),
        ("fix_plan_pddl/goal_solver/problem_generater.py", ["PDDLProblemGenerator"]),
        (
            "fix_plan_pddl/subgoal_eval/eval_parsable.py",
            [
                "_parse_related_object_list",
                "parse_effect_and_related_obj_from_cot",
                "check_parsability",
            ],
        ),
        (
            "fix_plan_pddl/subgoal_eval/eval_correct.py",
            [
                "parse_predicate_str",
                "check_predicate_correctness",
                "check_related_obj_correctness",
            ],
        ),
        (
            "fix_plan_pddl/subgoal_eval/eval_solvable.py",
            ["extent_goal_list", "extent_goal_related_obj", "check_subgoal_solvable"],
        ),
        ("fix_plan_pddl/subgoal_eval/eval_subgoal.py", ["eval_subgoal_validate"]),
        (
            "evaluate/validate_eval/eval_subgoal.py",
            ["eval_validate", "_eval_validate_single_item", "eval_validate_batch"],
        ),
        (
            "utils/data_related_utils.py",
            ["get_content", "convert_to_parenthesis_style"],
        ),
        (
            "evaluate/llm_review_eval/utils/sg_summary.py",
            [
                "sum_sg_content",
                "sum_sg_content_for_read",
                "organize_content_list",
                "get_parents_str",
                "get_obj_details_from_scene_graph",
                "format_obj_details_str",
                "get_related_objs_info",
            ],
        ),
        (
            "evaluate/final_eval/final_eval_subgoal.py",
            [
                "extract_related_obj_list_from_reply_or_pair",
                "final_eval_single",
                "final_eval_batch",
            ],
        ),
    ):
        _functions(base / file, names, ns)
    ns["module_path"] = str(base / "fix_plan_pddl")
    spec = importlib.util.spec_from_file_location(
        "_tgpo_original_solver", base / "fix_plan_pddl/goal_solver/pddl_solver.py"
    )
    original_solver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original_solver)
    ns["solve_pddl"] = original_solver.solve_pddl
    constants = runpy.run_path(
        str(base / "evaluate/llm_review_eval/llm_review_summary/constants.py")
    )
    ns.update(
        {k: v for k, v in constants.items() if k.startswith("eval_plan_summary_")}
    )
    ns.update(ABC=ABC, abstractmethod=abstractmethod, LLMClient=object)
    # This import boundary is the original scene-summary module, whose actual
    # function bodies were loaded above. No evaluator logic is substituted.
    summary_name = "iter.evaluate.llm_review_eval.utils.sg_summary"
    previous_summary = sys.modules.get(summary_name)
    sys.modules[summary_name] = module
    _functions(
        base / "evaluate/llm_review_eval/base/base_evaluator.py", ["BaseEvaluator"], ns
    )
    _functions(
        base / "evaluate/llm_review_eval/evaluators/summary_evaluator.py",
        ["SummaryEvaluator"],
        ns,
    )
    ns["PddlPlanParser"] = (
        lambda: None
    )  # instantiated but never used by the active normalization path
    _functions(
        base / "evaluate/llm_review_eval/utils/plan_processor.py", ["PlanProcessor"], ns
    )
    ns["SYSTEM_PROMPT_DICT"] = {
        k: constants[k]
        for k in (
            "eval_plan_summary_1006",
            "eval_plan_summary_1006_2",
            "eval_plan_summary_1006_testing_2",
            "eval_plan_summary_0104",
            "eval_plan_summary_test_1011",
        )
    }
    _functions(
        base / "evaluate/llm_review_eval/llm_review_summary/evaluate_plan.py",
        ["_create_evaluator", "batch_evaluate_list"],
        ns,
    )
    ns["create_default_llm_client"] = lambda **kwargs: FixedClient()
    yield module
    if previous_summary is None:
        sys.modules.pop(summary_name, None)
    else:
        sys.modules[summary_name] = previous_summary
    sys.modules.pop(module.__name__, None)


@pytest.fixture
def domain(tmp_path):
    path = tmp_path / "domain.pddl"
    path.write_text(DOMAIN)
    return str(path)


@pytest.fixture(scope="module")
def planner():
    configured = os.environ.get("FAST_DOWNWARD_PATH")
    if not configured:
        pytest.skip("Set FAST_DOWNWARD_PATH to run real Fast Downward integration")
    assert Path(configured).is_file()
    return configured


@pytest.mark.parametrize(
    "reply",
    [
        REPLY,
        REPLY.replace("cup_1, table_1", "'cup_1', table_1"),
        REPLY.replace("cup_1, table_1", "'cup_1', 'table_1'"),
        "",
        "invalid",
        REPLY.replace("[cup_1, table_1]", "[cup_1,]"),
        "Subtask goal: []\nImplicitly related object: []",
    ],
)
def test_parsing_parity(legacy, reply):
    assert check_parsability(reply) == legacy.check_parsability(reply)


def test_scene_parser_and_problem_parity(legacy, domain):
    original = legacy.SceneGraphManager(copy.deepcopy(SCENE), domain)
    migrated = SceneGraphManager(copy.deepcopy(SCENE), domain)
    assert migrated.scene_graph == original.scene_graph
    assert migrated.pddl_parser.predicates == original.pddl_parser.predicates
    assert migrated.pddl_parser.actions == original.pddl_parser.actions
    for action in [("clean", ["cup_1"]), ("open", ["cup_1"])]:
        original.apply_action(*action)
        migrated.apply_action(*action)
        assert migrated.scene_graph == original.scene_graph
        assert (
            migrated.get_all_predicates_from_sg()
            == original.get_all_predicates_from_sg()
        )
    pair = [["(is_clean cup_1)"], ["cup_1"]]
    for seed in range(8):
        random.seed(seed)
        expected = legacy.PDDLProblemGenerator(original).generate_problem_file(
            copy.deepcopy(pair)
        )
        random.seed(seed)
        actual = PDDLProblemGenerator(migrated).generate_problem_file(
            copy.deepcopy(pair)
        )
        assert actual == expected


def test_summary_parity(legacy):
    for function in ("sum_sg_content", "sum_sg_content_for_read"):
        assert getattr(summary, function)(copy.deepcopy(SCENE)) == getattr(
            legacy, function
        )(copy.deepcopy(SCENE))
    assert summary.get_related_objs_info(
        ["cup_1", "agent_0"], SCENE
    ) == legacy.get_related_objs_info(["cup_1", "agent_0"], SCENE)


@pytest.mark.parametrize("mode", public.MODES)
@pytest.mark.parametrize(
    "reply",
    [
        REPLY,
        REPLY.replace("is_clean", "undefined_predicate"),
        REPLY.replace("(is_open cup_1)", "(is_filled cup_1)"),
        "invalid",
    ],
)
def test_real_solver_single_parity(legacy, domain, planner, mode, reply):
    legacy.create_default_llm_client = lambda **kwargs: FixedClient("Normal")
    random.seed(9)
    expected = legacy.final_eval_single(
        "Clean and open the cup",
        copy.deepcopy(SCENE),
        reply=reply,
        domain_file_path=domain,
        eval_mode=mode,
    )
    random.seed(9)
    actual = public.final_eval_single(
        "Clean and open the cup",
        copy.deepcopy(SCENE),
        reply=reply,
        domain_file_path=domain,
        eval_mode=mode,
        llm_client=FixedClient("Normal"),
        solver_path=planner,
    )
    assert actual == expected
    if reply == REPLY:
        assert actual["plan"] == ["(clean cup_1)", "(open cup_1)"]
        assert actual["final_result"] == (1 if mode == "only_validate" else 0.5)


@pytest.mark.parametrize("mode", public.MODES)
def test_real_solver_nested_batch_parity(legacy, domain, planner, mode):
    legacy.create_default_llm_client = lambda **kwargs: FixedClient()
    data = [
        dict(
            instruction="clean and open",
            scene_graph=copy.deepcopy(SCENE),
            reply_list=[REPLY, "invalid"],
        ),
        dict(
            instruction="clean and open", scene_graph=copy.deepcopy(SCENE), reply=REPLY
        ),
    ]
    expected = legacy.final_eval_batch(
        copy.deepcopy(data), domain_file_path=domain, eval_mode=mode, max_workers=2
    )
    actual = public.final_eval_batch(
        copy.deepcopy(data),
        domain_file_path=domain,
        eval_mode=mode,
        max_workers=2,
        llm_client=FixedClient(),
        solver_path=planner,
    )
    assert actual == expected


def test_missing_scene_does_not_shift_valid_row(domain, planner):
    data = [
        dict(instruction="missing", reply=REPLY),
        dict(instruction="valid", scene_graph=SCENE, reply=REPLY),
    ]
    result = public.final_eval_batch(
        data, domain, "only_validate", max_workers=1, solver_path=planner
    )
    assert result[0]["final_result"] == 0
    assert result[1]["final_result"] == 1


@pytest.mark.parametrize(
    "grade, score", [("Good", 1), (" Normal\n", 0.5), ("Bad", 0), ("unknown", 0)]
)
def test_reviewer_score(grade, score):
    client = FixedClient(grade)
    result = SummaryEvaluator(client, "prompt").evaluate(
        "clean", SCENE, ["(clean cup_1)"]
    )
    assert result["score"] == score
    assert client.messages[0][0] == {"role": "system", "content": "prompt"}


def test_repeated_ties_select_later_full_evaluation(monkeypatch):
    runs = iter(
        [
            [dict(final_result=0.5, llm_summary="first")],
            [dict(final_result=0.5, llm_summary="second")],
        ]
    )
    monkeypatch.setattr(public, "final_eval_batch", lambda *args, **kwargs: next(runs))
    assert public.evaluate_repeated([{}], repeats=2)[0]["llm_summary"] == "second"


@pytest.mark.parametrize(
    "grade,fail",
    [
        ("Good", False),
        ("Normal", False),
        ("Bad", False),
        ("unknown", False),
        ("Good", True),
    ],
)
def test_real_original_reviewer_messages_and_results(legacy, grade, fail):
    old_client, new_client = FixedClient(grade, fail), FixedClient(grade, fail)
    old = legacy.SummaryEvaluator(old_client, legacy.eval_plan_summary_1006_2)
    new = SummaryEvaluator(new_client, resolve_prompt("eval_plan_summary_1006_2"))
    kwargs = dict(
        instruction="clean cup",
        scene_graph=copy.deepcopy(SCENE),
        plan=["(clean cup_1)"],
        related_obj_list=["cup_1"],
    )
    assert new.evaluate(**kwargs) == old.evaluate(**kwargs)
    assert new_client.messages == old_client.messages


def test_action_problem_generator_parity(legacy, domain):
    base = (
        Path((os.environ.get("TGPO_SOURCE_ROOT") or os.environ["TGPO_REFERENCE_ROOT"]))
        / "policy_iteration/iter"
    )
    namespace = dict(legacy.__dict__)
    _functions(
        base / "fix_plan_pddl/goal_solver/problem_generator_plan.py",
        ["PDDLProblemGenerator"],
        namespace,
    )
    old = namespace["PDDLProblemGenerator"](
        legacy.SceneGraphManager(copy.deepcopy(SCENE), domain)
    )
    new = PDDLProblemGenerator(SceneGraphManager(copy.deepcopy(SCENE), domain))
    for actions in ([], ["(clean cup_1)"], ["(clean cup_1)", "(open cup_1)"]):
        for seed in range(5):
            random.seed(seed)
            expected = old.generate_problem_from_actions(actions)
            random.seed(seed)
            assert new.generate_problem_from_actions(actions) == expected


def test_ppddl_actual_source_parity(legacy):
    pytest.importorskip("pyppddl")
    from pyppddl.ppddl.parser import load_domain, load_problem
    from pyppddl.mdp.ppddl_mdp import PPDDLMDP
    from pyppddl.solvers.mcts import run_mcts_episode
    from robot_planning.pddl.probabilistic import solve_ppddl

    namespace = dict(
        legacy.__dict__,
        os=os,
        tempfile=tempfile,
        load_domain=load_domain,
        load_problem=load_problem,
        PPDDLMDP=PPDDLMDP,
        run_mcts_episode=run_mcts_episode,
    )
    source = (
        Path((os.environ.get("TGPO_SOURCE_ROOT") or os.environ["TGPO_REFERENCE_ROOT"]))
        / "policy_iteration/iter/fix_plan_pddl/goal_solver/ppddl_solver.py"
    )
    _functions(
        source,
        [
            "_get_initialized_functions",
            "_convert_pddl_to_ppddl_problem",
            "_ground_action_to_pddl_str",
            "solve_ppddl",
        ],
        namespace,
    )
    domain = "(define (domain clean) (:requirements :strips) (:predicates (dirty) (clean)) (:action wipe :parameters () :precondition (dirty) :effect (and (not (dirty)) (clean))))"
    problem = "(define (problem clean) (:domain clean) (:init (dirty)) (:goal (clean)))"
    for seed in range(3):
        kwargs = dict(num_iterations=15, max_steps=5, seed=seed)
        old = namespace["solve_ppddl"](domain, problem, **kwargs)
        new = solve_ppddl(domain, problem, **kwargs)
        assert new == old == (["(wipe)"], True)


def test_original_empty_action_plan_semantics(legacy, domain, planner):
    reply = (
        "Subtask goal: ['(is_clean chair_1)']\nImplicitly related object: ['chair_1']"
    )
    for mode in public.MODES:
        old = legacy.final_eval_single(
            "keep chair clean",
            copy.deepcopy(SCENE),
            reply=reply,
            domain_file_path=domain,
            eval_mode=mode,
        )
        new = public.final_eval_single(
            "keep chair clean",
            copy.deepcopy(SCENE),
            reply=reply,
            domain_file_path=domain,
            eval_mode=mode,
            solver_path=planner,
            llm_client=FixedClient(),
        )
        assert new == old
        assert new["validate_result"] and new["plan"] == []
        assert new["final_result"] == (1 if mode == "only_validate" else 0)


def test_concurrent_real_solver_isolation(legacy, domain, planner):
    manager = SceneGraphManager(copy.deepcopy(SCENE), domain)
    problem, _ = PDDLProblemGenerator(manager).generate_problem_file(
        [["(is_clean cup_1)"], ["cup_1"]]
    )
    old_script = str(
        Path(os.environ.get("TGPO_SOURCE_ROOT") or os.environ["TGPO_REFERENCE_ROOT"])
        / "policy_iteration/iter/fix_plan_pddl/goal_solver/run_pddl.sh"
    )
    with ThreadPoolExecutor(max_workers=4) as pool:
        original = list(
            pool.map(lambda _: legacy.solve_pddl(DOMAIN, problem, old_script), range(8))
        )
        migrated = list(
            pool.map(lambda _: solve_pddl(DOMAIN, problem, planner), range(8))
        )
    assert original == migrated == [(["(clean cup_1)"], True)] * 8


def test_real_solver_diagnostics(domain, planner):
    result = public.final_eval_single(
        "clean and open",
        SCENE,
        reply=REPLY,
        domain_file_path=domain,
        eval_mode="only_validate",
        solver_path=planner,
        diagnostics=True,
    )
    assert [item["status"] for item in result["solver_results"]] == [
        "success",
        "success",
    ]
    assert result["final_result"] == 1


def test_missing_planner_is_not_scored_as_task_failure(monkeypatch, domain):
    monkeypatch.delenv("FAST_DOWNWARD_PATH", raising=False)
    with pytest.raises(RuntimeError, match="FAST_DOWNWARD_PATH"):
        public.final_eval_single(
            "clean",
            SCENE,
            reply=REPLY,
            domain_file_path=domain,
            eval_mode="only_validate",
        )


def test_spatial_and_conditional_scene_effects_parity(legacy, tmp_path):
    domain_text = (
        DOMAIN[:-1]
        + """
 (:predicates (unused))
 (:action relocate :parameters (?obj - object ?old - furniture ?new - furniture)
  :precondition (item_on_surface ?obj ?old)
  :effect (and (not (item_on_surface ?obj ?old)) (item_in_receptacle ?obj ?new)
               (when (is_clean ?obj) (is_open ?obj)))))"""
    )
    # Add predicates to the existing section (not an additional PDDL section).
    domain_text = domain_text.replace(" (:predicates (unused))\n", "")
    domain_text = domain_text.replace(
        "(handempty ?agent - agent))",
        "(handempty ?agent - agent) (item_in_receptacle ?obj - object ?container - furniture) (next_to ?a - object ?b - object ?surface - furniture))",
    )
    path = tmp_path / "spatial.pddl"
    path.write_text(domain_text)
    old = legacy.SceneGraphManager(copy.deepcopy(SCENE), str(path))
    new = SceneGraphManager(copy.deepcopy(SCENE), str(path))
    for name, args in [
        ("clean", ["cup_1"]),
        ("relocate", ["cup_1", "table_1", "chair_1"]),
    ]:
        old.apply_action(name, args)
        new.apply_action(name, args)
        assert new.scene_graph == old.scene_graph
        concrete = "(" + " ".join([name] + args) + ")"
        assert new.pddl_parser.get_action_effects(
            concrete
        ) == old.pddl_parser.get_action_effects(concrete)
    for predicate in (
        ["next_to", "cup_1", "chair_1", "table_1"],
        ["not", ["next_to", "cup_1", "chair_1", "table_1"]],
        ["agent_at", "agent_0", "chair_1"],
    ):
        old.apply_predicates(copy.deepcopy(predicate))
        new.apply_predicates(copy.deepcopy(predicate))
        assert new.scene_graph == old.scene_graph
    for goal in (
        ["when", ["is_clean", "cup_1"], ["is_open", "cup_1"]],
        ["and", ["is_clean", "cup_1"], ["not", ["is_open", "cup_1"]]],
    ):
        assert new._parse_goal_and_extract_objects(
            goal
        ) == old._parse_goal_and_extract_objects(goal)


def test_ppddl_numeric_conversion_parity(legacy):
    from robot_planning.pddl.probabilistic import _convert_pddl_to_ppddl_problem

    source = (
        Path(os.environ.get("TGPO_SOURCE_ROOT") or os.environ["TGPO_REFERENCE_ROOT"])
        / "policy_iteration/iter/fix_plan_pddl/goal_solver/ppddl_solver.py"
    )
    namespace = dict(legacy.__dict__)
    _functions(
        source,
        ["_get_initialized_functions", "_convert_pddl_to_ppddl_problem"],
        namespace,
    )
    domain = types.SimpleNamespace(
        functions=[
            types.SimpleNamespace(name="total-reward"),
            types.SimpleNamespace(name="cost"),
        ]
    )
    for problem in (
        "(define (problem p) (:init) (:goal (clean)))",
        "(define (problem p) (:init (= (cost) 4)) (:goal (clean)))",
        "(define (problem p) (:goal (clean)))",
        "(define (problem p) (:init) (:goal (clean)) (:metric minimize (cost)))",
    ):
        assert _convert_pddl_to_ppddl_problem(problem, domain) == namespace[
            "_convert_pddl_to_ppddl_problem"
        ](problem, domain)


@pytest.mark.parametrize("mode", ["validate_first", "always_eval"])
def test_live_original_and_public_evaluation(legacy, domain, planner, mode):
    """Opt-in live-model check, separate from recorded-response protocol tests."""
    base_url = os.environ.get("TGPO_LIVE_LLM_BASE_URL")
    model = os.environ.get("TGPO_LIVE_LLM_MODEL")
    if not base_url or not model:
        pytest.skip(
            "Set TGPO_LIVE_LLM_BASE_URL and TGPO_LIVE_LLM_MODEL for real-model parity"
        )
    from robot_planning.clients.llm import OpenAICompatibleClient

    class CapturingClient(OpenAICompatibleClient):
        def __init__(self):
            super().__init__(
                base_url=base_url,
                model=model,
                temperature=0,
                max_tokens=2048,
                max_retries=1,
                timeout=180,
                seed=7,
            )
            self.messages = []
            self.call_fn = self.call

        def call(self, messages, model_name=None, api_key=None, **kwargs):
            self.messages.append(copy.deepcopy(messages))
            return super().call(
                messages, model_name=model_name, api_key=api_key, **kwargs
            )

    old_client, new_client = CapturingClient(), CapturingClient()
    legacy.create_default_llm_client = lambda **kwargs: old_client
    random.seed(128)
    expected = legacy.final_eval_single(
        "Clean the cup and then open it.",
        copy.deepcopy(SCENE),
        reply=REPLY,
        domain_file_path=domain,
        eval_mode=mode,
        exp_model=model,
    )
    random.seed(128)
    actual = public.final_eval_single(
        "Clean the cup and then open it.",
        copy.deepcopy(SCENE),
        reply=REPLY,
        domain_file_path=domain,
        eval_mode=mode,
        exp_model=model,
        llm_client=new_client,
        solver_path=planner,
    )
    report = dict(
        mode=mode,
        model=model,
        base_url=base_url,
        temperature=0,
        max_tokens=2048,
        seed=7,
        original=expected,
        migrated=actual,
        messages_equal=old_client.messages == new_client.messages,
        results_equal=expected == actual,
    )
    report_path = os.environ.get("TGPO_LIVE_LLM_REPORT")
    if report_path:
        destination = Path(report_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        previous = json.loads(destination.read_text()) if destination.exists() else []
        destination.write_text(
            json.dumps(previous + [report], ensure_ascii=False, indent=2) + "\n"
        )
    assert expected["llm_evaluated"] and actual["llm_evaluated"]
    assert not expected.get("llm_error") and not actual.get("llm_error")
    assert old_client.messages == new_client.messages
    assert actual == expected
