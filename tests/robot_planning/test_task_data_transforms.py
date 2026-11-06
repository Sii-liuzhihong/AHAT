"""Task-tree and effect conversion retained from the useful PDDL utilities."""

import ast
from copy import deepcopy
import os
from pathlib import Path
import re
import typing
import unittest

from robot_planning.data import effects, task_tree


TREE = """### Task Decomposition (Tree-Structured CoT)
1. Clean the table.
   **Subtask 1.1**: Go to table.
   **Subtask 1.2**: Wipe table.
2. Return.
   **Subtask 2.1**: Return to kitchen.
### PDDL Action Mapping
**Subtask 1.1**: navigate(robot, table)
**Subtask 1.2**: wipe(robot, table)
**Subtask 1.3**: dry(robot, table)
**Subtask 2.1**: navigate(robot, kitchen)
"""


def original(relative, names):
    path = Path(os.environ["TGPO_SOURCE_ROOT"]) / relative
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"re": re, "ast": ast, "Dict": typing.Dict, "List": typing.List, "Set": typing.Set,
                 "Any": typing.Any, "ParsedList": typing.Any}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class TaskDataTests(unittest.TestCase):
    def test_tree_grouping_and_effect_reply(self):
        tree = task_tree.parse_task_string_recursively(TREE)
        self.assertEqual(task_tree.collect_actions_by_parent_task(tree),
                         [["navigate(robot, table)", "wipe(robot, table)", "dry(robot, table)"], ["navigate(robot, kitchen)"]])
        annotations = [[["(is_clean table)"], ["table"]], [["(agent_at robot kitchen)"], ["kitchen"]]]
        text = task_tree.build_effect_cot_data(tree, annotations)
        self.assertEqual(effects.extract_task_details(text),
                         ([["(is_clean table)"], ["(agent_at robot kitchen)"]], [["table"], ["kitchen"]]))
        # The source renders grouped action lists as Python list literals, so
        # this older format is not a lossless serialization for overflow IDs.
        rendered = task_tree.convert_dict_to_string(tree)
        self.assertIn("['wipe(robot, table)', 'dry(robot, table)']", rendered)

    def test_effect_rules_and_goal_extraction(self):
        self.assertEqual(set(effects.merge_effect(["(p a)", "(not (p a))", "(q b)"])), {"(q b)"})
        self.assertEqual(set(effects.get_related_obj(["(is_clean table)"], ["(wipe robot table cloth)"])), {"robot", "cloth"})
        self.assertEqual(effects.extract_pddl_goals("(define (problem p) (:goal (and (p a) (not (q b)))))"),
                         ["(p a)", "(not (q b))"])
        with self.assertRaises(ValueError):
            effects.extract_pddl_goals("(define (problem p) (:goal (and (p a)")

    @unittest.skipUnless(os.environ.get("TGPO_SOURCE_ROOT"), "Set TGPO_SOURCE_ROOT for source parity")
    def test_source_transform_parity(self):
        tree_names = ["parse_task_string_recursively", "convert_dict_to_string", "build_effect_cot_data",
                      "collect_actions_by_parent_task", "remove_trailing_whitespace", "remove_blank_lines"]
        effect_names = ["merge_effect", "replace_action_list", "get_related_obj_action", "get_related_obj_predicate",
                        "get_related_obj", "remove_not_effect", "extract_task_details", "extract_goal_from_problem_file"]
        old = original("policy_iteration/iter/fix_plan_pddl/process_data/data_structure.py", tree_names + effect_names)
        tree = task_tree.parse_task_string_recursively(TREE)
        self.assertEqual(tree, old["parse_task_string_recursively"](TREE))
        for name in ("convert_dict_to_string", "collect_actions_by_parent_task"):
            self.assertEqual(getattr(task_tree, name)(tree), old[name](tree))
        for name in ("remove_trailing_whitespace", "remove_blank_lines"):
            self.assertEqual(getattr(task_tree, name)(" a  \n\n b\t \n"), old[name](" a  \n\n b\t \n"))
        annotation = [[["(is_clean table)"], ["table"]], [["(agent_at robot kitchen)"], ["kitchen"]]]
        text = task_tree.build_effect_cot_data(tree, annotation)
        self.assertEqual(text, old["build_effect_cot_data"](tree, annotation))
        self.assertEqual(effects.extract_task_details(text), old["extract_task_details"](text))
        values = {"merge_effect": (["(p a)", "(not (p a))", "(q b)"],),
                  "get_related_obj_action": (["(navigate robot kitchen)", "(wipe robot table cloth)"],),
                  "get_related_obj_predicate": (["(p a)", "(not (q b))"],),
                  "get_related_obj": (["(p table)"], ["(wipe robot table cloth)"]),
                  "remove_not_effect": (["(p a)", "(not (q b))"],),
                  "replace_action_list": ([["wipe(robot, table)"]], ["(wipe robot table)"])}
        for name, args in values.items():
            self.assertEqual(getattr(effects, name)(*deepcopy(args)), old[name](*deepcopy(args)), name)
        other_names = ["tokenize", "parse_tokens", "format_parsed_list", "extract_pddl_goals", "build_effect_cot_data",
                       "parse_effect_and_related_obj_from_cot", "merge_effect_list", "filter_goal_rule_base"]
        other = original("policy_iteration/iter/fix_plan_pddl/process_data/process_data_structrue/transfer_hi_pddl_data.py", other_names)
        for goal in ("(p a)", "(not (p a))", "(and (p a) (not (q b)))"):
            problem = f"(define (problem p) (:goal {goal}))"
            self.assertEqual(effects.extract_pddl_goals(problem), other["extract_pddl_goals"](problem))
            self.assertEqual(effects.extract_goal_from_problem_file(problem), other["extract_pddl_goals"](problem))
        args = (["wipe"], [["(is_clean table)"]], [["table"]])
        reply = effects.build_effect_reply(*args)
        self.assertEqual(reply, other["build_effect_cot_data"](*args))
        self.assertEqual(effects.parse_effect_reply(reply), other["parse_effect_and_related_obj_from_cot"](reply))
        self.assertEqual(effects.merge_effect_list(["p", "q"], [["p"]]), other["merge_effect_list"](["p", "q"], [["p"]]))
        goals = ["(p a)", "(not (p a))", "(not (q a b))"]
        self.assertEqual(effects.filter_goal_rule_base(goals), other["filter_goal_rule_base"](goals))
        comparison = original("policy_iteration/iter/fix_plan_pddl/process_data/data_fix_check.py", ["compare_plan"])
        for first, second in ((["(wipe a)"], ["(wipe a)"]), (["(wipe a)"], ["(wipe b)"]),
                              (["(place_on_surface a b)"], ["(place_next_to_surface a c d)"])):
            self.assertEqual(effects.compare_plan(first, second), comparison["compare_plan"](first, second))


if __name__ == "__main__":
    unittest.main()
