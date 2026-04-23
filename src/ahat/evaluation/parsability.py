"""
Parsability Checker
-------------------
Parses a model-generated CoT string and checks whether it contains a
well-formed list of subgoal predicates and related objects.

Expected CoT format::

    Subtask goal: [['(item_on_surface apple_1 table_1)'], ...]
    Implicitly related object: [['table_1'], ...]
"""

import ast
import re
from typing import List, Tuple


def parse_subgoal_and_related_obj_from_cot(
    effect_cot: str,
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Extract subgoal lists and related-object lists from a CoT string.

    Args:
        effect_cot: Raw Chain-of-Thought string produced by the model.

    Returns:
        ``(subgoal_list, related_objects_list)`` where each inner list
        corresponds to one subtask step.

    Raises:
        TypeError: If the parsed values are not lists.
    """
    subtask_goals = re.findall(
        r"Subtask goal: (.*?)\s*Implicitly related objects?:", effect_cot
    )
    related_objects = re.findall(
        r"Implicitly related objects?: (.*?)$", effect_cot, re.MULTILINE
    )

    def safe_eval(s: str) -> List[str]:
        try:
            return ast.literal_eval(s)
        except Exception:
            s = s.strip()
            if s.startswith("[") and s.endswith("]"):
                s = s[1:-1].strip()
                if not s:
                    return []
                items = []
                for item in s.split(","):
                    item = item.strip()
                    if item:
                        # Remove existing surrounding quotes if any
                        item = re.sub(r"^['\"]+(.*)['\"]+$", r"\1", item)
                        items.append(item)
                return items
            return []

    subgoal_list = [safe_eval(g) for g in subtask_goals]
    related_objects_list = [safe_eval(o) for o in related_objects]

    if not isinstance(subgoal_list, list):
        raise TypeError(f"Parsed subgoal_list is not a list: {subgoal_list}")
    if not isinstance(related_objects_list, list):
        raise TypeError(f"Parsed related_objects_list is not a list: {related_objects_list}")

    return subgoal_list, related_objects_list


def check_parsability(
    tree_structure_subgoal: str,
) -> Tuple[bool, List[List[str]], List[List[str]]]:
    """
    Check whether a CoT string can be parsed into valid subgoal / related-
    object lists.

    Validity rules:
        - Both lists must be non-empty.
        - Every subgoal must be a ``list``.
        - Every related-object entry must be a ``list``.

    Args:
        tree_structure_subgoal: Raw CoT string from the model.

    Returns:
        ``(parsable, subgoal_list, related_obj_list)`` – *parsable* is
        ``True`` only when all validity rules pass.  On failure the lists
        are empty.
    """
    try:
        subgoal_list, related_obj_list = parse_subgoal_and_related_obj_from_cot(
            tree_structure_subgoal
        )

        if not subgoal_list or not related_obj_list:
            return False, [], []

        for subgoal in subgoal_list:
            if not isinstance(subgoal, list):
                raise TypeError(f"Subgoal is not a list: {subgoal}")

        for related_obj in related_obj_list:
            if not isinstance(related_obj, list):
                raise TypeError(f"Related object entry is not a list: {related_obj}")

        return True, subgoal_list, related_obj_list

    except Exception:
        return False, [], []
