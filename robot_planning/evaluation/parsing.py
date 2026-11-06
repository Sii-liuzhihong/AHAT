"""Parse subtask goals, related objects, tagged reviews and action notation."""

import ast
import re


def _parse_related_object_list(serialized_objects):
    serialized_objects = serialized_objects.strip()
    try:
        parsed_objects = ast.literal_eval(serialized_objects)
    except (SyntaxError, ValueError):
        if not (
            serialized_objects.startswith("[") and serialized_objects.endswith("]")
        ):
            raise
        body = serialized_objects[1:-1].strip()
        if not body:
            return []
        parsed_objects = []
        for item in body.split(","):
            item = item.strip()
            if not item:
                raise ValueError("related object list contains an empty item")
            if item[0] in {"'", '"'}:
                quoted_item = ast.literal_eval(item)
                if not isinstance(quoted_item, str):
                    raise TypeError("related object must be a string")
                item = quoted_item
            elif any((character in item for character in "[]")):
                raise ValueError(f"invalid related object: {item!r}")
            if not item.strip():
                raise ValueError("related object must not be empty")
            parsed_objects.append(item)
    if not isinstance(parsed_objects, list):
        raise TypeError(f"{parsed_objects!r} is not a list")
    if not all((isinstance(item, str) for item in parsed_objects)):
        raise TypeError("all related objects must be strings")
    return parsed_objects


def parse_effect_and_related_obj_from_cot(effect_cot):
    subtask_goals = re.findall(
        "Subtask goal: (.*?)\\s*Implicitly related object:", effect_cot
    )
    related_objects = re.findall(
        "Implicitly related object: (.*?)$", effect_cot, re.MULTILINE
    )
    subtask_goals_list = [ast.literal_eval(goal) for goal in subtask_goals]
    related_objects_list = [_parse_related_object_list(obj) for obj in related_objects]
    if type(subtask_goals_list) is not list:
        raise TypeError(f"{subtask_goals_list} is not list")
    if type(related_objects_list) is not list:
        raise TypeError(f"{related_objects_list} is not list")
    return (subtask_goals_list, related_objects_list)


def check_parsability(tree_structure_subgoal):
    parsable = False
    try:
        subgoal_list, related_obj_list = parse_effect_and_related_obj_from_cot(
            tree_structure_subgoal
        )
        format_right = True
        if len(subgoal_list) > 0:
            for subgoal in subgoal_list:
                if type(subgoal) is not list:
                    format_right = False
                    raise TypeError(f"unparsable!: subgoal format wrong: {subgoal}")
        else:
            format_right = False
        if len(related_obj_list) > 0:
            for related_obj in related_obj_list:
                if type(related_obj) is not list:
                    format_right = False
                    raise TypeError(
                        f"unparsable: related obj format wrong: {related_obj}"
                    )
        else:
            format_right = False
        if format_right:
            parsable = True
    except Exception:
        subgoal_list = []
        related_obj_list = []
    return (parsable, subgoal_list, related_obj_list)


def convert_to_parenthesis_style(function_str):
    pattern = "^([\\w-]+)\\s*\\(([\\s\\S]*)\\)$"
    match = re.match(pattern, function_str)
    if not match:
        return function_str
    action = match.group(1)
    params = []
    current_param = []
    in_quote = None
    escape_next = False
    for char in match.group(2):
        if escape_next:
            current_param.append(char)
            escape_next = False
            continue
        if char == "\\":
            escape_next = True
            continue
        if char in ['"', "'"] and (not escape_next):
            if not in_quote:
                in_quote = char
            elif char == in_quote:
                in_quote = None
            current_param.append(char)
            continue
        if char == "," and (not in_quote):
            params.append("".join(current_param).strip())
            current_param = []
            continue
        current_param.append(char)
    if current_param:
        params.append("".join(current_param).strip())
    return f"({action} {' '.join(params)})"


def get_content(text, tag):
    pattern = f"<{tag}>(.*?)</{tag}>"
    try:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            content = match.group(1)
            return content
        else:
            return None
    except Exception:
        return None


def extract_related_obj_list_from_reply_or_pair(
    reply=None, subgoal_related_obj_list_pair=None
):
    """Extract review context with the reference reply/pair conventions.

    Reply predicates yield argument names; structured pairs historically pass
    their string elements through verbatim. Keep this distinction for parity.
    """
    objects = []
    if subgoal_related_obj_list_pair is not None:
        pair = subgoal_related_obj_list_pair
        if isinstance(pair, list) and len(pair) >= 2:
            for groups in pair[:2]:
                if isinstance(groups, list):
                    for group in groups:
                        for item in group if isinstance(group, list) else [group]:
                            if isinstance(item, str) and item.strip():
                                objects.append(item.strip())
    elif reply is not None:
        try:
            parsable, subgoals, related = check_parsability(reply)
            if parsable:
                for group in subgoals:
                    for predicate in group:
                        objects.extend(predicate[1:-1].split(" ")[1:])
                for group in related:
                    for item in group:
                        if isinstance(item, str) and item.strip():
                            objects.append(item.strip())
        except Exception:
            pass
    return list(dict.fromkeys(objects))
