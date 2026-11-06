"""Reusable task-data transforms extracted from fix_plan_pddl/process_data.

Historical filesystem orchestration and private source extraction are excluded.
"""

import ast
import re
from typing import Set
from robot_planning.pddl.parser import parse_tokens


def _format_expression(value):
    return "(" + " ".join(_format_expression(item) for item in value) + ")" if isinstance(value, list) else value


def extract_pddl_goals(text):
    """Extract goal expressions using the shared PDDL S-expression parser."""
    match = re.search(r"\(:goal\s*(.*)\s*\)", text, re.DOTALL)
    if not match or not match.group(1).strip():
        return []
    value = parse_tokens(match.group(1).replace("(", " ( ").replace(")", " ) ").split())
    if isinstance(value, list) and value and value[0].lower() == "and":
        return [_format_expression(item) for item in value[1:] if isinstance(item, list)]
    return [_format_expression(value)]


extract_goal_from_problem_file = extract_pddl_goals


def compare_plan(plan_1, plan_2):
    """Historical loose plan comparison, intended only for data checks.

    The two placement action pairs are treated as interchangeable even when
    parameters differ. This is intentionally separate from strict evaluation.
    """
    if len(plan_1) != len(plan_2):
        return False
    equivalent = ({"place_on_surface", "place_next_to_surface"},
                  {"place_in_receptacle", "place_next_to_inside"})
    for first, second in zip(plan_1, plan_2):
        if first == second:
            continue
        first_name = re.findall(r"\b[a-zA-Z0-9_]+\b", first)[0]
        second_name = re.findall(r"\b[a-zA-Z0-9_]+\b", second)[0]
        if first_name != second_name and any(first_name in pair and second_name in pair for pair in equivalent):
            continue
        return False
    return True


def build_effect_reply(subtask_list, goal_list, related_obj_list):
    """Preserve the original single-stage subgoal data format."""
    if not (len(subtask_list) == len(goal_list) == len(related_obj_list)):
        raise ValueError("Subtasks, goals, and related objects must align")
    text = "### Task Decomposition (Tree-Structured CoT)\n"
    for index, (task, goals, objects) in enumerate(zip(subtask_list, goal_list, related_obj_list), 1):
        text += f"{index}.{task}\n   subtask goal: {goals}\n   implicitly related object: {objects}\n"
    return text


def parse_effect_reply(text):
    goals = re.findall(r"subtask goal: (.*?)\s*implicitly related object:", text)
    objects = re.findall(r"implicitly related object: (.*?)$", text, re.MULTILINE)
    return [ast.literal_eval(value) for value in goals], [ast.literal_eval(value) for value in objects]


def merge_effect_list(effect_list, history_effect_list):
    """Remove effects already present in earlier subgoals without reordering."""
    return [effect for effect in effect_list if not any(effect in history for history in history_effect_list)]


def filter_goal_rule_base(goals):
    """Keep positive and unary negative goals, preserving the historical rule.

    The old comment called unary goals spatial predicates, but the executed
    condition counts parameters rather than inspecting predicate semantics.
    """
    return [goal for goal in goals if not goal.startswith("(not ") or len(goal[6:-1].split(" ")) - 1 == 1]


def get_important_related_obj(goal, action_list, problem_generator):
    """Use the shared problem generator to find implicitly required objects."""
    goal_objects = problem_generator.get_related_obj(goal)
    return [obj for obj in get_related_obj_action(action_list) if obj not in goal_objects]

def merge_effect(predicate_list):
    """Cancel opposite effects; preserve the source set iteration order."""
    unmatched_predicates = set()
    for predicate in predicate_list:
        predicate = predicate.strip()
        is_negative = predicate.startswith('(not ') and predicate.endswith('))')
        opposite = ''
        if is_negative:
            opposite = predicate[5:-1]
        else:
            opposite = f'(not {predicate})'
        if opposite in unmatched_predicates:
            unmatched_predicates.remove(opposite)
        else:
            unmatched_predicates.add(predicate)
    return list(unmatched_predicates)

def replace_action_list(action_block_list, raw_plan):
    """Replace matching action names sequentially with the grounded plan."""
    index = 0
    for i, block in enumerate(action_block_list):
        for j, action in enumerate(block):
            action_name = re.findall('\\b[a-zA-Z0-9_]+\\b', action)[0]
            raw_plan_action_name = re.findall('\\b[a-zA-Z0-9_]+\\b', raw_plan[index])[0]
            if action_name != raw_plan_action_name:
                print(f'### action name not pair, action index: {index}, action_name: {action_name}, raw_action_name: {raw_plan_action_name}')
            else:
                action_block_list[i][j] = raw_plan[index]
            index += 1
    return action_block_list

def get_related_obj_action(action_list):
    """Collect non-navigation action parameters."""
    collected_params: Set[str] = set()
    for action_str in action_list:
        cleaned_str = action_str.strip().lstrip('(').rstrip(')')
        parts = cleaned_str.split()
        if not parts:
            continue
        action_name = parts[0]
        parameters = parts[1:]
        if action_name != 'navigate':
            collected_params.update(parameters)
    return list(collected_params)

def get_related_obj_predicate(action_list):
    """Collect parameters of positive and negated predicates."""
    collected_params: Set[str] = set()
    for action_str in action_list:
        cleaned_str = action_str.strip().lstrip('(').rstrip(')')
        if cleaned_str[:3] == 'not':
            cleaned_str = cleaned_str[4:].lstrip('(').rstrip(')')
        parts = cleaned_str.split()
        if not parts:
            continue
        parameters = parts[1:]
        collected_params.update(parameters)
    return list(collected_params)

def get_related_obj(effects, action_list):
    """Find action parameters absent from the resulting effects."""
    action_related_obj = get_related_obj_action(action_list)
    effect_related_obj = get_related_obj_predicate(effects)
    related_obj_list = []
    for obj in action_related_obj:
        if obj not in effect_related_obj:
            related_obj_list.append(obj)
    return related_obj_list

def remove_not_effect(effect_list):
    """Retain only positive effects."""
    filtered_effect_list = []
    for effect in effect_list:
        if not effect.startswith('(not '):
            filtered_effect_list.append(effect)
    return filtered_effect_list

def extract_task_details(text_blob: str):
    """Read effect/related-object list annotations from tree text."""
    all_effects = []
    all_related_objs = []
    lines = text_blob.strip().split('\n')
    for line in lines:
        cleaned_line = line.strip()
        if cleaned_line.startswith('**effect**:') or cleaned_line.startswith('effect:'):
            try:
                list_str_start = line.find('[')
                if list_str_start != -1:
                    list_str = line[list_str_start:]
                    effect_list = ast.literal_eval(list_str)
                    all_effects.append(effect_list)
            except (ValueError, SyntaxError) as e:
                print(f'Could not parse effect line: {line}\nError: {e}')
        elif cleaned_line.startswith('**related_obj**:') or cleaned_line.startswith('related_obj:'):
            try:
                list_str_start = line.find('[')
                if list_str_start != -1:
                    list_str = line[list_str_start:]
                    related_obj_list = ast.literal_eval(list_str)
                    all_related_objs.append(related_obj_list)
            except (ValueError, SyntaxError) as e:
                print(f'Could not parse related_obj line: {line}\nError: {e}')
    return (all_effects, all_related_objs)
