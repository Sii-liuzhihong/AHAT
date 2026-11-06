"""Reusable task-data transforms extracted from fix_plan_pddl/process_data.

Historical filesystem orchestration and private source extraction are excluded.
"""

import re
from typing import Any, Dict, List

def parse_task_string_recursively(raw_string: str) -> dict:
    """Parse nested task IDs; extra action IDs attach to the last valid sibling."""
    if '### PDDL Action Mapping' not in raw_string:
        raise ValueError("输入字符串中未找到 '### PDDL Action Mapping' 分隔符")
    parts = raw_string.split('### PDDL Action Mapping')
    task_decomposition_str = parts[0]
    pddl_action_mapping_str = parts[1]
    action_map = {}
    action_pattern = re.compile('\\*\\*Subtask\\s+([\\d\\.]+)\\*\\*:\\s*(.*)')
    for line in pddl_action_mapping_str.splitlines():
        match = action_pattern.search(line)
        if match:
            subtask_id = match.group(1).strip()
            action_description = match.group(2).strip()
            action_map[subtask_id] = action_description
    nodes = {}
    task_pattern = re.compile('^(?P<main_task_id>\\d+)\\.\\s+(?P<main_task_desc>.*)|\\*\\*(?P<prefix>\\w+)\\s+(?P<sub_task_id>[\\d\\.]+)\\*\\*:\\s+(?P<sub_task_desc>.*)')
    for line in task_decomposition_str.splitlines():
        line = line.strip()
        if not line:
            continue
        match = task_pattern.match(line)
        if match:
            if match.group('main_task_id'):
                task_id = match.group('main_task_id')
                task_desc = match.group('main_task_desc')
            else:
                task_id = match.group('sub_task_id')
                task_desc = match.group('sub_task_desc')
            nodes[task_id] = {'task': task_desc.strip()}
    valid_task_ids = set(nodes.keys())
    # Historical grounding can emit more action IDs than task IDs. Route those
    # actions to the largest valid sibling before constructing the task tree.
    sorted_valid_ids_desc = sorted(list(valid_task_ids), key=lambda x: [int(i) for i in x.split('.')], reverse=True)
    for action_id, action_desc in action_map.items():
        target_id = None
        if action_id in valid_task_ids:
            target_id = action_id
        elif '.' in action_id:
            parent_prefix = '.'.join(action_id.split('.')[:-1]) + '.'
            for valid_id in sorted_valid_ids_desc:
                if valid_id.startswith(parent_prefix):
                    target_id = valid_id
                    break
        if target_id:
            if 'action' not in nodes[target_id]:
                nodes[target_id]['action'] = action_desc
            else:
                current_action = nodes[target_id]['action']
                if isinstance(current_action, list):
                    current_action.append(action_desc)
                else:
                    nodes[target_id]['action'] = [current_action, action_desc]
    result_tree = {}
    for node_id, node_data in sorted(nodes.items(), key=lambda x: len(x[0].split('.'))):
        node_data.setdefault('Subtasks', {})
        if '.' not in node_id:
            result_tree[int(node_id)] = node_data
        else:
            parts = node_id.split('.')
            parent_id = '.'.join(parts[:-1])
            child_key = int(parts[-1])
            parent_node = result_tree
            for part in parent_id.split('.'):
                parent_node = parent_node[int(part)]
                if 'Subtasks' in parent_node:
                    parent_node = parent_node['Subtasks']
            parent_node[child_key] = node_data

    def cleanup_tree(node: Dict[str, Any]):
        if node.get('Subtasks'):
            if 'action' in node:
                del node['action']
            for child in list(node['Subtasks'].values()):
                cleanup_tree(child)
            if not node['Subtasks']:
                del node['Subtasks']
        elif 'action' in node and node['action'] is None:
            del node['action']
    for node in result_tree.values():
        cleanup_tree(node)
    return result_tree

def convert_dict_to_string(data_dict: dict) -> str:
    """Serialize a task tree and action mapping in numeric task-ID order."""
    task_lines = []
    action_lines = []
    prefixes = ['Subtask', 'Subtask', 'Subtask', 'Subtask']

    def get_prefix(level):
        if level - 1 < len(prefixes):
            return prefixes[level - 1]
        return 'Sub' * level + 'task'

    def build_strings_recursive(sub_dict, parent_id, level):
        for key, value in sorted(sub_dict.items()):
            current_id = f'{parent_id}.{key}'
            indent = '   ' * level
            prefix = get_prefix(level)
            task_desc = value['task']
            task_lines.append(f'{indent}**{prefix} {current_id}**: {task_desc}')
            if 'Subtasks' in value and value['Subtasks']:
                build_strings_recursive(value['Subtasks'], current_id, level + 1)
            elif 'action' in value:
                action_desc = value['action']
                action_lines.append(f'**Subtask {current_id}**: {action_desc}')
    for key, value in sorted(data_dict.items()):
        task_lines.append(f"{key}. {value['task']}")
        if 'Subtasks' in value and value['Subtasks']:
            build_strings_recursive(value['Subtasks'], str(key), 1)
    output_parts = []
    output_parts.append('### Task Decomposition (Tree-Structured CoT)')
    output_parts.append('\n'.join(task_lines))
    output_parts.append('\n### PDDL Action Mapping')

    def get_sort_key(line):
        match = re.search('Subtask\\s+([\\d\\.]+)', line)
        if not match:
            return []
        id_parts = match.group(1).split('.')
        return [int(part) for part in id_parts]
    output_parts.append('\n'.join(sorted(action_lines, key=get_sort_key)))
    return '\n'.join(output_parts)

def build_effect_cot_data(data_dict: dict, effects_and_objects: list[list[str]]) -> str:
    """Replace the lowest action layer with parent-level effects and related objects."""
    task_lines = []
    effect_iterator = iter(effects_and_objects)
    prefixes = ['Subtask', 'Subtask', 'Subtask', 'Subtask']

    def get_prefix(level):
        if level - 1 < len(prefixes):
            return prefixes[level - 1]
        return 'Sub' * level + 'task'

    def build_strings_recursive(sub_dict, parent_id, level):
        for key, value in sorted(sub_dict.items()):
            current_id = f'{parent_id}.{key}'
            indent = '    ' * level
            prefix = get_prefix(level)
            task_desc = value['task']
            task_lines.append(f'{indent}**{prefix} {current_id}**: {task_desc}')
            children = value.get('Subtasks', {})
            if children:
                first_child = next(iter(children.values()))
                if 'action' in first_child:
                    try:
                        effect, related_obj = next(effect_iterator)
                        effect_indent = '    ' * (level + 1)
                        task_lines.append(f'{effect_indent}**effect**: {effect}')
                        task_lines.append(f'{effect_indent}**related_obj**: {related_obj}')
                    except StopIteration:
                        print(f'警告: 任务 {current_id} 期望一个 effect，但 effect 列表已用尽。')
                else:
                    build_strings_recursive(children, current_id, level + 1)
    for key, value in sorted(data_dict.items()):
        task_lines.append(f"{key}. {value['task']}")
        children = value.get('Subtasks', {})
        if children:
            first_child = next(iter(children.values()))
            if 'action' in first_child:
                try:
                    effect, related_obj = next(effect_iterator)
                    effect_indent = '    ' * 1
                    task_lines.append(f'{effect_indent}**effect**: {effect}')
                    task_lines.append(f'{effect_indent}**related_obj**: {related_obj}')
                except StopIteration:
                    print(f'警告: 根任务 {key} 期望一个 effect，但 effect 列表已用尽。')
            else:
                build_strings_recursive(children, str(key), 1)
    output_parts = []
    output_parts.append('### Task Decomposition (Tree-Structured CoT)')
    output_parts.append('\n'.join(task_lines))
    return '\n\n'.join(output_parts)

def remove_trailing_whitespace(text: str) -> str:
    """Remove trailing whitespace from each line."""
    cleaned_lines = [line.rstrip() for line in text.splitlines()]
    return '\n'.join(cleaned_lines)

def remove_blank_lines(text: str) -> str:
    """Drop blank lines while keeping other whitespace."""
    lines = text.splitlines()
    non_blank_lines = [line for line in lines if line.strip()]
    return '\n'.join(non_blank_lines)

def collect_actions_by_parent_task(task_tree: Dict[int, Any]) -> List[List[str]]:
    """Collect leaf actions under each direct parent in depth-first order."""
    final_action_groups: List[List[str]] = []

    def _traverse_and_group(node: Dict[str, Any]):
        if 'Subtasks' not in node or not node['Subtasks']:
            if 'action' in node:
                action = node['action']
                if isinstance(action, str):
                    final_action_groups.append([action])
                elif isinstance(action, list):
                    final_action_groups.append(action)
            return
        subtasks = node['Subtasks']
        is_parent_of_leaves = True
        for child in subtasks.values():
            if 'Subtasks' in child and child['Subtasks']:
                is_parent_of_leaves = False
                break
        if is_parent_of_leaves:
            current_group_actions: List[str] = []
            for key in sorted(subtasks.keys()):
                child = subtasks[key]
                if 'action' in child:
                    action = child['action']
                    if isinstance(action, str):
                        current_group_actions.append(action)
                    elif isinstance(action, list):
                        current_group_actions.extend(action)
            if current_group_actions:
                final_action_groups.append(current_group_actions)
        else:
            for key in sorted(subtasks.keys()):
                _traverse_and_group(subtasks[key])
    for key in sorted(task_tree.keys()):
        _traverse_and_group(task_tree[key])
    return final_action_groups
