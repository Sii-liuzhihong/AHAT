"""Small reusable data utilities retained from the approved e2e data directory.

These operate on public task fields; raw private experiment extraction is excluded.
They do not add an e2e evaluation or reward implementation.
"""

from copy import deepcopy
import json
import re


def normalize_legacy_rooms(scene_graph):
    result = deepcopy(scene_graph)
    if result.get("rooms") and isinstance(result["rooms"][0], str):
        result["rooms"] = [{"name": room, "states": {"is_clean": False, "require_floor_cleaner": True}}
                           for room in result["rooms"]]
    return result


def convert_to_function_style(action):
    match = re.match(r"^\(([\w-]+)\s+([\s\S]*)\)$", action)
    if not match:
        return action
    return f"{match.group(1)}({', '.join(match.group(2).split())})"


def build_e2e_training_record(record, *, split="train", index=0, data_source="robot_planning"):
    """Retain approved e2e *data formatting* for already extracted public fields.

    This does not attach an e2e reward or evaluation. The caller supplies the
    matching task prompt/domain; private llm_response extraction is omitted.
    """
    scene = record["scene_graph"]
    if not isinstance(scene, str):
        scene = json.dumps(normalize_legacy_rooms(scene))
    instruction = record["instruction"]
    user = (f"{record['system_prompt']}\n<Instruction>{instruction}</Instruction>\n"
            f"<SceneGraph>{scene}</SceneGraph>")
    return {"data_source": data_source,
            "prompt": [{"role": "system", "content": "You are a helpful assistant."},
                       {"role": "user", "content": user}],
            "ability": "decision_making",
            "reward_model": {"style": "rule", "ground_truth": f"<think>{record['tree_structure_cot']}</think>\n<plan>{record['plan']}</plan>"},
            "extra_info": {"split": split, "index": index, "domain_file": record["domain_file"],
                           "scene_graph": scene, "instruction": instruction, "plan": record["plan"]}}


def build_two_stage_reply(subtask_list, goal_list, related_obj_list):
    if not (len(subtask_list) == len(goal_list) == len(related_obj_list)):
        raise ValueError("Subtasks, goals, and related objects must align")
    text = "### Task Decomposition ###\n"
    for index, task in enumerate(subtask_list, 1):
        text += f"   {index}. {task}\n"
    text += "\n### Subgoal Grounding ###\n"
    for index, (task, goals, objects) in enumerate(zip(subtask_list, goal_list, related_obj_list), 1):
        text += f"   {index}. {task}\n      Subtask goal: {goals}\n      Implicitly related object: {objects}\n"
    return text
