"""Task repair independent of tensors, Ray, and the training framework."""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from robot_planning.clients.llm import create_default_llm_client
from robot_planning.prompts import load_prompt

DEFAULT_PROMPT = "subgoal_fix_cot_0102_prompt"


def extract_task_decomposition(text, data_type="subgoal"):
    start_marker, end_marker = "### Task Decomposition ###", "### Subgoal Grounding ###"
    if data_type and "e2e" in data_type:
        start_marker, end_marker = "Task Decomposition", "PDDL Action Mapping"
    lines = text.split("\n")
    start = end = -1
    for index, line in enumerate(lines):
        if start == -1 and start_marker in line:
            start = index
            continue
        if start != -1 and end_marker in line:
            end = index
            break
    if start == -1 or end == -1:
        return ""
    extracted = lines[start + 1:end]
    while extracted and not extracted[-1].strip():
        extracted.pop()
    return "\n".join(extracted)


def parse_fix_response(response):
    match = re.search(r"<fixed_cot>(.*?)</fixed_cot>", response, re.DOTALL)
    if not match or not match.group(1):
        return None
    return match.group(1).strip()


def format_fix_input(instruction, house_content, action_list=None, tree_structure_cot="", evaluate=""):
    if isinstance(house_content, dict):
        from robot_planning.scene_graph.summary import sum_sg_content
        house_content = sum_sg_content(house_content)
    # action_list was unused in the source message. Do not silently append it.
    return (f"**instruction**: {instruction}\n"
            f"**house_content**: {house_content}\n"
            f"**original_cot**: {tree_structure_cot}\n"
            f"**evaluate** {evaluate}\n")


def format_repair_prefix(text, output_format="subgoal", phase="generation"):
    if phase not in ("generation", "response"):
        raise ValueError("phase must be generation or response")
    if output_format not in ("subgoal", "e2e"):
        raise ValueError("output_format must be subgoal or e2e")
    if phase == "generation" and output_format == "subgoal":
        return "### Task Decomposition ###\n" + "\n".join("   " + line for line in text.splitlines())
    # Preserve the historical asymmetry between generation and merged response.
    return "<think>\n### Task Decomposition (Tree-Structured CoT)\n" + text


def fix_cot(instruction, house_content, tree_structure_cot, evaluate, *, client=None,
            prompt=DEFAULT_PROMPT, model_name=None, api_key=None, max_retries=1):
    client = client or create_default_llm_client()
    messages = [{"role": "system", "content": load_prompt(prompt)},
                {"role": "user", "content": format_fix_input(instruction, house_content,
                           tree_structure_cot=tree_structure_cot, evaluate=evaluate)}]
    for _ in range(max_retries):
        try:
            result = client.call(messages, model_name=model_name, api_key=api_key)
            if result.success:
                fixed = parse_fix_response(result.content)
                if fixed:
                    return {"fix_cot": fixed, "response": result.content}
        except Exception:
            # Failed repair keeps the extracted original CoT, as in the source.
            # The HTTP client owns transport errors; one row must not cancel a batch.
            continue
    return {"fix_cot": tree_structure_cot, "response": "no response"}


def repair_batch_records(records, *, client=None, prompt=DEFAULT_PROMPT, model_name=None,
                         output_format="subgoal", max_workers=32, api_key=None,
                         max_retries=1, system_prompt=None, fix_system_prompt=None):
    prompt = fix_system_prompt or system_prompt or prompt
    client = client or create_default_llm_client()
    if not records:
        return []

    def repair(record):
        scene = record.get("scene_graph")
        if isinstance(scene, str):
            scene = json.loads(scene)
        result = fix_cot(record.get("instruction"), scene,
                         extract_task_decomposition(record["reply"], output_format), record["evaluate"],
                         client=client, prompt=prompt, model_name=model_name, api_key=api_key,
                         max_retries=max_retries)
        # The source regenerates from a nonempty original prefix when the LLM
        # repair fails. Mark that fallback explicitly; only empty prefixes fail.
        status = "ok" if result["response"] != "no response" else ("fallback" if result["fix_cot"] else "failed")
        return {**result, "status": status}

    with ThreadPoolExecutor(max_workers=min(max_workers, len(records))) as pool:
        return list(pool.map(repair, records))
