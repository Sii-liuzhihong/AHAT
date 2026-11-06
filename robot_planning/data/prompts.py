"""Shared subgoal input formatting, including the original whitespace protocol."""

from robot_planning.prompts import load_prompt
from pathlib import Path

DEFAULT_PROMPT = "subgoal_system_prompt_with_predicate_0112"
SYSTEM_MESSAGE = "You are a helpful assistant."


def format_task(record, prompt=DEFAULT_PROMPT, *, dynamic_domain=None, data_path=None):
    # The source interpolates the Python dict (not JSON) into the prompt.
    # Changing quotation marks here changes tokenization and is not a cleanup.
    scene = record.get("searched_sg") or record.get("scene_graph", "")
    template = load_prompt(prompt)
    if "{dynamic_domain}" in template:
        content = dynamic_domain if dynamic_domain is not None else record.get("dynamic_domain")
        if content is None and record.get("dynamic_domain_path"):
            path = Path(record["dynamic_domain_path"]).expanduser()
            if not path.is_absolute():
                if data_path is None:
                    raise ValueError("Relative dynamic_domain_path requires the input data_path")
                path = Path(data_path).resolve().parent / path
            content = path.read_text(encoding="utf-8")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Dynamic-domain prompt requires dynamic_domain text or dynamic_domain_path")
        template = template.format(dynamic_domain=content)
    return {"instruction": template,
            "input": f"<Instruction>{record['instruction']}</Instruction>\n<SceneGraph>{scene}</SceneGraph>"}


def build_messages(record, prompt=DEFAULT_PROMPT, *, dynamic_domain=None, data_path=None):
    formatted = format_task(record, prompt, dynamic_domain=dynamic_domain, data_path=data_path)
    # RL data preparation joins instruction/input with TWO newlines and strips.
    user = "\n\n".join(value for value in formatted.values() if value).strip()
    return [{"role": "system", "content": SYSTEM_MESSAGE}, {"role": "user", "content": user}]


def build_sft_input(record, prompt=DEFAULT_PROMPT, *, dynamic_domain=None, data_path=None):
    formatted = format_task(record, prompt, dynamic_domain=dynamic_domain, data_path=data_path)
    # The historical SFT wrapper deliberately used ONE newline and no strip.
    return formatted["instruction"] + "\n" + formatted["input"]


def build_inference_messages(record, prompt=DEFAULT_PROMPT, *, dynamic_domain=None, data_path=None):
    formatted = format_task(record, prompt, dynamic_domain=dynamic_domain, data_path=data_path)
    # eval_subgoal_model -> prompt_utils.apply_chat_template differs from RL:
    # no final period in the system message and no outer whitespace stripping.
    user = "\n\n".join(value for value in formatted.values() if value)
    return [{"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": user}]
