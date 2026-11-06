"""The public boundary starts at tasks, after private source extraction."""

import json
from pathlib import Path


def validate_task(record, *, supervised=False):
    if not isinstance(record.get("instruction"), str) or not record["instruction"].strip():
        raise ValueError("Task requires a nonempty instruction string")
    scene = record.get("scene_graph")
    if isinstance(scene, str):
        scene = json.loads(scene)
    if not isinstance(scene, dict):
        raise ValueError("scene_graph must be an object or its JSON representation")
    if supervised:
        reference_reply(record)
    return scene


def reference_reply(record):
    reply = record.get("output", record.get("reply"))
    if reply is None and record.get("reply_list"):
        reply = record["reply_list"][0]
    if not isinstance(reply, str) or not reply.strip():
        raise ValueError("SFT task requires output, reply, or a nonempty reply_list")
    return reply


def resolve_domain(record, *, data_path=None, domain=None):
    value = domain or record.get("domain_file_path")
    if not value:
        raise ValueError("Provide domain_file_path or --domain")
    path = Path(value).expanduser()
    if not path.is_absolute() and data_path is not None:
        path = Path(data_path).resolve().parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"PDDL domain does not exist: {path}")
    return str(path)
