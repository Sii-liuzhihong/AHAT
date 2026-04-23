"""
Global Project Constants
========================

This module defines project-wide constants using namespace classes.
It centralizes filenames, directory structures, data keys, environment
variables, and internal logger names to ensure consistency across the
entire codebase.

By using these constants instead of hardcoded strings, the project enforces
a strict "Single Source of Truth" for system protocols and file conventions.

Architecture Position:
    [Configuration / Constants] -> Shared static definitions used by
    all pipelines and infrastructure modules.
"""

from typing import Final

__all__ = ["JSONKey", "EnvVar", "LoggerName"]



class JSONKey:
    """JSON keys for statistics aggregation."""

    # For Stat Summary
    TOTAL_CALLS: Final[str] = "nums_llm_call"
    TOTAL_COSTS: Final[str] = "costs"
    TOTAL_INPUT_TOKENS: Final[str] = "input_tokens"
    TOTAL_OUTPUT_TOKENS: Final[str] = "output_tokens"
    TOTAL_THINKING_TIME: Final[str] = "thinking_time"
    TOTAL_PLANNER_TIME: Final[str] = "pddl_planner_time"




class EnvVar:
    """Keys for Environment Variables loaded from .env or system env."""
    
    # LLM setup
    API_KEY: Final[str] = "API_KEY"
    BASE_URL: Final[str] = "BASE_URL"
    CLIENT_TYPE: Final[str] = "CLIENT_TYPE"
    
    # External Tools
    FAST_DOWNWARD_PATH: Final[str] = "FAST_DOWNWARD_PATH"
    


class LoggerName:
    """Internal logger names used to retrieve specific logging instances."""
    EXECUTION: Final[str] = "execution"
    LLM_USAGE: Final[str] = "llm_usage"


AHAT_MODEL_DEFAULT_PATH = "models/AHAT-TGPO"
AHAT_MODEL_REPO_ID = "Sii-liuzhihong/AHAT-TGPO"


AHAT_DATASET_REPO_ID = "SII-liyang2024/AHAT-dataset"
AHAT_DATASET_DEFAULT_PATH = "data/AHAT-dataset"
AHAT_DATASET_DEFAULT_FILE = "data/AHAT-dataset/ahat_eval_set.jsonl"

# model deployment
DEPLOYMENT_DEFAULT_PORT = 8000
DEPLOYMENT_DEFAULT_HOST = "0.0.0.0"