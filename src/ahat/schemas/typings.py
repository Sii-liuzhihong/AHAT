"""
Type Definitions
================

This module contains common type aliases used throughout the codebase for static
type checking. It centralizes definitions for PDDL structures, file paths,
and batch processing containers to maintain consistency across modules.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# ==========================================
# File System & Paths
# ==========================================
Paths = Union[str, Path]


# ==========================================
# PDDL Domain Structures
# ==========================================

# Key: PDDL format predicate string (e.g., "(at ?x ?y)")
# Value: Natural language explanation or comment
Predicates = Dict[str, str]

# Key: Action name (e.g., "pick-up")
# Value: Full PDDL action string block
Operators = Dict[str, str]

# A Domain is represented as a tuple of its components: (Predicates, Operators)
Domain = Tuple[Predicates, Operators]
Domains = List[Domain]


# ==========================================
# PDDL Problem Structures
# ==========================================

# List of object definitions (e.g., "block_a")
Objects = List[str]

# List of grounded predicates representing the initial state (e.g., "(on block_a table)")
Init = List[str]

# The raw PDDL goal string (e.g., "(:goal (and ...))")
Goal = str

# A Problem is a tuple: (Objects, Init, Goal)
Problem = Tuple[Objects, Init, Goal]
Problems = List[Problem]


# ==========================================
# Planner Interfaces
# ==========================================

# A tuple representing paths for one planning task:
# (domain_file_path, problem_file_path, output_plan_path)
PDDLPlannerFile = Tuple[Paths, Paths, Paths]
PDDLPlannerFiles = List[PDDLPlannerFile]


# ==========================================
# Batch Processing & Configs
# ==========================================

# Structure for batch task inputs (used in atomic_domain and task_planner).
# Format:
# {
#     "task_name_1": {
#         "instruction": "...",
#         "path": Path("/abs/path/to/data")  # Resolved absolute path
#     },
#     ...
# }
BatchInputs = Dict[str, Dict[str, Paths]]

# Status checklist for batch execution results.
# Key: Task name
# Value:
#   - True: Success
#   - False: Failure (Logic failure, e.g., planning failed)
#   - None: Error (System exception, e.g., network timeout)
BatchCheckList = Dict[str, Optional[bool]]

