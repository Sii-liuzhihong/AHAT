"""
AHAT: AI-powered Hierarchical Action Task Planner
=================================================

AHAT is an intelligent planning system that combines task decomposition
with PDDL-based planning to solve complex robot manipulation tasks.

Main Components
---------------
1. **Task Decomposer**: Uses LLM to decompose complex tasks into subtasks.
2. **Solvability Checker**: Validates subgoal reachability using PDDL solvers.
3. **Solve Coordinator**: Orchestrates plan generation from LLM outputs.
4. **AHAT Pipeline**: End-to-end planning pipeline integrating all components.

Quick Start
-----------
>>> from ahat.planning.pipeline import AHATPipeline
>>> pipeline = AHATPipeline(
...     data_path="examples/example_datas.json",
...     domain_file_path="examples/ahat_domain.pddl",
...     output_dir="outputs"
... )
>>> results = pipeline.run(max_items=1)
"""

__version__ = "1.0.0"

# Main public API - classes
from ahat.planning.pipeline import AHATPipeline
from ahat.planning.solve import SolveCoordinator
from ahat.planning.decompose import TaskDecomposer
from ahat.evaluation.solvability import PDDLSolvabilityChecker

# PDDL components
from ahat.pddl.planner import PDDLPlanner
from ahat.pddl.problem_generator import PDDLProblemGenerator
from ahat.scene_graph import SceneGraphManager

# Data schemas
from ahat.schemas.entities import AHATData

# Type definitions
from ahat.schemas.typings import (
    Paths,
    Predicates,
    Operators,
    Domain,
    Domains,
    Objects,
    Init,
    Goal,
    Problem,
    Problems,
    PDDLPlannerFile,
    PDDLPlannerFiles,
    BatchInputs,
    BatchCheckList,
)


__all__ = [
    # Main classes
    "AHATPipeline",
    "SolveCoordinator",
    "TaskDecomposer",
    "PDDLSolvabilityChecker",
    
    # PDDL classes
    "PDDLPlanner",
    "PDDLProblemGenerator",
    "SceneGraphManager",
    
    # Data schemes
    "AHATData",
    
    # Type definitions
    "Paths",
    "Predicates",
    "Operators",
    "Domain",
    "Domains",
    "Objects",
    "Init",
    "Goal",
    "Problem",
    "Problems",
    "PDDLPlannerFile",
    "PDDLPlannerFiles",
    "BatchInputs",
    "BatchCheckList",
]
