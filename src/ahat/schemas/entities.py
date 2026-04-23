"""
Entities
===============

This module contains dataclasses and entities used throughout the AHAT system.
"""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class AHATData:
    """
    Represents a single AHAT task input with instruction and environment context.
    
    Attributes:
        instruction (str): Natural language task description 
            (e.g., "Move the coffee cup from the kitchen to the living room")
        scene_graph (dict): Current environment state represented as a scene graph
        id (str): Unique identifier for this task instance
        
    """
    instruction: str
    scene_graph: Dict[str, Any]
    id: str
    
    def __repr__(self) -> str:
        """String representation showing task id and truncated instruction."""
        return f"AHATData(id={self.id!r}, instruction={self.instruction[:50]!r}...)"


