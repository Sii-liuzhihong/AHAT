"""
PDDL Problem Generator
----------------------
Generates PDDL ``:problem`` files from a live scene graph.

Canonical shared PDDL infrastructure for the AHAT project.

Two generation modes are provided:

* ``generate_problem_from_actions`` – derives the context objects from an
  action list; produces an empty goal section (used when the goal is managed
  externally, e.g. through subgoal solvability testing).

* ``generate_problem_file`` – derives context objects from an explicit goal
  predicate list and a seed set of related objects; the goal section is
  populated with those predicates.
"""

import copy
import random
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ahat.scene_graph import SceneGraphManager


class PDDLProblemGenerator:
    """
    Generates PDDL problem files from a :class:`SceneGraphManager` instance.

    Args:
        scene_graph_manager: An initialised scene-graph manager.
    """

    def __init__(self, scene_graph_manager: SceneGraphManager):
        self.sgm = scene_graph_manager

    # ------------------------------------------------------------------
    # Shared predicates / helpers (delegate to SceneGraphManager)
    # ------------------------------------------------------------------

    @staticmethod
    def parse_parentheses(s: str) -> List[Any]:
        """Delegate to :meth:`SceneGraphManager.parse_parentheses`."""
        return SceneGraphManager.parse_parentheses(s)

    def _is_condition_satisfied(self, condition: Union[List, Tuple]) -> bool:
        """Delegate to :meth:`SceneGraphManager._is_condition_satisfied`."""
        return self.sgm._is_condition_satisfied(condition)

    def _parse_goal_and_extract_objects(
        self, goal_condition: Union[List, Tuple]
    ) -> Tuple[List[str], Set[str]]:
        """
        Parse a goal condition and extract involved object names.

        Delegates to the SceneGraphManager's implementation to handle complex
        goal conditions while extracting all referenced objects.

        Args:
            goal_condition: The goal condition as a list or tuple.

        Returns:
            A tuple of (goal_predicates, objects_set) where goal_predicates is
            a list of predicate strings and objects_set is a set of all
            referenced object names.

        Example:
            >>> preds, objs = self._parse_goal_and_extract_objects(
            ...     ["and", ["at", "agent", "kitchen"], ["on", "cup", "table"]]
            ... )
            >>> preds
            ['at(agent, kitchen)', 'on(cup, table)']
            >>> objs
            {'agent', 'kitchen', 'cup', 'table'}
        """
        return self.sgm._parse_goal_and_extract_objects(goal_condition)

    def _is_attribute_predicate(self, pred: List[str]) -> bool:
        """
        Check if a predicate is a unary (attribute) predicate.

        """
        return len(pred) <= 2

    def _get_room_for_object(self, object_name: str) -> Optional[str]:
        """Recursively walk parent links until a ``room``-typed object is found."""
        current = object_name
        for _ in range(10):
            if self.sgm.get_object_pddl_type(current) == "room":
                return current
            parent = self.sgm.get_parent(current)
            if parent is None:
                return None
            current = parent
        return None

    def _get_furniture_in_room(self, room_name: str) -> List[str]:
        """Return all furniture objects whose parent room is *room_name*."""
        result: List[str] = []
        for link in self.sgm.scene_graph.get("links", []):
            if not isinstance(link, dict):
                continue
            if (
                link.get("to") == room_name
                and link.get("relation") == "in"
                and self.sgm.get_object_pddl_type(link.get("from", "")) == "furniture"
            ):
                result.append(link["from"])
        return result

    # ------------------------------------------------------------------
    # Context-object collection helpers
    # ------------------------------------------------------------------

    def _collect_related_objects_and_predicates(
        self,
        seed_objects: Set[str],
    ) -> Tuple[Set[Tuple[str, str]], Set[Tuple]]:
        """
        BFS from *seed_objects*, collecting object-type pairs and init
        predicates.  Seed objects get all predicates; parent objects get only
        attribute (unary) predicates.

        Returns:
            (objects_to_define, init_predicates)
        """
        objects_to_define: Set[Tuple[str, str]] = set()
        init_predicates: Set[Tuple] = set()
        processed: Set[str] = set()

        # Always include agent and its location / held items.
        for agent_info in self.sgm.scene_graph.get("agent", []) or []:
            if not isinstance(agent_info, dict):
                continue
            name = agent_info.get("name")
            if name:
                seed_objects.add(name)
            loc = agent_info.get("location")
            if loc:
                seed_objects.add(loc)
            holding = agent_info.get("holding")
            if holding:
                if isinstance(holding, list):
                    seed_objects.update(holding)
                else:
                    seed_objects.add(holding)

        queue = list(seed_objects)
        while queue:
            obj = queue.pop(0)
            if obj in processed or obj not in self.sgm.object_map:
                continue
            processed.add(obj)

            obj_type = self.sgm.get_object_pddl_type(obj)
            if obj_type:
                objects_to_define.add((obj, obj_type))

            obj_preds = self.sgm.object_map[obj]["data"].get("predicates", [])

            if obj in seed_objects:
                # Seed objects: include all predicates and referenced objects.
                for pred in obj_preds:
                    init_predicates.add(tuple(pred))
                    for item in pred[1:]:
                        item_type = self.sgm.get_object_pddl_type(str(item))
                        if item_type:
                            objects_to_define.add((str(item), item_type))
            else:
                # Parent/context objects: only attribute predicates.
                for pred in obj_preds:
                    if self._is_attribute_predicate(pred):
                        init_predicates.add(tuple(pred))

            parent = self.sgm.get_parent(obj)
            if parent and parent not in processed:
                queue.append(parent)

        return objects_to_define, init_predicates

    def _sample_room_furniture(
        self,
        seed_objects: Set[str],
        processed: Set[str],
        objects_to_define: Set[Tuple[str, str]],
        init_predicates: Set[Tuple],
        max_samples: int = 3,
    ) -> None:
        """
        Robustness sampling: pick up to *max_samples* furniture items from the
        room of interest and add their attribute predicates to the init section.
        This gives the planner extra spatial context.
        """
        room = None
        for obj in seed_objects:
            room = self._get_room_for_object(obj)
            if room:
                break
        if not room:
            return

        candidates = [
            f for f in self._get_furniture_in_room(room) if f not in processed
        ]
        # Sort for determinism; the caller is responsible for seeding random
        # when deterministic output is required.
        for furn in random.sample(candidates, min(len(candidates), max_samples)):
            processed.add(furn)
            furn_type = self.sgm.get_object_pddl_type(furn)
            if furn_type:
                objects_to_define.add((furn, furn_type))
            furn_preds = self.sgm.object_map.get(furn, {}).get("data", {}).get("predicates", [])
            for pred in furn_preds:
                if self._is_attribute_predicate(pred):
                    init_predicates.add(tuple(pred))

    # ------------------------------------------------------------------
    # PDDL string formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_objects_section(objects_to_define: Set[Tuple[str, str]]) -> str:
        by_type: Dict[str, List[str]] = defaultdict(list)
        for name, obj_type in objects_to_define:
            by_type[obj_type].append(name)
        return "".join(
            f"\n\t{' '.join(sorted(names))} - {obj_type}"
            for obj_type, names in sorted(by_type.items())
        )

    @staticmethod
    def _format_init_section(init_predicates: Set[Tuple]) -> str:
        sorted_preds = sorted(init_predicates, key=lambda x: str(x))
        return "".join(f"\n\t({' '.join(map(str, pred))})" for pred in sorted_preds)

    @staticmethod
    def _build_pddl_problem(
        problem_name: str,
        domain_name: str,
        objects_str: str,
        init_str: str,
        goal_str: str,
    ) -> str:
        return (
            f"(define (problem {problem_name})\n"
            f"    (:domain {domain_name})\n"
            f"    (:objects {objects_str}\n"
            f"    )\n\n"
            f"    (:init {init_str}\n"
            f"    )\n\n"
            f"    (:goal (and {goal_str}\n"
            f"    ))\n"
            f")"
        )

    # ------------------------------------------------------------------
    # Public API: action-list based generation
    # ------------------------------------------------------------------

    def generate_problem_from_actions(
        self,
        action_list: List[str],
        problem_name: str = "pddl_problem_from_sg",
        domain_name: str = "habitat_world_domain_final",
    ) -> str:
        """
        Generate a PDDL problem file whose ``:init`` section is populated
        with the state relevant to the given action list.  The ``:goal``
        section is intentionally left empty (the caller is expected to add
        goal predicates separately if needed).

        Args:
            action_list: List of grounded action strings, e.g.
                ``["(navigate agent_1 room_1 couch_9)"]``.
            problem_name: Name for the PDDL problem.
            domain_name: Name of the PDDL domain to reference.

        Returns:
            Formatted PDDL problem file string.
        """
        seed_objects: Set[str] = set()
        for action in action_list or []:
            params = re.findall(r"\b[a-zA-Z0-9_]+\b", action)
            if params:
                seed_objects.update(params[1:])

        objects_to_define, init_predicates = self._collect_related_objects_and_predicates(
            seed_objects
        )

        processed = {name for name, _ in objects_to_define}
        self._sample_room_furniture(
            seed_objects, processed, objects_to_define, init_predicates
        )

        objects_str = self._format_objects_section(objects_to_define)
        init_str = self._format_init_section(init_predicates)
        return self._build_pddl_problem(
            problem_name, domain_name, objects_str, init_str, goal_str=""
        )

    # ------------------------------------------------------------------
    # Public API: goal-predicate based generation
    # ------------------------------------------------------------------

    def get_related_obj(self, goal_list: List) -> List[str]:
        """
        Return all object names that appear (directly or via parent links)
        in the predicates of objects referenced by *goal_list*.
        """
        objects_to_define: Set[Tuple[str, str]] = set()
        goal_related: Set[str] = set()

        for g in goal_list:
            structured = g if not isinstance(g, str) else self.parse_parentheses(g)
            if not structured:
                continue
            _, objects = self._parse_goal_and_extract_objects(structured)
            goal_related.update(objects)

        # Include agent.
        for agent_info in self.sgm.scene_graph.get("agent", []) or []:
            if not isinstance(agent_info, dict):
                continue
            name = agent_info.get("name")
            if name:
                goal_related.add(name)
            loc = agent_info.get("location")
            if loc:
                goal_related.add(loc)
            holding = agent_info.get("holding")
            if holding:
                if isinstance(holding, list):
                    goal_related.update(holding)
                else:
                    goal_related.add(holding)

        processed: Set[str] = set()
        queue = list(goal_related)
        while queue:
            obj = queue.pop(0)
            if obj in processed or obj not in self.sgm.object_map:
                continue
            processed.add(obj)

            obj_type = self.sgm.get_object_pddl_type(obj)
            if obj_type:
                objects_to_define.add((obj, obj_type))

            if obj in goal_related:
                for pred in self.sgm.object_map[obj]["data"].get("predicates", []):
                    for item in pred[1:]:
                        item_type = self.sgm.get_object_pddl_type(str(item))
                        if item_type:
                            objects_to_define.add((str(item), item_type))

            parent = self.sgm.get_parent(obj)
            if parent and parent not in processed:
                queue.append(parent)

        return [name for name, _ in objects_to_define]

    def merge_goal_with_init(
        self, init_predicates: Set[Tuple], goal_list: List[str]
    ) -> List[str]:
        """
        Return the subset of *goal_list* predicates that are NOT already
        present in *init_predicates* (i.e. still need to be achieved).
        """
        init_pred_strs = {f"({' '.join(p)})" for p in init_predicates}
        return [pred for pred in goal_list if pred not in init_pred_strs]

    def generate_problem_file(
        self,
        goal_pair: List,
        problem_name: str = "pddl_problem_from_sg",
        domain_name: str = "habitat_world_domain_final",
    ) -> Tuple[str, List[str]]:
        """
        Generate a PDDL problem file with an explicit goal section.

        Args:
            goal_pair: A two-element list ``[goal, related_obj_list]`` where
                *goal* is a list of goal predicate strings/lists and
                *related_obj_list* is a list of seed object names.
            problem_name: Name for the PDDL problem.
            domain_name: Name of the PDDL domain to reference.

        Returns:
            ``(pddl_problem_string, merged_goal)`` where *merged_goal* is the
            subset of goal predicates that are not already in the init state.
        """
        goal: List = goal_pair[0]
        goal_related_objects: Set[str] = set(goal_pair[1])

        # Collect additional related objects from the goal predicates.
        goal_related_obj_from_goal = self.get_related_obj(goal)
        goal_related_objects.update(goal_related_obj_from_goal)

        # Parse goals and build the formatted goal strings.
        goal_str_formatted_list: List[str] = []
        for g in goal:
            structured = g if not isinstance(g, str) else self.parse_parentheses(g)
            if not structured:
                continue
            formatted_strings, objects = self._parse_goal_and_extract_objects(structured)
            goal_str_formatted_list.extend(formatted_strings)
            goal_related_objects.update(objects)

        goal_str_formatted = "".join(
            f"\n\t{s}" for s in sorted(set(goal_str_formatted_list))
        )

        # Collect predicates and object types via BFS.
        objects_to_define, init_predicates = self._collect_related_objects_and_predicates(
            copy.copy(goal_related_objects)
        )

        # Robustness sampling of extra furniture in the room.
        processed = {name for name, _ in objects_to_define}
        self._sample_room_furniture(
            goal_related_objects, processed, objects_to_define, init_predicates
        )

        merged_goal = self.merge_goal_with_init(init_predicates, goal_str_formatted_list)

        objects_str = self._format_objects_section(objects_to_define)
        init_str = self._format_init_section(init_predicates)
        pddl_problem = self._build_pddl_problem(
            problem_name, domain_name, objects_str, init_str, goal_str_formatted
        )
        return pddl_problem, merged_goal
