"""Construct goal- and action-based planning problems from the same scene state.

Furniture sampling intentionally uses Python random, as in the reference implementation.
Seed random before evaluation to replay this part of the protocol.
"""

import re
import random
import copy
from typing import List, Union, Set, Tuple
from robot_planning.scene_graph.manager import SceneGraphManager


class PDDLProblemGenerator:

    def __init__(self, scene_graph_manager: SceneGraphManager):
        self.sgm = scene_graph_manager
        self.pddl_parser = self.sgm.pddl_parser

    def _is_attribute_predicate(self, pred_name: str) -> bool:
        if pred_name in self.pddl_parser.predicates:
            return len(self.pddl_parser.predicates[pred_name]) < 2
        return False

    def _get_parent(self, child_name: str) -> Union[str, None]:
        if "links" not in self.sgm.scene_graph:
            return None
        for link in self.sgm.scene_graph["links"]:
            if isinstance(link, dict):
                if link.get("from") == child_name and link.get("relation") in [
                    "on",
                    "in",
                ]:
                    return link.get("to")
        return None

    def _get_room_for_object(self, object_name: str) -> Union[str, None]:
        current_obj = object_name
        for _ in range(10):
            if self.sgm.get_object_pddl_type(current_obj) == "room":
                return current_obj
            parent = self._get_parent(current_obj)
            if parent is None:
                return None
            current_obj = parent
        return None

    def _get_furniture_in_room(self, room_name: str) -> List[str]:
        furniture_list = []
        if "links" not in self.sgm.scene_graph:
            return []
        for link in self.sgm.scene_graph["links"]:
            if isinstance(link, dict):
                if (
                    link.get("to") == room_name
                    and link.get("relation") == "in"
                    and (self.sgm.get_object_pddl_type(link.get("from")) == "furniture")
                ):
                    furniture_list.append(link.get("from"))
        return furniture_list

    def _is_condition_satisfied(self, condition: Union[List, Tuple]) -> bool:
        return self.sgm._is_condition_satisfied(condition)

    def _parse_goal_and_extract_objects(
        self, goal_condition: Union[List, Tuple]
    ) -> Tuple[List[str], Set[str]]:
        return self.sgm._parse_goal_and_extract_objects(goal_condition)

    @staticmethod
    def parse_parentheses(s):
        return SceneGraphManager.parse_parentheses(s)

    def get_parsed_goal(self, goal: List[str]):
        processed_goal_list = []
        for g in goal:
            structured_g = g
            if isinstance(g, str):
                structured_g = self.parse_parentheses(g)
                formatted_strings, objects = self._parse_goal_and_extract_objects(
                    structured_g
                )
                processed_goal = self.parse_parentheses(formatted_strings)
                processed_goal_list.append(processed_goal)
        return processed_goal_list

    def get_related_obj(self, goal_list):
        objects_to_define: Set[Tuple[str, str]] = set()
        goal = copy.deepcopy(goal_list)
        goal_related_objects = set([])
        goal_str_formatted_list = []
        for g in goal:
            structured_g = g
            if isinstance(g, str):
                structured_g = self.parse_parentheses(g)
            if not structured_g:
                continue
            formatted_strings, objects = self._parse_goal_and_extract_objects(
                structured_g
            )
            goal_str_formatted_list.extend(formatted_strings)
            goal_related_objects.update(objects)
        goal_str_formatted = ""
        for s in sorted(list(set(goal_str_formatted_list))):
            goal_str_formatted += f"\n\t{s}"
        processed_objects = set()
        if "agent" in self.sgm.scene_graph:
            for agent_info in self.sgm.scene_graph["agent"]:
                agent_name = agent_info["name"]
                goal_related_objects.add(agent_name)
                if "location" in agent_info and agent_info["location"]:
                    goal_related_objects.add(agent_info["location"])
                if "holding" in agent_info and agent_info["holding"]:
                    held_items = agent_info["holding"]
                    if isinstance(held_items, list):
                        goal_related_objects.update(held_items)
                    elif isinstance(held_items, str):
                        goal_related_objects.add(held_items)
        queue = list(goal_related_objects)
        while queue:
            obj_name = queue.pop(0)
            if obj_name in processed_objects or obj_name not in self.sgm.object_map:
                continue
            processed_objects.add(obj_name)
            obj_type = self.sgm.get_object_pddl_type(obj_name)
            if obj_type:
                objects_to_define.add((obj_name, obj_type))
            obj_predicates = self.sgm.object_map[obj_name]["data"].get("predicates", [])
            if obj_name in goal_related_objects:
                for pred in obj_predicates:
                    for item in pred[1:]:
                        item_type = self.sgm.get_object_pddl_type(item)
                        if item_type:
                            objects_to_define.add((item, item_type))
            parent = self._get_parent(obj_name)
            if parent and parent not in processed_objects:
                queue.append(parent)
        objects_list = set([])
        for name, type in objects_to_define:
            objects_list.add(name)
        return list(objects_list)

    def merge_goal_with_init(self, init_list, goal_list):
        merged_goal = []
        init_pred_list = []
        for pred_tuple in init_list:
            init_pred_list.append(f"({' '.join(pred_tuple)})")
        for pred in goal_list:
            if pred not in init_pred_list:
                merged_goal.append(pred)
        return merged_goal

    def _problem_parts(self, related_objects):
        objects_to_define = set()
        init_predicates = set()
        processed_objects = set()
        if "agent" in self.sgm.scene_graph:
            for agent_info in self.sgm.scene_graph["agent"]:
                agent_name = agent_info["name"]
                related_objects.add(agent_name)
                if "location" in agent_info and agent_info["location"]:
                    related_objects.add(agent_info["location"])
                if "holding" in agent_info and agent_info["holding"]:
                    held_items = agent_info["holding"]
                    if isinstance(held_items, list):
                        related_objects.update(held_items)
                    elif isinstance(held_items, str):
                        related_objects.add(held_items)
        queue = list(related_objects)
        while queue:
            obj_name = queue.pop(0)
            if obj_name in processed_objects or obj_name not in self.sgm.object_map:
                continue
            processed_objects.add(obj_name)
            obj_type = self.sgm.get_object_pddl_type(obj_name)
            if obj_type:
                objects_to_define.add((obj_name, obj_type))
            obj_predicates = self.sgm.object_map[obj_name]["data"].get("predicates", [])
            if obj_name in related_objects:
                for pred in obj_predicates:
                    init_predicates.add(tuple(pred))
                    for item in pred[1:]:
                        item_type = self.sgm.get_object_pddl_type(item)
                        if item_type:
                            objects_to_define.add((item, item_type))
            else:
                for pred in obj_predicates:
                    if self._is_attribute_predicate(pred[0]):
                        init_predicates.add(tuple(pred))
            parent = self._get_parent(obj_name)
            if parent and parent not in processed_objects:
                queue.append(parent)
        room_of_interest = None
        for obj_name in related_objects:
            room = self._get_room_for_object(obj_name)
            if room:
                room_of_interest = room
                break
        if room_of_interest:
            all_furniture_in_room = self._get_furniture_in_room(room_of_interest)
            candidate_furniture = [
                f for f in all_furniture_in_room if f not in processed_objects
            ]
            sample_size = min(len(candidate_furniture), 3)
            sampled_furniture = random.sample(candidate_furniture, sample_size)
            for furn_name in sampled_furniture:
                processed_objects.add(furn_name)
                furn_type = self.sgm.get_object_pddl_type(furn_name)
                if furn_type:
                    objects_to_define.add((furn_name, furn_type))
                furn_predicates = self.sgm.object_map[furn_name]["data"].get(
                    "predicates", []
                )
                for pred in furn_predicates:
                    if self._is_attribute_predicate(pred[0]):
                        init_predicates.add(tuple(pred))
        return (objects_to_define, init_predicates)

    def _format_problem(
        self, objects, init, goals, problem_name, domain_name, sort_types=False
    ):
        objects_by_type = {}
        for name, kind in objects:
            objects_by_type.setdefault(kind, []).append(name)
        groups = (
            sorted(objects_by_type.items()) if sort_types else objects_by_type.items()
        )
        objects_text = "".join(
            (f"\n\t{' '.join(sorted(names))} - {kind}" for kind, names in groups)
        )
        init_text = "".join(
            (
                f"\n\t({' '.join(map(str, predicate))})"
                for predicate in sorted(init, key=str)
            )
        )
        goal_text = "".join((f"\n\t{goal}" for goal in sorted(set(goals))))
        return f"(define (problem {problem_name})\n    (:domain {domain_name})\n    (:objects {objects_text}\n    )\n\n    (:init {init_text}\n    )\n\n    (:goal (and {goal_text}\n    ))\n)"

    def generate_problem_file(
        self,
        goal_pair,
        problem_name="pddl_problem_from_sg",
        domain_name="habitat_world_domain_final",
    ):
        """Return problem text and goals not already present in its initial state."""
        goals, related_objects = (goal_pair[0], set(goal_pair[1]))
        related_objects.update(self.get_related_obj(goals))
        goal_strings = []
        for goal in goals:
            structure = self.parse_parentheses(goal) if isinstance(goal, str) else goal
            if not structure:
                continue
            strings, objects = self._parse_goal_and_extract_objects(structure)
            goal_strings.extend(strings)
            related_objects.update(objects)
        objects, initial = self._problem_parts(related_objects)
        problem = self._format_problem(
            objects, initial, goal_strings, problem_name, domain_name
        )
        return (problem, self.merge_goal_with_init(initial, goal_strings))

    def generate_problem_from_actions(
        self,
        action_list,
        problem_name="pddl_problem_from_sg",
        domain_name="habitat_world_domain_final",
    ):
        """Construct the legacy action-context problem, whose goal is intentionally empty.

        The development implementation does not derive the last action's effects,
        despite its old docstring. Preserve its actual output for data parity.
        """
        related_objects = set()
        for action in action_list:
            tokens = re.findall("\\b[a-zA-Z0-9_]+\\b", action)
            related_objects.update(tokens[1:])
        objects, initial = self._problem_parts(related_objects)
        return self._format_problem(
            objects, initial, [], problem_name, domain_name, sort_types=True
        )
