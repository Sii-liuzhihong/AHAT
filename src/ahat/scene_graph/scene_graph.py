"""
Scene Graph Manager
-------------------
Manages a mutable scene-graph state and applies PDDL actions to update it.

Canonical shared PDDL infrastructure for the AHAT project.  Used by both
the feasibility evaluator (``evaluator.feasibility``) and the subgoal
evaluator (``evaluator.subgoal``).

Extends the base scene-graph initialisation (links, states, agent/person)
with:
  - ``pddl_parser`` (PDDLDomainParser) for predicate / action look-ups
  - ``apply_action`` / ``apply_predicates`` / ``_apply_effect`` for state updates
  - ``parse_parentheses``, ``_is_condition_satisfied``,
    ``_parse_goal_and_extract_objects``, ``parse_predicate`` for goal handling
  - ``get_all_predicates_from_sg`` for introspection
"""

import copy
import pathlib
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ahat.pddl.domain_parser import PDDLDomainParser


class SceneGraphManager:
    """
    Manages the scene-graph state and applies PDDL actions to update it.

    The scene graph is expected to be a dict with the following top-level keys:
        ``objects``, ``links``, ``agent``, ``person``, ``furniture``, etc.

    Args:
        scene_graph: The initial scene-graph dict.
        domain_file_path_or_file: Either a path to a ``.pddl`` domain file or
            the raw PDDL domain content as a string.
    """

    def __init__(
        self, scene_graph: Dict[str, Any], domain_file_path_or_file: str
    ):
        # Accept both a file path and raw PDDL content.
        # Use try/except because a raw PDDL string may exceed the OS path-length
        # limit and cause an OSError when passed to pathlib.Path.exists().
        try:
            p = pathlib.Path(domain_file_path_or_file)
            if p.exists():
                domain_content = p.read_text(encoding="utf-8")
            else:
                domain_content = domain_file_path_or_file
        except OSError:
            domain_content = domain_file_path_or_file

        self.scene_graph = copy.deepcopy(scene_graph)
        self.pddl_parser = PDDLDomainParser(domain_content)

        self._upgrade_links_format()
        self._build_object_map()
        self.extract_all_predicates()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _upgrade_links_format(self) -> None:
        """
        Normalise the ``links`` list to dicts.

        Old format: ``[parent, child, relation]``
        New format: ``{"relation": ..., "from": child, "to": parent}``

        The conversion is idempotent: dicts are left unchanged.
        """
        if "links" not in self.scene_graph or not isinstance(self.scene_graph["links"], list):
            self.scene_graph["links"] = []
            return

        upgraded: List[Dict] = []
        for link in self.scene_graph["links"]:
            if isinstance(link, dict):
                upgraded.append(link)
                continue
            if isinstance(link, list) and len(link) == 3:
                parent, child, relation = link
                if relation in ("on", "in"):
                    upgraded.append({"relation": relation, "from": child, "to": parent})
        self.scene_graph["links"] = upgraded

    def _build_object_map(self) -> None:
        """
        Build ``self.object_map``: a flat mapping from object name to its
        category and data dict.  Ensures every item has a ``predicates`` list.
        """
        self.object_map: Dict[str, Dict[str, Any]] = {}
        for category, items in self.scene_graph.items():
            if category == "links" or not isinstance(items, list):
                continue
            for i, item in enumerate(items):
                if isinstance(item, str):
                    item = {"name": item, "predicates": []}
                    self.scene_graph[category][i] = item
                if isinstance(item, dict) and "name" in item:
                    item.setdefault("predicates", [])
                    self.object_map[item["name"]] = {"category": category, "data": item}

    # ------------------------------------------------------------------
    # Type helpers
    # ------------------------------------------------------------------

    def get_object_pddl_type(self, object_name: str) -> Optional[str]:
        """Return the PDDL type (singularised category name) of *object_name*."""
        obj = self.object_map.get(object_name)
        if not obj:
            return None
        category = obj.get("category") or obj.get("type")
        return category[:-1] if category.endswith("s") else category

    # ------------------------------------------------------------------
    # Predicate extraction
    # ------------------------------------------------------------------

    def extract_all_predicates(self):
        """
        (Re-)populate the ``predicates`` list on every object from the
        current scene-graph state (links, states, agent/person info).
        Call this after any structural change to keep the map consistent.
        """
        # Clear existing predicate lists.
        for name in self.object_map:
            preds = self.object_map[name]["data"].get("predicates")
            if isinstance(preds, list):
                preds.clear()
            else:
                self.object_map[name]["data"]["predicates"] = []

        # Unary state predicates (e.g. is_clean, is_open).
        for obj_name, obj_info in self.object_map.items():
            states = obj_info["data"].get("states") or {}
            if isinstance(states, dict):
                for state_name, state_value in states.items():
                    if bool(state_value):
                        obj_info["data"]["predicates"].append([state_name, obj_name])

        # Spatial relationship predicates from links.
        for link in self.scene_graph.get("links", []):
            if not isinstance(link, dict):
                continue
            relation = link.get("relation")

            if relation == "on":
                obj, surface = link.get("from"), link.get("to")
                if obj in self.object_map:
                    obj_type = self.get_object_pddl_type(obj)
                    surface_type = self.get_object_pddl_type(surface or "")
                    if obj_type not in ("agent", "person") and surface_type not in ("agent", "person"):
                        self.object_map[obj]["data"]["predicates"].append(
                            ["item_on_surface", obj, surface]
                        )

            elif relation == "in":
                obj, container = link.get("from"), link.get("to")
                if obj not in self.object_map:
                    continue
                obj_type = self.get_object_pddl_type(obj)
                container_type = self.get_object_pddl_type(container or "")
                if obj_type in ("agent", "person"):
                    continue
                if obj_type == "furniture" and container_type == "room":
                    pred = ["furniture_in_room", obj, container]
                elif obj_type == "device" and container_type == "room":
                    pred = ["device_in_room", obj, container]
                else:
                    pred = ["item_in_receptacle", obj, container]
                self.object_map[obj]["data"]["predicates"].append(pred)

            elif relation == "next_to":
                i1, i2, loc = link.get("obj1"), link.get("obj2"), link.get("location")
                pred = ["next_to", i1, i2, loc]
                for n in (i1, i2, loc):
                    if n in self.object_map:
                        self.object_map[n]["data"]["predicates"].append(pred)

        # Agent predicates.
        for agent in self.scene_graph.get("agent", []) or []:
            if not isinstance(agent, dict):
                continue
            agent_name = agent.get("name")
            if agent_name not in self.object_map:
                continue
            holding = agent.get("holding")
            if holding:
                if isinstance(holding, list):
                    for item in holding:
                        self.object_map[agent_name]["data"]["predicates"].append(
                            ["holding", agent_name, item]
                        )
                else:
                    self.object_map[agent_name]["data"]["predicates"].append(
                        ["holding", agent_name, holding]
                    )
            else:
                self.object_map[agent_name]["data"]["predicates"].append(
                    ["handempty", agent_name]
                )
            location = agent.get("location")
            if location and location in self.object_map:
                self.object_map[location]["data"]["predicates"].append(
                    ["agent_at", agent_name, location]
                )

        # Person predicates.
        for person in self.scene_graph.get("person", []) or []:
            if not isinstance(person, dict):
                continue
            person_name = person.get("name")
            if not person_name:
                continue
            location = person.get("location")
            if location and location in self.object_map:
                self.object_map[location]["data"]["predicates"].append(
                    ["person_at", person_name, location]
                )
            p_holding = person.get("p_holding")
            if p_holding and person_name in self.object_map:
                self.object_map[person_name]["data"]["predicates"].append(
                    ["p_holding", person_name, p_holding]
                )

        # De-duplicate predicate lists.
        for name in self.object_map:
            preds = self.object_map[name]["data"].get("predicates", [])
            self.object_map[name]["data"]["predicates"] = [
                list(p) for p in {tuple(x) for x in preds}
            ]

    # ------------------------------------------------------------------
    # Graph query helpers
    # ------------------------------------------------------------------

    def get_parent(self, child_name: str) -> Optional[str]:
        """Return the direct spatial parent of *child_name* (on/in relation)."""
        for link in self.scene_graph.get("links", []):
            if isinstance(link, dict):
                if link.get("relation") in ("on", "in") and link.get("from") == child_name:
                    return link.get("to")
        return None

    def get_all_predicates_from_sg(self) -> List[List[str]]:
        """Return a sorted, deduplicated list of all predicates in the scene graph."""
        all_preds: List[List[str]] = []
        for obj_info in self.object_map.values():
            all_preds.extend(obj_info["data"].get("predicates", []))
        return sorted([list(p) for p in {tuple(map(str, p)) for p in all_preds}])

    # ------------------------------------------------------------------
    # Action application
    # ------------------------------------------------------------------

    def apply_action(self, action_name: str, bound_params: List[str]) -> None:
        """
        Apply a grounded PDDL action to the scene graph.

        Args:
            action_name: Name of the action (e.g. ``"pick_from_surface"``).
            bound_params: Concrete argument values in parameter order.
        """
        if action_name not in self.pddl_parser.actions:
            raise ValueError(f"Action '{action_name}' is not defined in the domain.")

        action = self.pddl_parser.actions[action_name]
        param_keys = list(action["parameters"].keys())

        if len(bound_params) != len(param_keys):
            raise ValueError(
                f"Action '{action_name}' expects {len(param_keys)} parameters "
                f"but received {len(bound_params)}: {bound_params}"
            )

        param_map = dict(zip(param_keys, bound_params))
        effects = copy.deepcopy(action.get("effect", []))
        if not effects:
            return

        effects_to_apply = effects[1:] if effects and effects[0] == "and" else [effects]

        for effect in effects_to_apply:
            predicate_with_params = self._fill_predicate_with_params(effect, param_map)
            self.apply_predicates(predicate_with_params)

        # Rebuild maps so all downstream queries see a consistent state.
        self._build_object_map()
        self.extract_all_predicates()

    def apply_predicates(self, predicate_with_param: List[Any]) -> None:
        """
        Apply a single concrete predicate (positive or negative) to the
        scene graph.

        Args:
            predicate_with_param: A list such as
                ``["item_on_surface", "apple_1", "table_1"]`` (positive) or
                ``["not", ["holding", "agent_0", "apple_1"]]`` (negative).
        """
        if not isinstance(predicate_with_param, list) or not predicate_with_param:
            return

        predicate_exp_list = self.parse_predicate(predicate_with_param)
        for predicate_exp in predicate_exp_list:
            is_negated = predicate_exp[0] == "not"
            atomic = predicate_exp[1] if is_negated else predicate_exp

            if not isinstance(atomic, list) or not atomic:
                return

            self._apply_effect(atomic[0], atomic[1:], is_negated)

    def _apply_effect(
        self, pred_name: str, params: List[str], is_negated: bool
    ) -> None:
        """Apply a single grounded predicate to the scene-graph data structures."""
        add = not is_negated

        if pred_name == "next_to":
            if len(params) != 3:
                return
            link_dict = {"relation": "next_to", "obj1": params[0], "obj2": params[1], "location": params[2]}
            links = self.scene_graph.setdefault("links", [])
            if add:
                if link_dict not in links:
                    links.append(link_dict)
            else:
                if link_dict in links:
                    links.remove(link_dict)

        elif pred_name == "item_on_surface":
            if len(params) != 2:
                return
            link_dict = {"relation": "on", "from": params[0], "to": params[1]}
            links = self.scene_graph.setdefault("links", [])
            if add:
                if link_dict not in links:
                    links.append(link_dict)
            else:
                if link_dict in links:
                    links.remove(link_dict)

        elif pred_name in ("item_in_receptacle", "furniture_in_room", "device_in_room"):
            if len(params) != 2:
                return
            link_dict = {"relation": "in", "from": params[0], "to": params[1]}
            links = self.scene_graph.setdefault("links", [])
            if add:
                if link_dict not in links:
                    links.append(link_dict)
            else:
                if link_dict in links:
                    links.remove(link_dict)

        elif pred_name == "agent_at":
            agent_name, new_loc = params[0], params[1]
            if add:
                for agent in self.scene_graph.get("agent", []):
                    if agent.get("name") == agent_name:
                        agent["location"] = new_loc
                        break

        elif pred_name == "holding":
            agent_name, obj_name = params[0], params[1]
            for agent in self.scene_graph.get("agent", []):
                if agent.get("name") == agent_name:
                    agent["holding"] = obj_name if add else None
                    break

        elif pred_name == "handempty":
            agent_name = params[0]
            if add:
                for agent in self.scene_graph.get("agent", []):
                    if agent.get("name") == agent_name:
                        agent["holding"] = None
                        break

        elif pred_name == "p_holding":
            person_name, obj_name = params[0], params[1]
            for person in self.scene_graph.get("person", []):
                if person.get("name") == person_name:
                    person["p_holding"] = obj_name if add else None
                    break

        else:
            # Unary state predicates (e.g. is_clean, is_open).
            if not params:
                return
            obj_name = params[0]
            if obj_name in self.object_map:
                obj_data = self.object_map[obj_name]["data"]
                obj_data.setdefault("states", {})
                obj_data["states"][pred_name] = add

    # ------------------------------------------------------------------
    # Parameter substitution helper
    # ------------------------------------------------------------------

    @staticmethod
    def _fill_predicate_with_params(
        predicate: Any, param_map: Dict[str, str]
    ) -> Any:
        """
        Recursively substitute PDDL variable names (``?var``) with concrete
        values from *param_map* in a nested list predicate.
        """
        if isinstance(predicate, list):
            for idx, value in enumerate(predicate):
                if isinstance(value, list):
                    SceneGraphManager._fill_predicate_with_params(value, param_map)
                else:
                    if isinstance(value, str) and value.startswith("?"):
                        result = param_map.get(value)
                        if result is None:
                            raise ValueError(
                                f"Cannot find parameter '{value}' in param_map"
                            )
                        predicate[idx] = result
            return predicate
        return predicate

    # ------------------------------------------------------------------
    # S-expression / goal parsing utilities
    # (shared with PDDLProblemGenerator to avoid duplication)
    # ------------------------------------------------------------------

    @staticmethod
    def parse_parentheses(s: str) -> List[Any]:
        """
        Parse a parenthesised PDDL expression string into a nested list.

        Example::

            "(item_on_surface apple_1 table_1)"
            -> ["item_on_surface", "apple_1", "table_1"]
        """
        s = s.strip()
        if len(s) < 2 or s[0] != "(" or s[-1] != ")":
            raise ValueError(f"parse_parentheses: string must be wrapped in parentheses: {s!r}")

        content = s[1:-1].strip()
        result: List[Any] = []
        start = 0
        depth = 0

        for i, char in enumerate(content):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth < 0:
                    raise ValueError("parse_parentheses: unmatched ')'")
            elif char == " " and depth == 0:
                if start < i:
                    element = content[start:i].strip()
                    if element.startswith("(") and element.endswith(")"):
                        result.append(SceneGraphManager.parse_parentheses(element))
                    else:
                        result.append(element)
                start = i + 1

        if start <= len(content):
            element = content[start:].strip()
            if element:
                if element.startswith("(") and element.endswith(")"):
                    result.append(SceneGraphManager.parse_parentheses(element))
                else:
                    result.append(element)

        if depth != 0:
            raise ValueError("parse_parentheses: unmatched parentheses")

        return result

    def _is_condition_satisfied(self, condition: Union[List, Tuple]) -> bool:
        """
        Check whether *condition* is satisfied in the current scene graph.
        Handles ``and``, ``not``, and atomic predicates.
        """
        if not isinstance(condition, (list, tuple)):
            return False

        head = condition[0]

        if head == "and":
            return all(self._is_condition_satisfied(sub) for sub in condition[1:])

        if head == "not":
            inner = condition[1] if len(condition) == 2 else condition[1:]
            return not self._is_condition_satisfied(inner)

        # Atomic predicate.
        pred_tuple = tuple(condition)
        for obj in self.scene_graph.get("objects", []):
            obj_name = obj["name"]
            obj_preds = self.object_map.get(obj_name, {}).get("data", {}).get("predicates", [])
            for pred in obj_preds:
                if tuple(pred) == pred_tuple:
                    return True
        return False

    def _parse_goal_and_extract_objects(
        self, goal_condition: Union[List, Tuple]
    ) -> Tuple[List[str], Set[str]]:
        """
        Recursively flatten a goal condition into PDDL strings and a set of
        referenced object names.  Handles ``when``, ``and``, ``not``, and
        atomic predicates.

        Returns:
            (goal_strings, object_names)
        """
        if not isinstance(goal_condition, (list, tuple)) or not goal_condition:
            return [], set()

        all_strings: List[str] = []
        all_objects: Set[str] = set()
        head = goal_condition[0]

        if head == "when":
            if self._is_condition_satisfied(goal_condition[1]):
                strs, objs = self._parse_goal_and_extract_objects(goal_condition[2])
                all_strings.extend(strs)
                all_objects.update(objs)

        elif head == "and":
            for sub in goal_condition[1:]:
                strs, objs = self._parse_goal_and_extract_objects(sub)
                all_strings.extend(strs)
                all_objects.update(objs)

        else:
            if head == "not":
                # Support both structured ['not', ['pred', ...]]
                # and flat ['not', 'pred', 'arg1', ...] forms.
                if len(goal_condition) > 1 and isinstance(goal_condition[1], (list, tuple)):
                    inner_pred = goal_condition[1]
                else:
                    inner_pred = goal_condition[1:]
                if not inner_pred:
                    return [], set()
                all_strings.append(f"(not ({' '.join(map(str, inner_pred))}))")
                all_objects.update(inner_pred[1:])
            else:
                all_strings.append(f"({' '.join(map(str, goal_condition))})")
                all_objects.update(goal_condition[1:])

        return all_strings, all_objects

    def parse_predicate(
        self, predicate_with_param: Union[List[Any], str]
    ) -> List[List[Any]]:
        """
        Convert a raw predicate (string or list) into a list of parsed
        nested-list predicates via ``_parse_goal_and_extract_objects``.
        """
        structured = predicate_with_param
        if isinstance(predicate_with_param, str):
            structured = self.parse_parentheses(predicate_with_param)

        formatted_strings, _ = self._parse_goal_and_extract_objects(structured)
        result: List[List[Any]] = []
        for effect_string in formatted_strings:
            result.append(self.parse_parentheses(effect_string))
        return result
