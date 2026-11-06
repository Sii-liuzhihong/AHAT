"""Scene state and PDDL action execution, shared by validation and data tools."""

import copy
import pathlib
from typing import List, Dict, Any, Union, Set, Tuple
from robot_planning.pddl.parser import PDDLDomainParser


class SceneGraphManager:

    def __init__(self, scene_graph: Dict[str, Any], domain_file_path_or_file: str):
        if pathlib.Path(domain_file_path_or_file).exists():
            domain_content = pathlib.Path(domain_file_path_or_file).read_text(
                encoding="utf-8"
            )
        else:
            domain_content = domain_file_path_or_file
        self.scene_graph = copy.deepcopy(scene_graph)
        self.pddl_parser = PDDLDomainParser(domain_content)
        self._upgrade_links_format()
        self._build_object_map()
        self.extract_all_predicates()

    def _upgrade_links_format(self):
        if "links" not in self.scene_graph or not isinstance(
            self.scene_graph["links"], list
        ):
            self.scene_graph["links"] = []
            return
        upgraded_links = []
        for link in self.scene_graph["links"]:
            if isinstance(link, list) and len(link) == 3:
                obj1, obj2, relation = link
                new_link_dict = {}
                if relation in ["on", "in"]:
                    new_link_dict = {"relation": relation, "from": obj2, "to": obj1}
                if new_link_dict:
                    upgraded_links.append(new_link_dict)
            elif isinstance(link, dict):
                upgraded_links.append(link)
        self.scene_graph["links"] = upgraded_links

    def _build_object_map(self):
        self.object_map = {}
        for category, items in self.scene_graph.items():
            if category == "links":
                continue
            if isinstance(items, list):
                for i, item in enumerate(items):
                    if isinstance(item, str):
                        item_dict = {"name": item, "predicates": []}
                        self.scene_graph[category][i] = item_dict
                        item = item_dict
                    if isinstance(item, dict):
                        if "predicates" not in item:
                            item["predicates"] = []
                        if "name" in item:
                            self.object_map[item["name"]] = {
                                "category": category,
                                "data": item,
                            }

    def get_object_pddl_type(self, object_name: str) -> Union[str, None]:
        if object_name not in self.object_map:
            return None
        category = self.object_map[object_name]["category"]
        return category[:-1] if category.endswith("s") else category

    def extract_all_predicates(self) -> None:
        for obj_name in self.object_map:
            if "predicates" in self.object_map[obj_name]["data"]:
                self.object_map[obj_name]["data"]["predicates"].clear()
        for obj_name, obj_info in self.object_map.items():
            obj_data = obj_info["data"]
            if "states" in obj_data and obj_data["states"] is not None:
                for state_name, state_value in obj_data["states"].items():
                    if state_value and state_name in self.pddl_parser.predicates:
                        obj_data["predicates"].append([state_name, obj_name])
        if "links" in self.scene_graph:
            for link in self.scene_graph["links"]:
                if not isinstance(link, dict) or "relation" not in link:
                    continue
                relation = link["relation"]
                predicate = None
                if relation == "on" and "from" in link and ("to" in link):
                    obj, surface = (link["from"], link["to"])
                    obj_type = self.get_object_pddl_type(obj)
                    surface_type = self.get_object_pddl_type(surface)
                    if obj_type not in ["agent", "person"] and surface_type not in [
                        "agent",
                        "person",
                    ]:
                        predicate = ["item_on_surface", obj, surface]
                        if obj in self.object_map:
                            self.object_map[obj]["data"]["predicates"].append(predicate)
                elif relation == "in" and "from" in link and ("to" in link):
                    obj, container = (link["from"], link["to"])
                    obj_type = self.get_object_pddl_type(obj)
                    container_type = self.get_object_pddl_type(container)
                    if obj_type == "furniture" and container_type == "room":
                        predicate = ["furniture_in_room", obj, container]
                    elif obj_type == "device" and container_type == "room":
                        predicate = ["device_in_room", obj, container]
                    elif obj_type == "agent" or obj_type == "person":
                        continue
                    else:
                        predicate = ["item_in_receptacle", obj, container]
                    if obj in self.object_map:
                        self.object_map[obj]["data"]["predicates"].append(predicate)
                elif (
                    relation == "next_to"
                    and "obj1" in link
                    and ("obj2" in link)
                    and ("location" in link)
                ):
                    i1, i2, s = (link["obj1"], link["obj2"], link["location"])
                    predicate = ["next_to", i1, i2, s]
                    if i1 in self.object_map:
                        self.object_map[i1]["data"]["predicates"].append(predicate)
                    if i2 in self.object_map:
                        self.object_map[i2]["data"]["predicates"].append(predicate)
                    if s in self.object_map:
                        self.object_map[s]["data"]["predicates"].append(predicate)
        if "agent" in self.scene_graph:
            for agent_info in self.scene_graph["agent"]:
                agent_name = agent_info["name"]
                if "holding" in agent_info and agent_info["holding"] is not None:
                    held_item = agent_info["holding"]
                    self.object_map[agent_name]["data"]["predicates"].append(
                        ["holding", agent_name, held_item]
                    )
                else:
                    self.object_map[agent_name]["data"]["predicates"].append(
                        ["handempty", agent_name]
                    )
                if "location" in agent_info:
                    location_name = agent_info["location"]
                    if location_name in self.object_map:
                        self.object_map[location_name]["data"]["predicates"].append(
                            ["agent_at", agent_name, location_name]
                        )
        if "person" in self.scene_graph and self.scene_graph["person"] is not None:
            for person_info in self.scene_graph["person"]:
                person_name = person_info["name"]
                if "loaction" in person_info:
                    location_name = person_info["location"]
                    if location_name in self.object_map:
                        self.object_map[location_name]["data"]["predicates"].append(
                            ["person_at", person_name, location_name]
                        )
                if "p_holding" in person_info and person_info["p_holding"] is not None:
                    hold_item = person_info["p_holding"]
                    self.object_map[person_name]["data"]["predicates"].append(
                        ["p_holding", person_name, hold_item]
                    )
        for obj_name in self.object_map:
            preds = self.object_map[obj_name]["data"]["predicates"]
            if preds:
                unique_preds = [list(p) for p in set((tuple(p) for p in preds))]
                self.object_map[obj_name]["data"]["predicates"] = unique_preds
        return

    @staticmethod
    def parse_parentheses(s):
        s = s.strip()
        if len(s) < 2 or s[0] != "(" or s[-1] != ")":
            raise ValueError("输入字符串必须以括号开头和结尾")
        content = s[1:-1].strip()
        result = []
        start = 0
        parentheses_count = 0
        for i, char in enumerate(content):
            if char == "(":
                parentheses_count += 1
            elif char == ")":
                parentheses_count -= 1
                if parentheses_count < 0:
                    raise ValueError("括号不匹配")
            elif char == " " and parentheses_count == 0:
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
        if parentheses_count != 0:
            raise ValueError("括号不匹配")
        return result

    def _is_condition_satisfied(self, condition: Union[List, Tuple]) -> bool:
        if not isinstance(condition, (list, tuple)):
            return False
        head = condition[0]
        if head == "and":
            for sub_cond in condition[1:]:
                if not self._is_condition_satisfied(sub_cond):
                    return False
            return True
        elif head == "not":
            inner_cond = condition[1] if len(condition) == 2 else condition[1:]
            return not self._is_condition_satisfied(inner_cond)
        else:
            if isinstance(condition[0], (list, tuple)):
                pred_tuple = tuple(condition)
            else:
                pred_tuple = tuple(condition)
            if "objects" not in self.scene_graph:
                return False
            for obj in self.scene_graph["objects"]:
                obj_name = obj["name"]
                obj_predicates = (
                    self.object_map.get(obj_name, {})
                    .get("data", {})
                    .get("predicates", [])
                )
                for pred in obj_predicates:
                    if tuple(pred) == pred_tuple:
                        return True
            return False

    def _parse_goal_and_extract_objects(
        self, goal_condition: Union[List, Tuple]
    ) -> Tuple[List[str], Set[str]]:
        if not isinstance(goal_condition, (list, tuple)) or not goal_condition:
            return ([], set())
        all_goal_strings, all_objects = ([], set())
        head = goal_condition[0]
        if head == "when":
            condition_satisfied = self._is_condition_satisfied(goal_condition[1])
            if condition_satisfied:
                eff_strs, eff_objs = self._parse_goal_and_extract_objects(
                    goal_condition[2]
                )
                all_goal_strings.extend(eff_strs)
                all_objects.update(eff_objs)
        elif head == "and":
            for sub_condition in goal_condition[1:]:
                sub_strs, sub_objs = self._parse_goal_and_extract_objects(sub_condition)
                all_goal_strings.extend(sub_strs)
                all_objects.update(sub_objs)
        elif head == "not":
            if len(goal_condition) > 1 and isinstance(goal_condition[1], (list, tuple)):
                inner_pred = goal_condition[1]
            else:
                inner_pred = goal_condition[1:]
            if not inner_pred:
                return ([], set())
            inner_pred_str = " ".join(map(str, inner_pred))
            all_goal_strings.append(f"(not ({inner_pred_str}))")
            all_objects.update(inner_pred[1:])
        else:
            all_goal_strings.append(f"({' '.join(map(str, goal_condition))})")
            all_objects.update(goal_condition[1:])
        return (all_goal_strings, all_objects)

    def parse_predicate(self, predicate_with_param: list[str]):
        processed_goal_list = []
        structured_g = predicate_with_param
        if isinstance(predicate_with_param, str):
            structured_g = self.parse_parentheses(predicate_with_param)
        formatted_strings, objects = self._parse_goal_and_extract_objects(structured_g)
        for effect_string in formatted_strings:
            processed_goal = self.parse_parentheses(effect_string)
            processed_goal_list.append(processed_goal)
        return processed_goal_list

    @staticmethod
    def fill_predicate_with_params(predicate, param_map):
        if isinstance(predicate, list):
            for index, value in enumerate(predicate):
                if isinstance(value, list):
                    SceneGraphManager.fill_predicate_with_params(value, param_map)
                elif value[0] == "?":
                    predicate[index] = param_map.get(value)
                    if param_map.get(value) is None:
                        raise ValueError(f"can not get params {value} from param map")
            return predicate

    def apply_action(self, action_name: str, bound_params: List[str]):
        if action_name not in self.pddl_parser.actions:
            raise ValueError(f"动作 '{action_name}' 在PDDL领域中未定义。")
        action = self.pddl_parser.actions[action_name]
        param_keys = list(action["parameters"].keys())
        if len(bound_params) != len(param_keys):
            raise ValueError(
                f"动作 '{action_name}' 需要 {len(param_keys)} 个参数, 但提供了 {len(bound_params)} 个。 params: {bound_params}"
            )
        param_map = dict(zip(param_keys, bound_params))
        effects = copy.deepcopy(action.get("effect", []))
        if not effects:
            return
        if effects and effects[0] == "and":
            effects_to_apply = effects[1:]
        else:
            effects_to_apply = [effects]
        for effect in effects_to_apply:
            predicate_with_params = SceneGraphManager.fill_predicate_with_params(
                effect, param_map
            )
            self.apply_predicates(predicate_with_params)
        self._build_object_map()
        self.extract_all_predicates()

    def apply_predicates(self, predicate_with_param: List[Any]):
        if not isinstance(predicate_with_param, list) or not predicate_with_param:
            return
        predicate_exp_list = self.parse_predicate(predicate_with_param)
        for predicate_exp in predicate_exp_list:
            is_negated = predicate_exp[0] == "not"
            atomic_predicate = predicate_exp[1] if is_negated else predicate_exp
            if not isinstance(atomic_predicate, list) or not atomic_predicate:
                return
            pred_name = atomic_predicate[0]
            pred_params = atomic_predicate[1:]
            self._apply_effect(pred_name, pred_params, is_negated)

    def _apply_effect(self, pred_name: str, params: List[str], is_negated: bool):
        value_to_set = not is_negated
        if pred_name == "next_to":
            if len(params) != 3:
                return
            link_dict = {
                "relation": "next_to",
                "obj1": params[0],
                "obj2": params[1],
                "location": params[2],
            }
            if value_to_set:
                if link_dict not in self.scene_graph["links"]:
                    self.scene_graph["links"].append(link_dict)
            elif link_dict in self.scene_graph["links"]:
                self.scene_graph["links"].remove(link_dict)
        elif pred_name == "item_on_surface":
            if len(params) != 2:
                return
            link_dict = {"relation": "on", "from": params[0], "to": params[1]}
            if value_to_set:
                if link_dict not in self.scene_graph["links"]:
                    self.scene_graph["links"].append(link_dict)
            elif link_dict in self.scene_graph["links"]:
                self.scene_graph["links"].remove(link_dict)
        elif pred_name in ["item_in_receptacle", "furniture_in_room", "device_in_room"]:
            if len(params) != 2:
                return
            link_dict = {"relation": "in", "from": params[0], "to": params[1]}
            if value_to_set:
                if link_dict not in self.scene_graph["links"]:
                    self.scene_graph["links"].append(link_dict)
            elif link_dict in self.scene_graph["links"]:
                self.scene_graph["links"].remove(link_dict)
        elif pred_name == "agent_at":
            agent_name, new_loc = (params[0], params[1])
            if value_to_set:
                for agent in self.scene_graph["agent"]:
                    if agent["name"] == agent_name:
                        agent["location"] = new_loc
                        break
        elif pred_name == "holding":
            agent_name, obj_name = (params[0], params[1])
            for agent in self.scene_graph["agent"]:
                if agent["name"] == agent_name:
                    agent["holding"] = obj_name if value_to_set else None
                    break
        elif pred_name == "handempty":
            agent_name = params[0]
            for agent in self.scene_graph["agent"]:
                if agent["name"] == agent_name:
                    if value_to_set:
                        agent["holding"] = None
                        break
        elif pred_name == "p_holding":
            person_name, obj_name = (params[0], params[1])
            for person in self.scene_graph["person"]:
                if person["name"] == person_name:
                    person["p_holding"] = obj_name if value_to_set else None
                    break
        else:
            if not params:
                return
            obj_name = params[0]
            if obj_name in self.object_map:
                obj_data = self.object_map[obj_name]["data"]
                if "states" not in obj_data:
                    obj_data["states"] = {}
                obj_data["states"][pred_name] = value_to_set

    def get_all_predicates_from_sg(self) -> List[List[str]]:
        all_preds = []
        for obj_info in self.object_map.values():
            all_preds.extend(obj_info["data"].get("predicates", []))
        sorted_unique_preds = sorted(
            [list(p) for p in set((tuple(map(tuple, p)) for p in all_preds))]
        )
        return sorted_unique_preds
