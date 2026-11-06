"""Human-readable scene summaries and related-object details for review/repair."""


def _content_lists(scene_graph):
    result = {}
    for key, label in (
        ("rooms", "room_list"),
        ("furnitures", "furniture_list"),
        ("objects", "object_list"),
        ("agent", "agent_list"),
        ("person", "person_list"),
    ):
        items = (
            scene_graph[key]
            if key in ("rooms", "furnitures", "objects")
            else scene_graph.get(key) or []
        )
        if key == "rooms" and type(items[0]) is str:
            names = items
        else:
            names = [item["name"] for item in items]
        if key in ("rooms", "furnitures", "objects"):
            names = organize_content_list(names)
        result[label] = [
            name + get_parents_str(name, scene_graph["links"]) for name in names
        ]
    return result


def sum_sg_content(scene_graph):
    lists = _content_lists(scene_graph)
    return ", ".join(f"'{key}': {value}" for key, value in lists.items())


def sum_sg_content_for_read(scene_graph):
    lists = _content_lists(scene_graph)
    values = ["\n".join(value) for value in lists.values()]
    return f"room list: {values[0]}, \nfurniture_list: {values[1]}, \nobject_list: {values[2]}, \nagent_list: {values[3]}, \nperson_list: {values[4]}"


def organize_content_list(content_list):

    def get_sort_key(s):
        last_underscore = s.rfind("_")
        if last_underscore == -1:
            return (s, 0)
        index_str = s[last_underscore + 1 :]
        if index_str.isdigit():
            name = s[:last_underscore]
            index = int(index_str)
            return (name, index)
        else:
            return (s, 0)

    return sorted(content_list, key=get_sort_key)


def get_parents_str(name, links):
    related_links = []
    for link in links:
        if name == link[1]:
            related_links.append(link)
    if len(related_links) > 1:
        pos_str_list = []
        for link in related_links:
            pos_str_list.append(f"{link[2]} {link[0]}")
        return f"({', '.join(pos_str_list)})"
    elif len(related_links) == 1:
        return f"({related_links[0][2]} {related_links[0][0]})"
    else:
        return ""


def get_obj_details_from_scene_graph(obj_name: str, scene_graph: dict) -> dict:
    if "rooms" in scene_graph:
        if isinstance(scene_graph["rooms"], list) and len(scene_graph["rooms"]) > 0:
            if isinstance(scene_graph["rooms"][0], str):
                if obj_name in scene_graph["rooms"]:
                    return {"type": "room", "name": obj_name, "data": obj_name}
            else:
                for room in scene_graph["rooms"]:
                    if room.get("name") == obj_name:
                        return {"type": "room", "name": obj_name, "data": room}
    if "furnitures" in scene_graph:
        for furniture in scene_graph["furnitures"]:
            if furniture.get("name") == obj_name:
                return {"type": "furniture", "name": obj_name, "data": furniture}
    if "objects" in scene_graph:
        for obj in scene_graph["objects"]:
            if obj.get("name") == obj_name:
                return {"type": "object", "name": obj_name, "data": obj}
    if "agent" in scene_graph and scene_graph["agent"] is not None:
        for agent in scene_graph["agent"]:
            if agent.get("name") == obj_name:
                return {"type": "agent", "name": obj_name, "data": agent}
    if "person" in scene_graph and scene_graph["person"] is not None:
        for person in scene_graph["person"]:
            if person.get("name") == obj_name:
                return {"type": "person", "name": obj_name, "data": person}
    return None


def format_obj_details_str(obj_details: dict, scene_graph: dict) -> str:
    if obj_details is None:
        return ""
    obj_name = obj_details["name"]
    obj_data = obj_details["data"]
    if isinstance(obj_data, dict) and "states" in obj_data:
        states = obj_data["states"]
        if isinstance(states, dict):
            states_parts = []
            for state_key, state_value in states.items():
                if isinstance(state_value, bool):
                    if state_value:
                        state_str = "yes"
                        states_parts.append(f"{state_key}: {state_str}")
                else:
                    state_str = str(state_value)
                    states_parts.append(f"{state_key}: {state_str}")
            states_str = ", ".join(states_parts)
            return f"{obj_name}{get_parents_str(obj_name, scene_graph['links'])}: {states_str}"
    return ""


def get_related_objs_info(related_obj_list: list, scene_graph: dict) -> str:
    if not related_obj_list:
        return ""
    obj_info_list = []
    for obj_name in related_obj_list:
        obj_details = get_obj_details_from_scene_graph(obj_name, scene_graph)
        if obj_details is not None:
            obj_info_str = format_obj_details_str(obj_details, scene_graph)
            obj_info_list.append(obj_info_str)
        else:
            obj_info_list.append(
                f"Warning: Object '{obj_name}' not found in scene_graph"
            )
    if obj_info_list:
        return "\n\nRelated_objects_details:" + "\n".join(obj_info_list)
    else:
        return ""
