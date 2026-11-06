"""PDDL domain parsing and grounded action effects."""

import re
from typing import Any, Dict, List


def parse_tokens(tokens):
    """Consume one parenthesized PDDL expression from a mutable token list."""
    if not tokens:
        raise ValueError("s_exp_parse: Unexpected end of input")
    token = tokens.pop(0)
    if token == "(":
        expression = []
        while tokens and tokens[0] != ")":
            expression.append(parse_tokens(tokens))
        if not tokens:
            raise ValueError("s_exp_parse: Missing ')'")
        tokens.pop(0)
        return expression
    if token == ")":
        raise ValueError("s_exp_parse: Unexpected ')'")
    return token


class PDDLDomainParser:

    def __init__(self, domain_content: str):
        self.content = self._preprocess_content(domain_content)
        self.tokens = self.content.split()
        self.parsed_structure = self._s_exp_parse(self.tokens)
        self.types = self._extract_types()
        self.predicates = self._extract_predicates()
        self.actions = self._extract_actions()

    def _preprocess_content(self, content: str) -> str:
        content = re.sub(";.*", "", content)
        content = content.lower()
        content = content.replace("(", " ( ").replace(")", " ) ")
        return content

    def _s_exp_parse(self, tokens: List[str]) -> List[Any]:
        return parse_tokens(tokens)

    def _find_section(self, section_name: str) -> List[Any]:
        for element in self.parsed_structure:
            if isinstance(element, list) and element and (element[0] == section_name):
                return element
        return []

    def _extract_types(self) -> Dict[str, str]:
        types_dict = {"object": None}
        types_section = self._find_section(":types")
        if not types_section:
            return types_dict
        type_list = types_section[1:]
        current_types = []
        for item in type_list:
            if item == "-":
                parent_type = type_list[type_list.index(item) + 1]
                for t in current_types:
                    types_dict[t] = parent_type
                current_types = []
            else:
                if item not in types_dict:
                    types_dict[item] = "object"
                current_types.append(item)
        return types_dict

    def _parse_typed_list(self, typed_list: List[str]) -> Dict[str, str]:
        params = {}
        i = 0
        while i < len(typed_list):
            param_name = typed_list[i]
            if i + 1 < len(typed_list) and typed_list[i + 1] == "-":
                param_type = typed_list[i + 2]
                params[param_name] = param_type
                i += 3
            else:
                params[param_name] = "object"
                i += 1
        return params

    def _extract_predicates(self) -> Dict[str, Dict[str, str]]:
        predicates_map = {}
        predicates_section = self._find_section(":predicates")
        if not predicates_section:
            return predicates_map
        for pred_def in predicates_section[1:]:
            if isinstance(pred_def, list):
                pred_name = pred_def[0]
                params = self._parse_typed_list(pred_def[1:])
                predicates_map[pred_name] = params
        return predicates_map

    def _extract_actions(self) -> Dict[str, Dict[str, Any]]:
        actions_map = {}
        for element in self.parsed_structure:
            if isinstance(element, list) and element and (element[0] == ":action"):
                action_name = element[1]
                action_info = {"name": action_name}
                for i in range(2, len(element), 2):
                    key = element[i]
                    value = element[i + 1]
                    if key == ":parameters":
                        action_info["parameters"] = self._parse_typed_list(value)
                    elif key == ":precondition":
                        action_info["precondition"] = value
                    elif key == ":effect":
                        action_info["effect"] = value
                actions_map[action_name] = action_info
        return actions_map

    def get_action_effects(self, concrete_action: str) -> List[str]:
        try:
            processed_str = self._preprocess_content(concrete_action)
            tokens = processed_str.split()
            parsed_action = self._s_exp_parse(tokens)
        except (ValueError, IndexError) as e:
            raise ValueError(f"无法解析动作字符串: '{concrete_action}'. 错误: {e}")
        if not isinstance(parsed_action, list) or len(parsed_action) < 1:
            raise ValueError(f"无效的动作格式: '{concrete_action}'")
        action_name = parsed_action[0]
        concrete_args = parsed_action[1:]
        if action_name not in self.actions:
            raise ValueError(f"动作 '{action_name}' 在领域定义中未找到。")
        action_def = self.actions[action_name]
        formal_params = list(action_def.get("parameters", {}).keys())
        if len(formal_params) != len(concrete_args):
            raise ValueError(
                f"动作 '{action_name}' 的参数数量不匹配。需要 {len(formal_params)} 个，但提供了 {len(concrete_args)} 个。"
            )
        param_map = dict(zip(formal_params, concrete_args))
        effect_structure = action_def.get("effect", [])
        if effect_structure and effect_structure[0] == "and":
            raw_effects = effect_structure[1:]
        else:
            raw_effects = [effect_structure] if effect_structure else []
        final_effects = []
        for effect_exp in raw_effects:
            substituted_exp = self._substitute_expression(effect_exp, param_map)
            final_effects.append(self._format_expression_to_string(substituted_exp))
        return final_effects

    def _substitute_expression(self, expression: Any, param_map: Dict[str, str]) -> Any:
        if not isinstance(expression, list):
            return param_map.get(expression, expression)
        else:
            return [self._substitute_expression(item, param_map) for item in expression]

    def _format_expression_to_string(self, expression: Any) -> str:
        if not isinstance(expression, list):
            return str(expression)
        inner_content = " ".join(
            (self._format_expression_to_string(item) for item in expression)
        )
        return f"({inner_content})"
