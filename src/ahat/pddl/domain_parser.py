"""
PDDL Domain Parser
------------------
Parses a PDDL domain file and exposes its types, predicates, and actions.
"""

import re
from typing import Any, Dict, List


class PDDLDomainParser:
    """
    Parses a PDDL domain file and extracts types, predicates, and actions
    including their parameters, preconditions, and effects.
    """

    def __init__(self, domain_content: str):
        """
        Args:
            domain_content: Full string content of the PDDL domain file.
        """
        self.content = self._preprocess_content(domain_content)
        self.tokens = self.content.split()
        # Parse the token stream into a nested S-expression structure.
        self.parsed_structure = self._parse_s_expression(self.tokens)

        self.types = self._extract_types()
        self.predicates = self._extract_predicates()
        self.actions = self._extract_actions()

    # ------------------------------------------------------------------
    # Preprocessing & parsing helpers
    # ------------------------------------------------------------------

    def _preprocess_content(self, content: str) -> str:
        """Remove comments, lowercase, and normalise parenthesis spacing."""
        content = re.sub(r";.*", "", content)
        content = content.lower()
        content = content.replace("(", " ( ").replace(")", " ) ")
        return content

    def _parse_s_expression(self, tokens: List[str]) -> Any:
        """
        Recursively parse a flat token list into a nested S-expression.
        Each '(' opens a new list; ')' closes the current list.
        """
        if not tokens:
            raise ValueError("_parse_s_expression: Unexpected end of input")
        token = tokens.pop(0)
        if token == "(":
            exp = []
            while tokens and tokens[0] != ")":
                exp.append(self._parse_s_expression(tokens))
            if not tokens:
                raise ValueError("_parse_s_expression: Missing ')'")
            tokens.pop(0)  # consume ')'
            return exp
        elif token == ")":
            raise ValueError("_parse_s_expression: Unexpected ')'")
        else:
            return token

    # ------------------------------------------------------------------
    # Section finders
    # ------------------------------------------------------------------

    def _find_section(self, section_name: str) -> List[Any]:
        """Return the first top-level element whose head matches *section_name*."""
        for element in self.parsed_structure:
            if isinstance(element, list) and element and element[0] == section_name:
                return element
        return []

    # ------------------------------------------------------------------
    # Type / predicate / action extractors
    # ------------------------------------------------------------------

    def _extract_types(self) -> Dict[str, str]:
        """
        Build a dict mapping each type name to its parent type.
        The PDDL root type 'object' maps to None.
        """
        types_dict: Dict[str, str] = {"object": None}
        types_section = self._find_section(":types")
        if not types_section:
            return types_dict

        type_list = types_section[1:]
        current_types: List[str] = []
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
        """
        Parse a typed parameter sequence such as ``(?ag - agent ?obj - physobj)``
        into a mapping ``{?ag: 'agent', ?obj: 'physobj'}``.
        Parameters without an explicit type default to 'object'.
        """
        params: Dict[str, str] = {}
        i = 0
        while i < len(typed_list):
            param_name = typed_list[i]
            if i + 1 < len(typed_list) and typed_list[i + 1] == "-":
                params[param_name] = typed_list[i + 2]
                i += 3
            else:
                params[param_name] = "object"
                i += 1
        return params

    def _extract_predicates(self) -> Dict[str, Dict[str, str]]:
        """Return a dict mapping each predicate name to its typed parameter dict."""
        predicates_map: Dict[str, Dict[str, str]] = {}
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
        """
        Return a dict mapping each action name to a sub-dict containing
        'parameters', 'precondition', and 'effect' (as S-expression lists).
        """
        actions_map: Dict[str, Dict[str, Any]] = {}
        for element in self.parsed_structure:
            if isinstance(element, list) and element and element[0] == ":action":
                action_name = element[1]
                action_info: Dict[str, Any] = {"name": action_name}

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

    # ------------------------------------------------------------------
    # Public API: grounded effect extraction
    # ------------------------------------------------------------------

    def get_action_effects(self, concrete_action: str) -> List[str]:
        """
        Given a grounded action string (e.g. ``"(pick apple_1 robot_1 arm_1)"``)
        return the list of grounded effect strings after substituting concrete
        parameter values.

        Returns:
            List of PDDL effect strings such as
            ``["(not (at robot_1 room_1))", "(at-obj apple_1 room_1)"]``.

        Raises:
            ValueError: If the action string is malformed, the action name is
                unknown, or the number of arguments does not match.
        """
        try:
            processed_str = self._preprocess_content(concrete_action)
            tokens = processed_str.split()
            parsed_action = self._parse_s_expression(tokens)
        except (ValueError, IndexError) as exc:
            raise ValueError(
                f"Cannot parse action string '{concrete_action}': {exc}"
            )

        if not isinstance(parsed_action, list) or not parsed_action:
            raise ValueError(f"Invalid action format: '{concrete_action}'")

        action_name = parsed_action[0]
        concrete_args = parsed_action[1:]

        if action_name not in self.actions:
            raise ValueError(
                f"Action '{action_name}' not found in domain definition."
            )

        action_def = self.actions[action_name]
        formal_params = list(action_def.get("parameters", {}).keys())
        if len(formal_params) != len(concrete_args):
            raise ValueError(
                f"Action '{action_name}' expects {len(formal_params)} arguments "
                f"but received {len(concrete_args)}."
            )

        param_map = dict(zip(formal_params, concrete_args))

        effect_structure = action_def.get("effect", [])
        if effect_structure and effect_structure[0] == "and":
            raw_effects = effect_structure[1:]
        else:
            raw_effects = [effect_structure] if effect_structure else []

        final_effects: List[str] = []
        for effect_exp in raw_effects:
            substituted = self._substitute_params(effect_exp, param_map)
            final_effects.append(self._format_s_expression(substituted))

        return final_effects

    # ------------------------------------------------------------------
    # Internal helpers for parameter substitution and formatting
    # ------------------------------------------------------------------

    def _substitute_params(self, expression: Any, param_map: Dict[str, str]) -> Any:
        """Recursively replace parameter variables in an S-expression."""
        if not isinstance(expression, list):
            return param_map.get(expression, expression)
        return [self._substitute_params(item, param_map) for item in expression]

    def _format_s_expression(self, expression: Any) -> str:
        """Recursively format a nested-list S-expression into a PDDL string."""
        if not isinstance(expression, list):
            return str(expression)
        inner = " ".join(self._format_s_expression(item) for item in expression)
        return f"({inner})"
