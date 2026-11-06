"""Versioned prompt resources, preserved byte for byte from the research code."""

from importlib.resources import files


def load_prompt(name: str) -> str:
    """Load an explicitly named prompt, or accept a literal multiline prompt."""
    if "\n" in name:
        return name
    if not name or not all(char.isalnum() or char == "_" for char in name):
        raise ValueError("Prompt must be a registered name or a multiline literal")
    resource = files(__package__).joinpath(f"{name}.txt")
    if not resource.is_file():
        raise KeyError(f"Unknown prompt: {name}")
    return resource.read_text(encoding="utf-8")
