"""Prompt templates for LLM interactions."""

import importlib.resources as pkg_resources
from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent


def load_prompt(name: str) -> str:
    """Load a prompt template by name (without .txt extension).

    Args:
        name: Prompt file name without extension

    Returns:
        Prompt template string
    """
    prompt_path = _PROMPTS_DIR / f"{name}.txt"
    return prompt_path.read_text(encoding="utf-8")


__all__ = ["load_prompt"]
