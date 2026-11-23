from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
from jinja2 import Environment, FileSystemLoader, select_autoescape

_TEMPLATES_DIR = Path(__file__).parent.parent / 'templates'


def md_cell(value: Any) -> str:
    """Text safe inside a Markdown table cell: a pipe would start a new column, a newline end the row."""
    return ' '.join(str(value).split()).replace('|', '\\|')


def get_env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=select_autoescape(['html', 'xml']),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters['md_cell'] = md_cell
    return env


def render(template_name: str, context: Dict[str, Any]) -> str:
    return get_env().get_template(template_name).render(**context)
