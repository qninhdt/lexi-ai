"""Render domain-owned Jinja/JSON-e resources without interpreting context as templates."""

import json
import re
from functools import lru_cache
from pathlib import Path

import jsone
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from typesafe_sdk import Choice, Noul

_ROOT = Path(__file__).parents[1]

_ENV = Environment(
    loader=FileSystemLoader(_ROOT),
    undefined=StrictUndefined,
    autoescape=False,
    keep_trailing_newline=True,
)
_ROLE = re.compile(r"\{#\s*(system|user)\s*#\}")


@lru_cache(maxsize=16)
def _templates(name):
    source, _, _ = _ENV.loader.get_source(_ENV, name)
    markers = list(_ROLE.finditer(source))
    if [marker.group(1) for marker in markers] != ["system", "user"] or source[
        : markers[0].start()
    ].strip():
        raise ValueError(f"Prompt {name} must declare system then user exactly once")
    instruction = source[markers[0].end() : markers[1].start()]
    data = source[markers[1].end() :]
    return _ENV.from_string(instruction), _ENV.from_string(data)


def render_prompt(name: str, **context) -> tuple[str, str]:
    instruction, data = _templates(name)
    return instruction.render(**context).strip(), data.render(**context).strip()


@lru_cache(maxsize=16)
def _decision_template(name):
    return json.loads((_ROOT / name).read_text(encoding="utf-8"))


def render_decision(name: str, **context):
    """Render trusted JSON-e once; context text is data, never a second template."""
    request = jsone.render(_decision_template(name), context)
    types = {"choice": Choice, "noul": Noul}
    questions = {
        key: types[value["type"]].model_validate(value)
        for key, value in request["questions"].items()
    }
    return request["state"], questions
