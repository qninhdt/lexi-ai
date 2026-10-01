import os

from jinja2 import Environment, FileSystemLoader

# Resolve the folder containing this loader.py file
_TEMPLATE_DIR = os.path.dirname(os.path.abspath(__file__))
_ENV = Environment(loader=FileSystemLoader(_TEMPLATE_DIR), autoescape=False)


class PromptLoader:
    """Loader for Jinja prompt templates."""

    @staticmethod
    def render(template_name: str, **kwargs) -> str:
        """Load and render a Jinja prompt template."""
        if not template_name.endswith(".jinja"):
            template_name += ".jinja"
        template = _ENV.get_template(template_name)
        # Strip trailing newlines/whitespace for cleaner messages
        return template.render(**kwargs).strip()
