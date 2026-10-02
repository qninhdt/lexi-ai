"""Pure public types can be imported without provider or persistence side effects."""

import subprocess
import sys


def test_public_values_do_not_eagerly_import_providers_or_orm():
    code = """
import sys
import lexi_ai
from lexi_ai import DecisionConfig, DecisionMode, LLMConfig, StructuredLLM, TokenUsage
from lexi_ai.models import Word, Question
assert 'sqlalchemy' not in sys.modules
assert 'openai' not in sys.modules
assert 'typesafe_sdk' not in sys.modules
assert StructuredLLM is not None
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
