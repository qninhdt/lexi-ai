"""Pure public types can be imported without provider or persistence side effects."""

import subprocess
import sys

from lexi_ai import Lexicon


def test_public_values_do_not_eagerly_import_providers_or_orm():
    code = """
import sys
import lexi_ai
from lexi_ai import DecisionConfig, DecisionMode, LLMConfig, StructuredLLM, TokenUsage
from lexi_ai.models import Word, Question
assert 'lexi_ai_v2' not in sys.modules
assert 'sqlalchemy' not in sys.modules
assert 'openai' not in sys.modules
assert 'typesafe_sdk' not in sys.modules
assert StructuredLLM is not None
assert LLMConfig().model == 'gpt-4o'
assert LLMConfig(structured_outputs=False).structured_outputs is False
assert DecisionConfig(0.8).accepts(0.8)
assert TokenUsage('actual', 1, 0, None, 2).input_tokens == 1
assert {mode.value for mode in DecisionMode} == {'llm_fallback', 'decision_only', 'llm_only'}
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_only_intended_public_methods():
    actual = {name for name in dir(Lexicon) if not name.startswith("_")}
    assert actual == {
        "search",
        "generate",
        "get_word",
        "get_senses",
        "close",
        "create_theme",
        "get_theme",
        "list_themes",
        "update_theme",
        "delete_theme",
        "generate_questions",
        "get_question",
        "list_questions",
        "retrieve_question",
        "delete_question",
        "grade_answer",
        "resolve_relations",
        "translate_text",
        "get_translation",
        "list_translations",
        "delete_translation",
        "purge_translations",
    }
