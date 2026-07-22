"""Public capability facades keep provider writes out of reader processes."""

import inspect

from lexi_ai import LexiconEngine, LexiconReader


def test_reader_exposes_only_provider_free_read_operations():
    operations = {
        name
        for name, member in vars(LexiconReader).items()
        if not name.startswith("_") and inspect.iscoroutinefunction(member)
    }

    assert operations == {
        "search",
        "get_entry",
        "get_senses",
        "get_question",
        "get_questions_for_sense",
    }
    assert not any(
        hasattr(LexiconReader, name)
        for name in (
            "generate",
            "generate_fenced",
            "generate_questions_for_sense",
            "grade_question",
            "translate_field",
            "tts_field",
        )
    )


def test_engine_remains_a_separate_provider_enabled_facade():
    assert LexiconEngine is not LexiconReader
