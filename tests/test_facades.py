"""Public capability facades keep provider writes out of reader processes."""

import inspect

from lexi_ai import LexiconEngine, LexiconReader


def _operations(facade):
    return {
        name
        for name, member in vars(facade).items()
        if not name.startswith("_")
        and name != "from_settings"
        and inspect.isfunction(member)
    }


def test_reader_exposes_exact_provider_free_question_surface():
    assert _operations(LexiconReader) == {
        "search",
        "get_entry",
        "get_senses",
        "question_types",
        "get_question",
        "list_questions_for_sense",
        "retrieve_question",
        "retrieve_exposure",
        "evaluate_answer",
    }
    assert not any(
        hasattr(LexiconReader, name)
        for name in (
            "generate",
            "generate_fenced",
            "prepare_questions",
            "generate_questions_for_sense",
            "grade_question",
            "get_questions_for_sense",
            "translate_field",
            "tts_field",
        )
    )


def test_engine_exposes_exact_provider_enabled_surface():
    assert _operations(LexiconEngine) == {
        "question_types",
        "prepare_questions",
        "evaluate_answer",
        "generate",
        "generate_fenced",
        "translate_field",
        "tts_field",
    }
    assert not any(
        hasattr(LexiconEngine, name)
        for name in (
            "get_question",
            "list_questions_for_sense",
            "retrieve_question",
            "retrieve_exposure",
            "generate_questions_for_sense",
            "grade_question",
            "get_questions_for_sense",
        )
    )


def test_engine_remains_a_separate_provider_enabled_facade():
    assert LexiconEngine is not LexiconReader
