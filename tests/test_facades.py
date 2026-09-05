"""The facade split is a capability boundary, so its surface is pinned by test.

The reader is the security-relevant half: a process holding only that object must be
unable to mutate a row or reach a provider. Its operation set is asserted exactly, so
adding a write to it fails here rather than in production.
"""

import inspect

from lexi_ai import Lexicon, LexiconEngine, LexiconReader

# Every operation that changes a row or spends a provider call. None may appear on
# the reader.
WRITE_OR_PROVIDER = frozenset(
    {
        "generate",
        "generate_many",
        "generate_fenced",
        "prepare_questions",
        "add_examples",
        "backfill_embeddings",
        "resolve_relations",
        "delete_entry",
        "rename_tag",
        "delete_tag",
        "merge_tags",
        "create_theme",
        "update_theme",
        "delete_theme",
        "translate_field",
        "translate_sense",
        "translate_many",
        "tts_field",
        "tts_sense",
        "tts_many",
        "delete_asset",
        "purge_assets",
        "sweep_asset_orphans",
        "init",
    }
)


def _operations(facade):
    return {
        name
        for name, member in vars(facade).items()
        if not name.startswith("_") and name != "from_settings" and inspect.isfunction(member)
    }


def test_reader_exposes_exactly_the_free_surface():
    assert _operations(LexiconReader) == {
        "close",
        # dictionary reads
        "search",
        "semantic_search",
        "get_entry",
            "get_many",
            "get_senses",
            "word_id_for",
            "get_status",
        "get_status_many",
        "list_entries",
        "list_entries_by_tag",
        "list_tags",
        "stats",
        # themes and cached assets, read-only
        "list_themes",
        "get_theme",
        "get_asset",
        "list_assets",
        "source_hash",
        # persisted questions
        "question_types",
        "get_question",
        "list_questions_for_sense",
        "retrieve_question",
        "retrieve_exposure",
        "evaluate_answer",
    }


def test_reader_carries_no_write_or_provider_operation():
    assert not (_operations(LexiconReader) & WRITE_OR_PROVIDER)
    assert not any(
        hasattr(LexiconReader, name)
        for name in ("generate_questions_for_sense", "grade_question", "get_questions_for_sense")
    )


def test_engine_carries_the_whole_write_and_provider_surface():
    assert WRITE_OR_PROVIDER <= _operations(LexiconEngine)


def test_engine_remains_a_separate_provider_enabled_facade():
    assert LexiconEngine is not LexiconReader


def test_the_composition_root_serves_no_use_case_itself():
    """Lexicon wires services; it must not re-expose their operations.

    A delegating method here would resurrect the god object the facades replaced.
    """
    leaked = (_operations(LexiconReader) | _operations(LexiconEngine)) & _operations(Lexicon)
    assert leaked == {"close", "init"}, leaked


def test_both_facades_of_one_lexicon_share_its_process_scoped_state():
    """The reader and the engine must be views on ONE graph, not two.

    Each facade used to carry a `from_settings` of its own, which built a second
    Lexicon: two database engines and — the actual bug — two single-flight lock
    registries, so the same word could be generated twice concurrently. Building
    now starts at `Lexicon.from_settings()` and both facades wrap that one graph.
    """
    from unittest.mock import MagicMock

    lexicon = Lexicon(MagicMock(), MagicMock(), MagicMock(), vectors=MagicMock())

    assert lexicon.reader()._lexicon is lexicon.engine()._lexicon is lexicon


def test_lexicon_keeps_explicit_settings_for_all_providers(tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    from lexi_ai.config import Settings

    settings = Settings(
        asset_cache_dir=str(tmp_path),
        tts_voice="custom-voice",
        tts_format="wav",
        vector_backend="none",
    )
    monkeypatch.setenv("LEXI_VECTOR_BACKEND", "memory")
    lexicon = Lexicon(
        MagicMock(),
        MagicMock(),
        MagicMock(),
        settings=settings,
    )

    assets = lexicon.assets()

    assert lexicon._providers._settings() is settings
    assert lexicon._embedder._settings is settings
    assert lexicon._vectors is None
    assert lexicon._assets.cache_dir == tmp_path
    assert assets._voice == "custom-voice"
    assert assets._fmt == "wav"
