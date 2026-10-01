"""The public surface is pinned by test.

``Lexicon`` is the whole API. Its method set is asserted exactly, so adding or
renaming a public method fails here rather than silently changing what consumers
can call.
"""

import inspect

from lexi_ai import Lexicon

# Every operation that changes a row or spends a provider call. A caller needs
# these to be deliberate, so the set is listed rather than derived.
WRITE_OR_PROVIDER = frozenset(
 {
  "generate",
  "generate_many",
  "generate_fenced",
  "prepare_questions",
  "add_examples",
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


def _operations(cls):
 return {
  name
  for name, member in vars(cls).items()
  if not name.startswith("_") and name != "from_settings" and inspect.isfunction(member)
 }


def test_lexicon_exposes_exactly_the_public_surface():
 assert _operations(Lexicon) == {
  # lifecycle
  "init",
  "close",
  # dictionary reads
  "search",
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
  # generation, enrichment, curation
  "generate",
  "generate_many",
  "generate_fenced",
  "prepare_questions",
  "add_examples",
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
 }


def test_lexicon_carries_the_whole_write_and_provider_surface():
 assert WRITE_OR_PROVIDER <= _operations(Lexicon)


def test_the_accessor_layer_is_gone():
 """No service is reached through a public accessor any more.

 Services are attributes of the graph, wired once in ``__init__``. A public
 ``dictionary()`` / ``themes()`` / ``reader()`` would re-open a second way in
 that the merge removed.
 """
 for name in (
  "reader",
  "engine",
  "dictionary",
  "lookup",
  "tags",
  "themes",
  "enrichment",
  "generation",
  "assets",
  "questions",
 ):
  assert not hasattr(Lexicon, name), name


def test_one_lexicon_shares_its_process_scoped_state():
 """The lock registries live on the graph, not per call.

 Rebuilding them per accessor call would have collapsed nothing: each caller
 would take its own lock, so two concurrent generations of one word would both
 run. The registries are attributes for exactly that reason.
 """
 from unittest.mock import MagicMock

 lexicon = Lexicon(MagicMock(), MagicMock(), MagicMock())

 assert lexicon._locks is lexicon._locks
 assert lexicon._theme_locks is not lexicon._locks
 assert lexicon._dictionary._uow.__self__ is lexicon


def test_lexicon_keeps_explicit_settings_for_all_providers(tmp_path):
 from unittest.mock import MagicMock

 from lexi_ai.config import Settings

 settings = Settings(
  asset_cache_dir=str(tmp_path),
  tts_voice="custom-voice",
  tts_format="wav",
 )
 lexicon = Lexicon(
  MagicMock(),
  MagicMock(),
  MagicMock(),
  settings=settings,
 )

 assert lexicon._providers._settings() is settings
 assert lexicon._assets_service._voice == "custom-voice"
 assert lexicon._assets_service._fmt == "wav"
