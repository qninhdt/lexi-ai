"""Every service, constructed over a duck-typed unit of work — no database.

This is the claim the split was made for: a service depends on the unit of work
it was handed, not on SQLAlchemy. The rest of the suite exercises the services
through `Lexicon` against a real SQLite file, which proves they *work* but not
that they are *decoupled* — a service that quietly imported the ORM would pass
those tests unchanged.

So the unit of work here is a plain object. If any of these tests ever needs a
database to run, the decoupling has regressed.

The behaviours chosen are the orchestration branches that belong to the service
rather than to SQL: batch failure isolation, no-op guards, and degradation when a
collaborator is missing or broken. Anything whose correctness lives in a query is
tested against the real database instead, where a fake would only assert that the
fake was called.
"""

import pytest

from lexi_ai.dictionary.enrich import EnrichmentService
from lexi_ai.dictionary.read import DictionaryService
from lexi_ai.dictionary.tags import TagService
from lexi_ai.errors import (
 InvalidThemeName,
 ThemedOverlayMissing,
 ThemeTargetInvalid,
 UnknownTheme,
)
from lexi_ai.models import Entry, SenseView
from lexi_ai.themes import ThemeService


class FakeRepo:
 """Answers any repository call from a dict of canned returns.

 Unknown calls raise rather than returning a mock, so a service reaching for a
 method this fake was not told about fails loudly instead of silently passing.
 """

 def __init__(self, **returns) -> None:
  self._returns = returns
  self.calls: list[tuple[str, tuple, dict]] = []

 def __getattr__(self, name: str):
  if name.startswith("_"):
   raise AttributeError(name)

  async def call(*args, **kwargs):
   self.calls.append((name, args, kwargs))
   if name not in self._returns:
    raise AssertionError(f"unexpected repository call: {name}")
   value = self._returns[name]
   return value(*args, **kwargs) if callable(value) else value

  return call


class FakeUnitOfWork:
 """A unit of work with no session, no engine, and no SQL."""

 def __init__(self, **repos) -> None:
  for aggregate in ("words", "senses", "themes", "tags", "entries", "stats"):
   setattr(self, aggregate, repos.get(aggregate, FakeRepo()))
  self.commits = 0
  self.rollbacks = 0

 def __call__(self) -> "FakeUnitOfWork":
  """The services take a factory; one instance is its own factory here so a
  test can inspect what happened across every `async with` block."""
  return self

 async def __aenter__(self) -> "FakeUnitOfWork":
  return self

 async def __aexit__(self, exc_type, exc, tb) -> None:
  return None

 async def commit(self) -> None:
  self.commits += 1

 async def rollback(self) -> None:
  self.rollbacks += 1

 async def flush(self) -> None:
  return None


def _entry(word_id: int) -> Entry:
 return Entry(
  display=f"w{word_id}",
  norm=f"w{word_id}",
  entry_type="word",
  pos="noun",
  status="done",
  word_id=word_id,
 )



# --- dictionary -------------------------------------------------------------


async def test_batch_entry_reads_isolate_one_failure_from_its_siblings():
 """A bad id must be reported in its own slot, never abort the batch."""

 def entry(word_id, _overlay):
  if word_id == 2:
   raise ValueError("no such word")
  return _entry(word_id)

 uow = FakeUnitOfWork(entries=FakeRepo(entry=entry))
 service = DictionaryService(uow)

 results = await service.entries([1, 2, 3])

 assert [result.error is None for result in results] == [True, False, True]
 assert [result.value.word_id for result in results if result.value] == [1, 3]
 assert "no such word" in results[1].error


async def test_an_unknown_theme_raises_instead_of_falling_back_to_neutral():
 """Silently returning the neutral entry would hide the caller's mistake."""

 uow = FakeUnitOfWork(themes=FakeRepo(resolve=None))
 service = DictionaryService(uow)

 with pytest.raises(UnknownTheme, match="unknown theme"):
  await service.entry(1, theme="nope")


async def test_reading_no_senses_asks_the_repository_nothing():
 uow = FakeUnitOfWork()
 service = DictionaryService(uow)

 assert await service.senses([]) == []
 assert uow.entries.calls == []





# --- tags -------------------------------------------------------------------


async def test_tag_writes_commit_once_each():
 uow = FakeUnitOfWork(tags=FakeRepo(rename=True, delete=True, merge=3))
 service = TagService(uow)

 assert await service.rename("cars", name="Cars") is True
 assert await service.delete("cars") is True
 assert await service.merge(["car", "auto"], "cars") == 3
 assert uow.commits == 3


async def test_merging_forwards_the_sources_as_a_list():
 """The port takes a list; a caller passing any sequence must still work."""
 uow = FakeUnitOfWork(tags=FakeRepo(merge=0))
 service = TagService(uow)

 await service.merge(("a", "b"), "c")

 assert uow.tags.calls[0] == ("merge", (["a", "b"], "c"), {})


# --- themes -----------------------------------------------------------------


async def test_a_theme_name_that_normalizes_to_nothing_is_rejected():
 service = ThemeService(FakeUnitOfWork(), _unused, _unused, 12)

 with pytest.raises(ValueError, match="no valid key"):
  await service.create(" ", "style", description="d", tone="t")


async def test_creating_a_theme_with_full_metadata_calls_no_model():
 """Supplying description and tone must skip the LLM expansion entirely."""

 def metadata():
  raise AssertionError("the metadata generator must not be called")

 uow = FakeUnitOfWork(themes=FakeRepo(create=_ThemeRecord()))
 service = ThemeService(uow, _unused, metadata, 12)

 await service.create("Pirate", "arrr", description="d", tone="t")

 assert uow.commits == 1


async def test_updating_an_unknown_theme_raises_rather_than_creating_one():
 uow = FakeUnitOfWork(themes=FakeRepo(update=None))
 service = ThemeService(uow, _unused, _unused, 12)

 with pytest.raises(ValueError, match="unknown theme") as raised:
  await service.update("ghost", name="Ghost")
 assert isinstance(raised.value, UnknownTheme)


async def test_creating_a_theme_with_an_empty_key_raises_a_typed_error():
 service = ThemeService(FakeUnitOfWork(), _unused, _unused, 12)

 with pytest.raises(ValueError, match="yields no valid key") as raised:
  await service.create("", "style")
 assert isinstance(raised.value, InvalidThemeName)


async def test_restyling_refuses_a_word_that_is_not_done():
 """Theming reads the neutral senses, so a pending word is a caller mistake."""

 uow = FakeUnitOfWork(words=FakeRepo(status="pending"))
 service = ThemeService(uow, _unused, _unused, 12)

 with pytest.raises(ValueError, match="not done") as raised:
  await service.restyle_word(1, 2, "style")
 assert isinstance(raised.value, ThemeTargetInvalid)


async def test_appending_themed_examples_requires_an_existing_overlay():
 """Asking for examples must never theme the whole word as a side effect."""

 async def resolve(_theme):
  return (2, "style")

 uow = FakeUnitOfWork(
  senses=FakeRepo(example_context=(object(), [])),
  themes=FakeRepo(overlay_for_sense=None),
 )
 service = ThemeService(uow, _unused, _unused, 12)
 service.resolve_or_raise = resolve # type: ignore[method-assign]

 with pytest.raises(ValueError, match="no themed overlay") as raised:
  await service.append_examples(5, 3, "pirate")
 assert isinstance(raised.value, ThemedOverlayMissing)


# --- enrichment -------------------------------------------------------------


def _enrichment(uow, *, generator=None, judge=None):
 return EnrichmentService(
  uow,
  lambda: generator,
  lambda: judge,
  _unused,
  12,
 )


async def test_adding_zero_examples_calls_no_model_and_writes_nothing():
 def generator():
  raise AssertionError("the generator must not be called for n <= 0")

 uow = FakeUnitOfWork(
  senses=FakeRepo(example_context=(object(), [])),
  entries=FakeRepo(
   sense_views=[SenseView(definition="d", tier="core", pos=None, cefr_level=None)]
  ),
 )
 service = EnrichmentService(uow, generator, lambda: None, _unused, 12)

 await service.add_examples(1, n=0)

 assert uow.commits == 0


async def test_resolving_relations_without_a_judge_is_a_no_op():
 """No LLM configured degrades to doing nothing, like the other model paths."""
 uow = FakeUnitOfWork(senses=FakeRepo(pending_relations=[_task()]))

 assert await _enrichment(uow, judge=None).resolve_relations() == []


# --- shared fakes and helpers -----------------------------------------------


class _ThemeRecord:
 id = 1
 key = "pirate"
 name = "Pirate"
 style_prompt = "arrr"
 description = "d"
 tone = "t"



def _task():
 from lexi_ai.models import ResolveTask

 return ResolveTask(
  edge_id=1,
  rel_type="synonym",
  gloss="g",
  source_def="a definition",
  source_pos="noun",
  candidates=[],
 )



def _unused(*_args, **_kwargs):
 raise AssertionError("this collaborator must not be used")
