"""Persistence ports: what the domain needs, stated without SQLAlchemy.

Each aggregate gets one narrow Protocol carrying only the operations its real
callers use. Implementations live in ``lexi_ai/infrastructure/db/repositories``.

Repositories are SESSION-BOUND: a :class:`UnitOfWork` hands each one the single
session it shares, so several repository calls compose into one transaction
without any repository opening a transaction of its own.

Two aggregates are deliberately absent from :class:`UnitOfWork`:

* Assets are a best-effort cache reconciled after the write commits, and their
  repository already accepts a caller session for the one case that must be
  atomic (garbage-collecting a deleted word's cached audio).
* Questions are written by the assessment plugins in their own transaction.

Folding either into the shared unit of work would put a best-effort step inside
a transaction that must not roll back because of it.
"""

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Protocol

from lexi_ai.domain.models import (
    EmbeddedSenseRow,
    GenerationFence,
    ResolveDecision,
    ResolveOutcome,
    ResolveTask,
    SenseEmbeddingNeed,
    TagName,
    TagUsage,
    ThemeRecord,
    ThemingSense,
    WordListing,
    WordRecord,
)

if TYPE_CHECKING:
    from lexi_ai.generation.schemas import (
        ExampleGenContext,
        GeneratedAlias,
        GeneratedEntry,
        GeneratedSense,
        GeneratedTopic,
        RelatedWord,
    )
    from lexi_ai.read_models import Stats
    from lexi_ai.theming.schemas import ThemedResult


class WordRepo(Protocol):
    """The ``words`` aggregate: identity, lifecycle status, aliases, word links."""

    async def get_or_create(self, norm: str) -> int:
        """Resolve a lemma to its word id, creating a stub row when absent."""

    async def claim_next_epoch(self, norm: str) -> GenerationFence:
        """Bump the generation epoch and return the resulting ownership token.

        The caller MUST commit before starting provider work — an uncommitted
        epoch is invisible to other workers and the fence stops fencing.
        """

    async def fence_is_current(self, fence: GenerationFence) -> bool:
        """Whether ``fence`` still owns its word, locking the row for the publish."""

    async def upsert_core(self, entry: "GeneratedEntry", cambridge_word_id: int | None) -> int:
        """Insert or update one unit's own columns and return its word id."""

    async def sync_aliases(self, word_id: int, aliases: Iterable["GeneratedAlias"]) -> None:
        """Replace a word's aliases; fully derived from generation."""

    async def link_related(self, word_id: int, related: Iterable["RelatedWord"]) -> None:
        """Ensure the word-level relation edges for one unit."""

    async def mark_done(self, word_id: int) -> None: ...

    async def record(self, word_id: int) -> WordRecord: ...

    async def records(self, word_ids: Sequence[int]) -> list[WordRecord]: ...

    async def done_keys(self) -> set[str]:
        """Every ``match_key`` already generated — the candidate diff."""

    async def delete(self, word_id: int) -> bool: ...

    async def listing(
        self, status: str = "done", limit: int | None = None, offset: int = 0
    ) -> list[WordListing]: ...

    async def seed_phrase_unit(
        self,
        phrase_title: str,
        host_display: str | None,
        entry_type: str | None,
        is_overlap: bool,
    ) -> None:
        """Create the stub word for a multi-word phrase and link it to its host."""


class SenseRepo(Protocol):
    """The ``senses`` aggregate: senses and everything hanging off one sense."""

    async def sync(
        self, word_id: int, senses: Iterable["GeneratedSense"], cefr_map: dict[str, str]
    ) -> None:
        """Replace a word's senses and their children."""

    async def word_id_for(self, sense_id: int) -> int:
        """Owning word id. Raises when the sense does not exist."""

    async def needing_embedding(
        self, model_name: str, word_ids: list[int] | None = None, limit: int | None = None
    ) -> list[SenseEmbeddingNeed]: ...

    async def store_embeddings(
        self, vectors: list[tuple[int, bytes]], model_name: str, dim: int
    ) -> int:
        """Write packed vectors by sense id; returns rows actually updated."""

    async def embedded(self, model_name: str) -> list[EmbeddedSenseRow]: ...

    async def pending_relations(
        self, batch_size: int, word_ids: list[int] | None = None
    ) -> list[ResolveTask]: ...

    async def apply_resolutions(
        self, decisions: Iterable[ResolveDecision]
    ) -> list[ResolveOutcome]: ...

    async def example_context(
        self, sense_id: int
    ) -> tuple["ExampleGenContext", list[str]] | None: ...

    async def append_examples(self, sense_id: int, texts: Sequence[str]) -> int: ...

    async def for_theming(self, word_id: int) -> list[ThemingSense]: ...


class ThemeRepo(Protocol):
    """The ``themes`` aggregate: themes and the themed overlay of a sense."""

    async def create(
        self,
        name: str,
        style_prompt: str,
        description: str | None = None,
        tone: str | None = None,
        key: str | None = None,
        overwrite: bool = False,
    ) -> ThemeRecord: ...

    async def list_all(self) -> list[ThemeRecord]: ...

    async def get(self, key: str) -> ThemeRecord | None: ...

    async def update(
        self,
        key: str,
        name: str | None = None,
        style_prompt: str | None = None,
        description: str | None = None,
        tone: str | None = None,
    ) -> ThemeRecord | None: ...

    async def delete(self, key: str) -> bool: ...

    async def resolve(self, key_or_id: str | int) -> tuple[int, str] | None:
        """``(theme_id, style_prompt)`` for a key or id, else ``None``."""

    async def persist_themed(
        self, theme_id: int, result: "ThemedResult", sense_ids: Sequence[int]
    ) -> None: ...

    async def overlay_for_word(
        self, word_id: int, theme_id: int
    ) -> dict[int, tuple[str, list[str]]]: ...

    async def overlay_for_sense(
        self, sense_id: int, theme_id: int
    ) -> tuple[int, list[str]] | None: ...

    async def append_themed_examples(self, themed_sense_id: int, texts: Sequence[str]) -> int: ...


class TagRepo(Protocol):
    """The ``tags`` aggregate: the LLM-authored topic vocabulary."""

    async def sync(self, word_id: int, topics: Iterable["GeneratedTopic"]) -> None: ...

    async def names(self) -> list[TagName]: ...

    async def usage(self) -> list[TagUsage]: ...

    async def words_for_key(self, key: str, limit: int | None = None) -> list[WordListing]: ...

    async def rename(self, tag: str, name: str | None = None, title: str | None = None) -> bool: ...

    async def delete(self, tag: str) -> bool: ...

    async def merge(self, sources: Sequence[str], into: str) -> int: ...


class StatsRepo(Protocol):
    """Cross-aggregate counts, deliberately read in one snapshot."""

    async def snapshot(self) -> "Stats": ...


class UnitOfWork(Protocol):
    """One session shared by the aggregate repositories, one commit boundary.

    Scope this to writes that must succeed or fail together. Steps that are
    best-effort by design (embedding, sense disambiguation, relation resolution)
    run AFTER the commit, so a failure there can never roll back published
    content.
    """

    words: WordRepo
    senses: SenseRepo
    themes: ThemeRepo
    tags: TagRepo
    stats: StatsRepo

    async def __aenter__(self) -> "UnitOfWork": ...

    async def __aexit__(self, exc_type, exc, tb) -> None: ...

    async def commit(self) -> None: ...

    async def rollback(self) -> None: ...

    async def flush(self) -> None: ...
