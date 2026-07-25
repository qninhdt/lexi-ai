"""Domain records returned across the persistence boundary.

Repositories return these, never SQLAlchemy entities. Keeping the boundary types
detached means a caller can read a result after its transaction closed without
depending on ``expire_on_commit=False`` or risking a lazy-load outside greenlet
context.

The row-shaped types are ``NamedTuple``s on purpose: they name their fields while
staying positionally unpackable, which is how query results are consumed today.
"""

from dataclasses import dataclass
from typing import NamedTuple


@dataclass(frozen=True)
class WordRecord:
    """A persisted word, detached from its session."""

    id: int
    match_key: str
    norm: str
    entry_type: str | None
    pos: str | None
    status: str
    error_msg: str | None = None


@dataclass(frozen=True)
class ThemeRecord:
    """A persisted theme, detached from its session.

    ``key`` is the durable ``theme_key`` identity; it is immutable once created,
    so renaming ``name`` never re-keys the theme.
    """

    id: int
    key: str
    name: str
    style_prompt: str
    description: str | None = None
    tone: str | None = None


@dataclass(frozen=True)
class GenerationFence:
    """Database-issued ownership token for one destructive word replacement."""

    match_key: str
    epoch: int


class WordListing(NamedTuple):
    """One row of a dictionary browse."""

    word_id: int
    norm: str
    entry_type: str | None


class WordMatch(NamedTuple):
    """A word resolved from a lookup key, with its lifecycle status."""

    word_id: int
    status: str


class TagName(NamedTuple):
    """Existing topic vocab, for prompt injection."""

    name: str
    title: str


class TagUsage(NamedTuple):
    """A topic with its live member count."""

    name: str
    title: str
    count: int


class SenseEmbeddingNeed(NamedTuple):
    """A sense missing a current-model vector, with the text to embed."""

    sense_id: int
    norm: str
    definition: str


class ThemingSense(NamedTuple):
    """One neutral sense as the theming prompt sees it."""

    sense_id: int
    definition: str
    pos: str | None
    guideword: str | None
    tier: str


class EmbeddedSenseRow(NamedTuple):
    """A done sense + its current-model vector, for semantic ranking."""

    sense_id: int
    word_id: int
    norm: str
    entry_type: str | None
    definition: str
    tier: str
    embedding: bytes


class ResolveCandidate(NamedTuple):
    """One target-sense option for a WSD task: the DB sense id + the facts the
    judge sees (``pos`` for the POS filter, ``definition`` shown in the prompt)."""

    sense_id: int
    pos: str | None
    definition: str


class ResolveTask(NamedTuple):
    """One pending edge lifted off the queue, ready for the judge. ``edge_id`` is
    the ``sense_relation`` row; ``candidates`` are its target word's senses
    (POS-filtered later). ``source_pos`` drives the POS filter; ``gloss`` +
    ``source_def`` are the (untrusted) prompt text."""

    edge_id: int
    rel_type: str
    gloss: str
    source_def: str
    source_pos: str | None
    candidates: list[ResolveCandidate]


class ResolveDecision(NamedTuple):
    """The bounds-validated verdict for one edge, ready to apply. ``to_sense_id``
    None means mark unresolvable; else resolve to that sense with the stamped
    ``target_hash``."""

    edge_id: int
    to_sense_id: int | None
    target_hash: str | None


class ResolveOutcome(NamedTuple):
    """One edge's WSD result.

    ``state`` is the DERIVED post-resolve state: ``resolved`` when a target sense
    was chosen, ``unresolvable`` when the judge returned none, ``noop`` when a
    racing regenerate made the conditional write a no-op, ``error`` when a
    poison-pill isolated the edge. Order-aligned with the input decisions.
    """

    edge_id: int
    state: str
    error: str | None = None
