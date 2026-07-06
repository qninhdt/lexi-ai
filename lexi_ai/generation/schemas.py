"""Pydantic structured-output schema for LLM generation (Phase 4).

Enums are built from :mod:`lexi_ai.constants` (single source of truth shared
with the Phase 2 ORM validation) so the generation path and the write path can
never drift. ``Literal[<tuple>]`` unpacks each vocabulary into a JSON-schema
enum, which steers the model and hard-rejects out-of-vocab values at validation.

``GeneratedResult.units`` is a list: length 1 normally, length N when a Cambridge
page bundles genuinely independent lemmas (decision #16 — splitting is a
correctness requirement). Same-meaning surface variants must NOT split; they
become ``aliases`` on one unit (decision #17, enforced by the prompt).
"""

# ``Literal[tuple(sorted(<frozenset>))]`` builds each enum from the constants
# vocab at import time — runtime-correct (validated below) but Pyright can't model
# a Literal whose members come from a variable, so it flags every field annotated
# with one (reportInvalidTypeForm) at the USAGE site, not the definition. A
# per-line ignore isn't possible without annotating each field, so this ONE rule
# is disabled file-wide. Tradeoff: a genuine reportInvalidTypeForm elsewhere in
# this file is masked; every OTHER Pyright rule (reportReturnType,
# reportGeneralTypeIssues, ...) still surfaces, so type-checking is not blind here.
# pyright: reportInvalidTypeForm=false

from typing import Literal
from warnings import filterwarnings

from pydantic import BaseModel, Field

from lexi_ai.constants import (
    ALIAS_TYPES,
    CONNOTATIONS,
    DIALECTS,
    ENTRY_TYPES,
    GRAMMAR_LABELS,
    REFERENCE_SOURCES,
    REGISTERS,
    REL_TYPES,
    TIERS,
)

# Sorted tuples give deterministic enum ordering in the emitted JSON schema.
_TierLit = Literal[TIERS]
_EntryTypeLit = Literal[tuple(sorted(ENTRY_TYPES))]
_AliasTypeLit = Literal[tuple(sorted(ALIAS_TYPES))]
_DialectLit = Literal[tuple(sorted(DIALECTS))]
_RelTypeLit = Literal[tuple(sorted(REL_TYPES))]
_SourceLit = Literal[tuple(sorted(REFERENCE_SOURCES))]
_GrammarLit = Literal[tuple(sorted(GRAMMAR_LABELS))]
_RegisterLit = Literal[tuple(sorted(REGISTERS))]
_ConnotationLit = Literal[tuple(sorted(CONNOTATIONS))]

# ``GeneratedSense.register`` mirrors the ``senses.register`` column and
# ``SenseView.register`` — renaming it to dodge the shadow would drift those
# three apart. The shadowed name is ``abc.ABCMeta.register`` (virtual-subclass
# registration), never used on a value object, so the collision is benign;
# silence just this one field's warning (the message pins it to ``register``).
filterwarnings(
    "ignore",
    message=r'Field name "register" .* shadows an attribute in parent "BaseModel"',
    category=UserWarning,
)


class GeneratedReference(BaseModel):
    source: _SourceLit
    source_ref: str = Field(
        description=(
            "The source sense/synset this sense maps to. For Cambridge use the "
            "bare numeric sense id shown after 'sense#' (e.g. '42'); for WordNet "
            "use the synset key (e.g. 'book.n.01')."
        )
    )


class GeneratedSense(BaseModel):
    definition: str = Field(description="Learner-friendly definition.")
    tier: _TierLit = Field(description="core | common | extended | rare.")
    pos: str | None = Field(default=None, description="Part of speech.")
    cefr_level: str | None = Field(
        default=None, description="A1..C2; prefer the Cambridge value when mapped."
    )
    examples: list[str] = Field(
        default_factory=list,
        description=(
            "1-3 natural English example sentences illustrating this sense. "
            "For every sentence, you MUST wrap the target word/phrase (or its inflected forms) "
            "using <t inf=\"value\">...</t> tags. Valid inf values are: "
            "base | past | past_participle | present_3sg | ing | plural | comparative | superlative. "
            "Example for 'glisten': 'The snow <t inf=\"past\">glistened</t> in the sun.'"
        ),
    )
    references: list[GeneratedReference] = Field(
        default_factory=list,
        description="N-N provenance to source senses/synsets; may be empty.",
    )
    # Learner-dictionary enrichments — all best-effort (defaulted), so a sense
    # with none of them still validates. Enum fields hard-reject out-of-vocab
    # values (same posture as tier/entry_type); free-text fields are bounded and
    # additionally control-char sanitized on the write path.
    guideword: str | None = Field(
        default=None,
        max_length=64,
        description=(
            "Short 1-3 word sense label disambiguating homographs "
            "(bank -> MONEY vs RIVER). Synthesize a concise label."
        ),
    )
    grammar: list[_GrammarLit] = Field(
        default_factory=list,
        max_length=3,
        description="0-3 grammar labels from the allowed set; only when clearly warranted.",
    )
    register: _RegisterLit | None = Field(
        default=None, description="Style/formality band; omit when plainly neutral."
    )
    connotation: _ConnotationLit | None = Field(
        default=None, description="Affective polarity: positive | negative | neutral."
    )
    collocations: list[str] = Field(
        default_factory=list,
        max_length=12,
        description=(
            "2-6 high-frequency partner phrases showing the word in use "
            "(make a decision, heavy rain) — each an illustrative phrase, not a headword."
        ),
    )
    # IPA is hard-anchored from Cambridge in the prompt (per POS). COPY the anchored
    # value verbatim when shown; generate it ONLY for out-of-Cambridge words
    # (neologisms, proper nouns) where no anchor exists — LLMs hallucinate IPA badly.
    ipa_uk: str | None = Field(
        default=None,
        max_length=64,
        description=(
            "UK IPA pronunciation (e.g. /ˈbʊk/). COPY the anchored 'ipa: UK ...' value "
            "when the prompt shows one; only synthesize when Cambridge lacks it."
        ),
    )
    ipa_us: str | None = Field(
        default=None,
        max_length=64,
        description=(
            "US IPA pronunciation. COPY the anchored 'ipa: ... US ...' value when the "
            "prompt shows one; only synthesize when Cambridge lacks it."
        ),
    )


class GeneratedAlias(BaseModel):
    # Bounded like RelatedWord.norm — alias_norm also feeds match_key ->
    # words.alias_match_key String(512); a sanity cap keeping absurd input out.
    alias_norm: str = Field(
        max_length=128, description="Canonical surface variant with brace placeholders."
    )
    type: _AliasTypeLit
    dialect: _DialectLit | None = None


class RelatedWord(BaseModel):
    # ``norm`` is normalized to a real entry via match_key -> words.match_key
    # String(512). A sanity cap that rejects only pathological lengths (a >128-char
    # canonical lemma is already junk — longest real idioms sit well under it),
    # keeping absurd input out of the match_key path. NOT a hard overflow proof:
    # adversarial NFKD-expanding Unicode can still exceed 512 (a caught error +
    # status='error' on Postgres, never corruption). Also bounds the existing
    # synonym/antonym related[] path, not just the new word-references.
    norm: str = Field(max_length=128, description="Canonical form of the related word/phrase.")
    rel_type: _RelTypeLit


class GeneratedTopic(BaseModel):
    """One open-vocabulary topic tag. Free strings (no enum) — consistency is
    steered by the prompt and enforced deterministically in the repository via
    ``tag_key``. ``max_length`` bounds are HARD: they stop a hallucinated giant
    tag from reaching the DB to crash Postgres or bloat the vocab."""

    tag: str = Field(
        max_length=64,
        description=(
            "Short topic slug: lowercase, singular noun, 1-2 words "
            "(e.g. 'business', 'food', 'social media')."
        ),
    )
    title: str = Field(
        max_length=128,
        description="Human-friendly display title in Title Case (e.g. 'Business & Finance').",
    )


class GeneratedEntry(BaseModel):
    """One ``words`` row."""

    # Bounded like RelatedWord.norm — the headword feeds match_key ->
    # words.match_key String(512); a sanity cap keeping absurd input out.
    norm: str = Field(
        max_length=128,
        description="Canonical headword; placeholders as {sb}/{sth}/{one's}/{oneself}.",
    )
    entry_type: _EntryTypeLit
    pos: str | None = None
    senses: list[GeneratedSense] = Field(min_length=1)
    aliases: list[GeneratedAlias] = Field(default_factory=list)
    related: list[RelatedWord] = Field(default_factory=list)
    topics: list[GeneratedTopic] = Field(
        default_factory=list,
        max_length=5,
        description=(
            "1-3 broad topic tags for this word. REUSE an existing topic when one "
            "fits; only invent a new one when none matches."
        ),
    )


class GeneratedResult(BaseModel):
    units: list[GeneratedEntry] = Field(
        min_length=1,
        description=(
            "One entry normally; multiple only when Cambridge bundled "
            "independent lemmas with different meanings on one page."
        ),
    )
