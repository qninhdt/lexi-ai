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

from dataclasses import dataclass, field
from typing import Literal
from warnings import filterwarnings

from pydantic import BaseModel, Field

from lexi_ai.constants import (
    ALIAS_TYPES,
    CONNOTATIONS,
    DIALECTS,
    ENTRY_TYPES,
    GRAMMAR_LABELS,
    INFLECTION_LABELS,
    POS_TAGS,
    REFERENCE_SOURCES,
    REGISTERS,
    SENSE_REL_TYPES,
    TIERS,
    WORD_REL_TYPES,
)

# Sorted tuples give deterministic enum ordering in the emitted JSON schema.
_TierLit = Literal[TIERS]
_EntryTypeLit = Literal[tuple(sorted(ENTRY_TYPES))]
_AliasTypeLit = Literal[tuple(sorted(ALIAS_TYPES))]
_DialectLit = Literal[tuple(sorted(DIALECTS))]
# Two derived enums pin each relation to its correct level so the LM cannot emit
# a sense-level rel at the entry level (or vice versa). Both come from REL_LEVEL
# via WORD_REL_TYPES/SENSE_REL_TYPES so they can never drift from the router.
_WordRelTypeLit = Literal[tuple(sorted(WORD_REL_TYPES))]
_SenseRelTypeLit = Literal[tuple(sorted(SENSE_REL_TYPES))]
_PosLit = Literal[tuple(sorted(POS_TAGS))]
_SourceLit = Literal[tuple(sorted(REFERENCE_SOURCES))]
_GrammarLit = Literal[tuple(sorted(GRAMMAR_LABELS))]
_RegisterLit = Literal[tuple(sorted(REGISTERS))]
_ConnotationLit = Literal[tuple(sorted(CONNOTATIONS))]
_InfLit = Literal[tuple(sorted(INFLECTION_LABELS))]

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


class GeneratedForm(BaseModel):
    """One inflected surface + its label. ``inf`` is a closed-vocab enum
    (INFLECTION_LABELS); ``surface`` is bounded free text sanitized on write."""

    inf: _InfLit = Field(description="Inflection label; must match the POS paradigm.")
    surface: str = Field(max_length=64, description="The inflected surface form.")


class GeneratedSenseRelation(BaseModel):
    """One SENSE-level relation emitted by a specific sense (Phase 3).

    Unlike ``RelatedWord`` (word-level), this carries a ``gloss`` describing the
    TARGET's intended meaning — the load-bearing signal the later WSD pass (Phase
    4) uses to pick the right target sense. ``gloss`` is REQUIRED + non-empty
    ([F12]); a blank one gets the whole edge skipped on the write path, never
    persisted as a dead row.
    """

    rel_type: _SenseRelTypeLit
    # Normalized to a real entry via match_key; bounded like RelatedWord.norm.
    norm: str = Field(max_length=128, description="Canonical form of the target word/phrase.")
    gloss: str = Field(
        min_length=1,
        max_length=255,
        description=(
            "A short gloss of the TARGET's intended meaning (which sense of the "
            "target this relation points to), e.g. for antonym 'dark' -> "
            "'lacking light'. Required — used to reconcile the exact target sense."
        ),
    )


class GeneratedSense(BaseModel):
    definition: str = Field(description="Learner-friendly definition.")
    tier: _TierLit = Field(description="core | common | extended | rare.")
    # [F2] REQUIRED per sense (no default): the WSD POS-filter (Phase 4) mass-marks
    # edges unresolvable when target senses carry no POS. Closed vocab (POS_TAGS)
    # hard-rejects out-of-vocab, and the prompt mandates emitting one per sense.
    pos: _PosLit = Field(description="Part of speech (required); one of the 12 allowed labels.")
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
    domain: str | None = Field(
        default=None,
        max_length=64,
        description=(
            "Subject-area / field label when the sense is domain-specific "
            "(computing, medicine, law, sport, music). Omit for everyday senses."
        ),
    )
    usage_note: str | None = Field(
        default=None,
        max_length=255,
        description=(
            "One short usage or confusable hint when helpful "
            "(e.g. \"often confused with 'affect'\"; \"usually in the passive\"). "
            "Omit when nothing notable."
        ),
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
    forms: list[GeneratedForm] = Field(
        default_factory=list,
        max_length=16,
        description=(
            "The COMPLETE inflection paradigm for this sense's headword, chosen by POS: "
            "verb -> base, past, past_participle, present_3sg, ing; "
            "noun -> plural; adjective/adverb -> comparative, superlative. "
            "Emit ONLY the forms valid for the POS; do NOT scrape from the examples "
            "(an example uses one form, but the paradigm must be whole). A label may "
            "repeat when a form has variants (dreamed / dreamt). Omit for invariant words."
        ),
    )
    # Sense-level semantic relations THIS sense emits (synonym/antonym/hypernym/
    # hyponym/meronym/holonym/see_also). Carried at the SENSE level (not entry) so
    # the write path knows WHICH meaning emitted each relation — the whole point of
    # sense-level relations. Word-level relations stay on GeneratedEntry.related.
    relations: list["GeneratedSenseRelation"] = Field(
        default_factory=list,
        description=(
            "Sense-level relations emitted by THIS specific sense. For each, give "
            "the target lemma (norm) and a SHORT gloss of the target's intended "
            "meaning so the system can later link it to the right target sense."
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
    # WORD-level rel_type only (word_family/confused_with/...): the sense-level
    # rels (synonym/antonym/hypernym/...) now live on GeneratedSense.relations, so
    # constraining this enum stops the LM emitting a sense-level rel at entry level.
    rel_type: _WordRelTypeLit


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
    # Word-level POS stays nullable: a headword may span several POS across its
    # senses, and it is NOT used for WSD (only per-sense pos is). Closed vocab.
    pos: _PosLit | None = None
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


class WsdCandidate(BaseModel):
    """One target-sense option shown to the WSD judge. ``index`` is the position
    in the task's ``candidates`` list (0-based) and is what the judge echoes back
    as ``chosen_index`` — the mapping index->sense_id is held server-side and NEVER
    trusted from the model ([F3])."""

    index: int
    definition: str


class WsdTask(BaseModel):
    """One reconciliation ask: given a source sense that emitted a relation with a
    ``gloss`` describing the intended target meaning, which candidate target sense
    (if any) does it point to? ``gloss``/``source_def`` are UNTRUSTED free text
    ([F14]) — the prompt fences them as data."""

    rel_type: str
    gloss: str
    source_def: str
    candidates: list[WsdCandidate]


class WsdChoice(BaseModel):
    """The judge's pick for one task. ``chosen_index`` is the candidate index the
    relation resolves to, or ``None`` when NO candidate fits the gloss (→ derived
    ``unresolvable``). The value is validated against the candidate bounds on apply
    ([F3]) — an out-of-range index is treated as ``None``, never indexed blindly."""

    chosen_index: int | None = Field(
        default=None,
        description=(
            "0-based index of the target sense this relation points to, or null if "
            "no candidate sense matches the gloss. Never guess — prefer null."
        ),
    )


class WsdBatch(BaseModel):
    """The judge's answers for a whole batch, order-aligned with the input tasks
    (``choices[i]`` answers ``tasks[i]``). Length must match the task count; a
    short/long list is a validation failure the caller treats as all-unresolvable
    for that batch."""

    choices: list[WsdChoice] = Field(default_factory=list)


class ExampleBatch(BaseModel):
    """Targeted example augmentation for ONE sense (neutral or themed).

    The same schema (and its ``<t inf>`` tag contract) is reused by both the
    neutral and themed ``add_examples`` paths so the two never drift — there is
    no themed-specific batch schema.
    """

    examples: list[str] = Field(
        default_factory=list,
        max_length=12,
        description=(
            "Fresh English example sentences for the given sense. For every "
            "sentence you MUST wrap the target word/phrase (or its inflected "
            'form) using <t inf="value">...</t> tags. Valid inf values are: '
            "base | past | past_participle | present_3sg | ing | plural | "
            "comparative | superlative. "
            'Example for \'glisten\': \'The snow <t inf="past">glistened</t> '
            "in the sun.' (same contract as GeneratedSense.examples)."
        ),
    )


@dataclass
class ExampleGenContext:
    """Internal carrier of the facts a targeted example generator needs for ONE
    sense: its definition/pos/guideword/tier plus the ``(inf, surface)`` paradigm.
    NOT a public read model — assembled in the repository, consumed by the
    generator."""

    definition: str
    pos: str | None
    guideword: str | None
    tier: str
    forms: list[tuple[str, str]] = field(default_factory=list)
