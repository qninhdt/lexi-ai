"""Controlled vocabularies — single source of truth.

Both the ORM models (Phase 2, app-level validation) and the LLM output schema
(Phase 4, strict enums) import these sets so the write path and the generation
path can never drift.
"""

# words.entry_type
ENTRY_TYPES = frozenset({"word", "phrasal_verb", "idiom", "phrase", "expression"})

# words.status
STATUSES = frozenset({"pending", "done", "error", "not_found"})

# senses.tier — ordering matters (core first). Used for sort in the read model.
TIERS = ("core", "common", "extended", "rare")
TIER_SET = frozenset(TIERS)
TIER_ORDER = {tier: i for i, tier in enumerate(TIERS)}

# word_aliases.type taxonomy (brainstorm §6.1).
ALIAS_TYPES = frozenset(
    {
        "spelling_uk",
        "spelling_us",
        "spelling_other",
        "hyphenation",
        "spacing",
        "capitalization",
        "diacritic",
        "abbreviation",
        "acronym",
        "clipping",
        "contraction",
        "preposition",
        "particle",
        "article",
        "optional_element",
        "word_choice",
        "placeholder_style",
        "number_slot",
    }
)

# word_aliases.dialect
DIALECTS = frozenset({"uk", "us"})

# entry_links.rel_type. ``word_family``/``confused_with`` are word-references:
# they NAME a lemma, so they ride the same normalized related[] → entry_links
# path as synonyms (match_key stub-rows + dedup), inheriting all of it for free.
REL_TYPES = frozenset(
    {
        "arrow_redirect",
        "another_word",
        "synonym",
        "antonym",
        "see_also",
        "variant_of",
        "part_of_phrasal_family",
        "word_family",
        "confused_with",
    }
)

# senses.grammar — countability / transitivity / complementation labels. Stored
# comma-joined in one column, so a label must NEVER contain a comma (the join
# separator) — guarded by a test. Each token is schema-validated before the join.
GRAMMAR_LABELS = frozenset(
    {
        "countable",
        "uncountable",
        "transitive",
        "intransitive",
        "verb + to-infinitive",
        "verb + -ing",
        "verb + that-clause",
        "usually singular",
        "usually plural",
        "before noun",
        "after verb",
    }
)

# senses.register — style / formality band.
REGISTERS = frozenset(
    {
        "formal",
        "informal",
        "neutral",
        "slang",
        "offensive",
        "humorous",
        "dated",
        "literary",
        "technical",
    }
)

# senses.connotation — affective polarity.
CONNOTATIONS = frozenset({"positive", "negative", "neutral"})

# sense_reference.source
REFERENCE_SOURCES = frozenset({"cambridge", "wordnet"})


def canonical_cambridge_ref(source_ref: str) -> str:
    """Canonicalize a Cambridge sense reference to its bare numeric id.

    The generation prompt shows senses labelled ``sense#42`` while the CEFR
    lookup map is keyed by the bare id ``"42"``. The model may echo either form
    as ``source_ref``. Both the write path (repository CEFR resolution) and the
    map builder (api) route through this function so the two sides can never
    disagree — otherwise the Cambridge-first CEFR rule silently falls through to
    the LLM value.
    """
    ref = source_ref.strip()
    lowered = ref.lower()
    if lowered.startswith("sense#"):
        ref = ref[len("sense#") :]
    elif lowered.startswith("sense"):
        ref = ref[len("sense") :]
    return ref.lstrip("#").strip()
