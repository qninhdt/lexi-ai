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

# senses.pos — closed controlled vocab (Phase 1 of sense-level relations). The
# 12 learner-facing part-of-speech labels. Closing this (previously free
# ``String(32)``) is the foundation for the WSD POS-filter: a target sense whose
# POS is a drifted variant ("adj" vs "adjective") would be filtered wrong. The
# generation schema builds a strict enum from this set (required per sense), and
# ``normalize_pos`` folds legacy/loose variants for the WSD matcher and for
# reading pre-vocab rows. Stored as ``String(32)`` (no native enum — portable).
POS_TAGS = frozenset(
    {
        "noun",
        "verb",
        "adjective",
        "adverb",
        "pronoun",
        "preposition",
        "conjunction",
        "determiner",
        "interjection",
        "numeral",
        "article",
        "auxiliary",
    }
)

# Loose/legacy surface forms -> canonical POS label. Applied on BOTH sides of the
# WSD POS comparison (source POS is already closed vocab; target POS may be older
# drifted data like "adj"/"v."/NULL) so ``NULL == NULL`` (false in SQL) and
# variant spellings never mis-filter candidates. Returns ``None`` for anything
# unmappable — never guesses.
_POS_ALIASES = {
    "n": "noun",
    "n.": "noun",
    "v": "verb",
    "v.": "verb",
    "adj": "adjective",
    "adj.": "adjective",
    "a": "adjective",
    "adv": "adverb",
    "adv.": "adverb",
    "pron": "pronoun",
    "pron.": "pronoun",
    "prep": "preposition",
    "prep.": "preposition",
    "conj": "conjunction",
    "conj.": "conjunction",
    "det": "determiner",
    "det.": "determiner",
    "interj": "interjection",
    "interj.": "interjection",
    "int": "interjection",
    "num": "numeral",
    "num.": "numeral",
    "art": "article",
    "art.": "article",
    "aux": "auxiliary",
    "aux.": "auxiliary",
    "modal": "auxiliary",
}


def normalize_pos(raw: str | None) -> str | None:
    """Fold a loose/legacy POS surface form to a canonical :data:`POS_TAGS` label.

    Pure function: lowercase + strip, accept an exact vocab label, else map a
    known alias (``adj`` -> ``adjective``, ``N`` -> ``noun``). Returns ``None``
    for empty / unmappable input — it NEVER guesses (an unknown POS is treated as
    "unknown", not force-fit to a tag). Used by the WSD POS-filter (Phase 4, both
    source and target side) and as a safety net when reading pre-vocab rows.
    """
    if not raw:
        return None
    token = raw.strip().lower()
    if not token:
        return None
    if token in POS_TAGS:
        return token
    return _POS_ALIASES.get(token)


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

# relation rel_type — the full vocabulary. Each type is routed to ONE table by
# ``REL_LEVEL`` below: word-level types ride the ``related[]`` → ``word_relation``
# path (match_key stub-rows + dedup), sense-level types become ``sense_relation``
# half-edges emitted per SENSE and later WSD-resolved (Phase 3/4).
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
        "hypernym",
        "hyponym",
        "meronym",
        "holonym",
    }
)

# The routing table (single source of truth): each ``rel_type`` maps to the level
# of the relation, ``"word"`` (→ ``word_relation``, no sense on either end, no WSD)
# or ``"sense"`` (→ ``sense_relation``, sense-DEPENDENT, WSD-resolved). The two
# generation enums (`_WordRelTypeLit`/`_SenseRelTypeLit`) and the persist router
# are all derived from THIS map so a rel_type can never be mis-levelled. A test
# asserts every REL_TYPES member appears here (no orphan level).
REL_LEVEL = {
    # word-level → word_relation
    "word_family": "word",
    "confused_with": "word",
    "variant_of": "word",
    "arrow_redirect": "word",
    "another_word": "word",
    "part_of_phrasal_family": "word",
    # sense-level → sense_relation
    "synonym": "sense",
    "antonym": "sense",
    "hypernym": "sense",
    "hyponym": "sense",
    "meronym": "sense",
    "holonym": "sense",
    "see_also": "sense",
}

WORD_REL_TYPES = frozenset(rt for rt, level in REL_LEVEL.items() if level == "word")
SENSE_REL_TYPES = frozenset(rt for rt, level in REL_LEVEL.items() if level == "sense")

# Direction semantics of each SENSE-level rel_type, used by the READ model (Phase
# 6) to surface inverse/symmetric edges from the far side WITHOUT ever mutating a
# row ([F8] — canonicalize-on-read, never on-write). Values:
#   "symmetric"      — the relation reads the same both ways (synonym↔synonym).
#   "inverse:<other>" — reading from the target flips the label (hypernym seen
#                       from the target is a hyponym, and vice versa).
# Only sense-level types appear (word-level relations are not WSD-resolved and are
# surfaced as emitted). A test asserts every SENSE_REL_TYPES member is classified
# and that every ``inverse:<other>`` names a real, mutually-inverse sense type.
REL_SYMMETRY = {
    "synonym": "symmetric",
    "antonym": "symmetric",
    "see_also": "symmetric",
    "hypernym": "inverse:hyponym",
    "hyponym": "inverse:hypernym",
    "meronym": "inverse:holonym",
    "holonym": "inverse:meronym",
}

# WSD batch cost guards ([F9]) — caller/data-controlled sizes are DoS vectors, so
# both are hard-clamped in the resolve path (never merely defaulted):
#   WSD_BATCH_CEIL   — max edges reconciled per resolve_relations() call.
#   WSD_CANDIDATE_CAP — max target senses shown to the judge per task (top-K by
#                       sense_order); keeps a pathological many-sense word from
#                       ballooning one prompt.
WSD_BATCH_CEIL = 50
WSD_CANDIDATE_CAP = 12

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

# assets.kind — reference-addressed derived-asset cache (translation, TTS).
ASSET_KINDS = frozenset({"translate", "tts"})

# assets.source_kind — the source row a cached asset derives from. Each kind maps
# to a (table, text column) in the asset repository's resolver; that mapping is
# driven by THIS set so a kind can never be half-wired (a test asserts every
# member has a resolver entry). Ship 3 kinds; themed kinds are added only when a
# themed-translation consumer lands.
SOURCE_KINDS = frozenset({"sense_def", "example", "collocation"})

# --- questions subsystem vocabularies -------------------------------------
#
# Three axes wired through ``answer_kind``: a format DECLARES an answer_kind, a
# generator PRODUCES a question, a scorer GRADES it — dispatched by answer_kind,
# never by which backend produced the question. Adding a format = one id here +
# one registry row.

# questions.format (v1 seed set).
QUESTION_FORMATS = frozenset(
    {
        "definition_mcq",
        "cloze",
        "contextual_mcq",
        "use_in_sentence",
        "matching",
        "listening",
        "spelling",
        "pronunciation_mcq",
        "collocation_fill",
    }
)

# questions.answer_kind — the coupling contract a scorer dispatches on.
ANSWER_KINDS = frozenset({"single_choice", "text_span", "free_text", "matching"})

# senses.forms inflection labels. The generation prompt marks example targets and
# emits a per-sense form table using EXACTLY these labels; the write path and the
# schema import this one set so the two can never drift (like GRAMMAR_LABELS).
INFLECTION_LABELS = frozenset(
    {
        "base",
        "past",
        "past_participle",
        "present_3sg",
        "ing",
        "plural",
        "comparative",
        "superlative",
    }
)

# Score.kind — how a verdict was reached (deterministic rule vs llm judge). This
# is caller-facing OUTPUT the scorer reports about itself; the engine never reads
# it (the engine is interface-driven and does not branch on backend identity).
SCORE_KINDS = frozenset({"rule", "llm"})

# ISO 639-1 language codes mapped to their English names
TRANSLATION_LANGUAGES = {
    "aa": "Afar",
    "ab": "Abkhaz",
    "ae": "Avestan",
    "af": "Afrikaans",
    "ak": "Akan",
    "am": "Amharic",
    "an": "Aragonese",
    "ar": "Arabic",
    "as": "Assamese",
    "av": "Avaric",
    "ay": "Aymara",
    "az": "Azerbaijani",
    "ba": "Bashkir",
    "be": "Belarusian",
    "bg": "Bulgarian",
    "bh": "Bihari",
    "bi": "Bislama",
    "bm": "Bambara",
    "bn": "Bengali, Bangla",
    "bo": "Tibetan Standard, Tibetan, Central",
    "br": "Breton",
    "bs": "Bosnian",
    "ca": "Catalan",
    "ce": "Chechen",
    "ch": "Chamorro",
    "co": "Corsican",
    "cr": "Cree",
    "cs": "Czech",
    "cu": "Old Church Slavonic, Church Slavonic, Old Bulgarian",
    "cv": "Chuvash",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "dv": "Divehi, Dhivehi, Maldivian",
    "dz": "Dzongkha",
    "ee": "Ewe",
    "el": "Greek (modern)",
    "en": "English",
    "eo": "Esperanto",
    "es": "Spanish",
    "et": "Estonian",
    "eu": "Basque",
    "fa": "Persian (Farsi)",
    "ff": "Fula, Fulah, Pulaar, Pular",
    "fi": "Finnish",
    "fj": "Fijian",
    "fo": "Faroese",
    "fr": "French",
    "fy": "Western Frisian",
    "ga": "Irish",
    "gd": "Scottish Gaelic, Gaelic",
    "gl": "Galician",
    "gn": "Guaraní",
    "gu": "Gujarati",
    "gv": "Manx",
    "ha": "Hausa",
    "he": "Hebrew (modern)",
    "hi": "Hindi",
    "ho": "Hiri Motu",
    "hr": "Croatian",
    "ht": "Haitian, Haitian Creole",
    "hu": "Hungarian",
    "hy": "Armenian",
    "hz": "Herero",
    "ia": "Interlingua",
    "id": "Indonesian",
    "ie": "Interlingue",
    "ig": "Igbo",
    "ii": "Nuosu",
    "ik": "Inupiaq",
    "io": "Ido",
    "is": "Icelandic",
    "it": "Italian",
    "iu": "Inuktitut",
    "ja": "Japanese",
    "jv": "Javanese",
    "ka": "Georgian",
    "kg": "Kongo",
    "ki": "Kikuyu, Gikuyu",
    "kj": "Kwanyama, Kuanyama",
    "kk": "Kazakh",
    "kl": "Kalaallisut, Greenlandic",
    "km": "Khmer",
    "kn": "Kannada",
    "ko": "Korean",
    "kr": "Kanuri",
    "ks": "Kashmiri",
    "ku": "Kurdish",
    "kv": "Komi",
    "kw": "Cornish",
    "ky": "Kyrgyz",
    "la": "Latin",
    "lb": "Luxembourgish, Letzeburgesch",
    "lg": "Ganda",
    "li": "Limburgish, Limburgan, Limburger",
    "ln": "Lingala",
    "lo": "Lao",
    "lt": "Lithuanian",
    "lu": "Luba-Katanga",
    "lv": "Latvian",
    "mg": "Malagasy",
    "mh": "Marshallese",
    "mi": "Māori",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mn": "Mongolian",
    "mr": "Marathi (Marāṭhī)",
    "ms": "Malay",
    "mt": "Maltese",
    "my": "Burmese",
    "na": "Nauruan",
    "nb": "Norwegian Bokmål",
    "nd": "Northern Ndebele",
    "ne": "Nepali",
    "ng": "Ndonga",
    "nl": "Dutch",
    "nn": "Norwegian Nynorsk",
    "no": "Norwegian",
    "nr": "Southern Ndebele",
    "nv": "Navajo, Navaho",
    "ny": "Chichewa, Chewa, Nyanja",
    "oc": "Occitan",
    "oj": "Ojibwe, Ojibwa",
    "om": "Oromo",
    "or": "Oriya",
    "os": "Ossetian, Ossetic",
    "pa": "(Eastern) Punjabi",
    "pi": "Pāli",
    "pl": "Polish",
    "ps": "Pashto, Pushto",
    "pt": "Portuguese",
    "qu": "Quechua",
    "rm": "Romansh",
    "rn": "Kirundi",
    "ro": "Romanian",
    "ru": "Russian",
    "rw": "Kinyarwanda",
    "sa": "Sanskrit (Saṁskṛta)",
    "sc": "Sardinian",
    "sd": "Sindhi",
    "se": "Northern Sami",
    "sg": "Sango",
    "si": "Sinhalese, Sinhala",
    "sk": "Slovak",
    "sl": "Slovene",
    "sm": "Samoan",
    "sn": "Shona",
    "so": "Somali",
    "sq": "Albanian",
    "sr": "Serbian",
    "ss": "Swati",
    "st": "Southern Sotho",
    "su": "Sundanese",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "tg": "Tajik",
    "th": "Thai",
    "ti": "Tigrinya",
    "tk": "Turkmen",
    "tl": "Tagalog",
    "tn": "Tswana",
    "to": "Tonga (Tonga Islands)",
    "tr": "Turkish",
    "ts": "Tsonga",
    "tt": "Tatar",
    "tw": "Twi",
    "ty": "Tahitian",
    "ug": "Uyghur",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "uz": "Uzbek",
    "ve": "Venda",
    "vi": "Vietnamese",
    "vo": "Volapük",
    "wa": "Walloon",
    "wo": "Wolof",
    "xh": "Xhosa",
    "yi": "Yiddish",
    "yo": "Yoruba",
    "za": "Zhuang, Chuang",
    "zh": "Chinese",
    "zu": "Zulu",
}


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
