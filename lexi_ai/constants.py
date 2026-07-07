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

# entry_links.rel_type. ``word_family``/``confused_with``/``hypernym``/``hyponym``
# are word-references: they NAME a lemma, so they ride the same normalized
# related[] → entry_links path as synonyms (match_key stub-rows + dedup),
# inheriting all of it for free.
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
