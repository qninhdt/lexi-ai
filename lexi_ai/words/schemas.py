"""Word generation schemas and lexical/evidence validation."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from lexi_ai.errors import InvalidOutputError
from lexi_ai.patterns import validate_pattern
from lexi_ai.references.cambridge import SourceEntry
from lexi_ai.references.wordnet import Synset
from lexi_ai.text import match_key, parse_marked_example, validate_lemma


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReferenceOutput(Strict):
    source: Literal["cambridge", "wordnet"]
    source_ref: str = Field(
        description="Exact Cambridge sense ID or WordNet synset key supplied in the input evidence."
    )


class FormOutput(Strict):
    surface: str
    inf: Literal[
        "base",
        "past",
        "past_participle",
        "present_3sg",
        "ing",
        "plural",
        "comparative",
        "superlative",
    ]


class WordRelationOutput(Strict):
    lemma: str
    rel_type: Literal["word_family", "confused_with"]

    @model_validator(mode="after")
    def check(self):
        validate_lemma(self.lemma)
        return self


class SenseRelationOutput(Strict):
    lemma: str = Field(description="Stable citation lemma of the related lexical item.")
    rel_type: Literal["synonym", "antonym", "hypernym", "hyponym", "meronym", "holonym"]
    gloss: str = Field(
        description="Short, discriminative meaning of the target lemma, not the source Sense."
    )

    @model_validator(mode="after")
    def check(self):
        validate_lemma(self.lemma)
        return self


class SenseOutput(Strict):
    definition: str = Field(
        min_length=1, max_length=16000, description="The single learner definition for this Sense"
    )
    pos: Literal[
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
    ]
    tier: Literal["core", "common", "extended", "rare"]
    examples: list[str] = Field(description="Natural use; mark target with <t inf=...>...</t>")
    forms: list[FormOutput]
    patterns: list[str]
    collocations: list[str]
    relations: list[SenseRelationOutput]
    references: list[ReferenceOutput]
    cefr_level: Literal["A1", "A2", "B1", "B2", "C1", "C2"] | None = None
    register_: str | None = Field(default=None, alias="register")
    usage_note: str | None = None
    ipa_uk: str | None = None
    ipa_us: str | None = None

    @model_validator(mode="after")
    def check(self):
        if not self.definition.strip():
            raise ValueError("Sense definition must not be blank")
        for content in self.examples:
            _, spans = parse_marked_example(content)
            if not spans:
                raise ValueError("example must mark the target")
        for pattern in self.patterns:
            validate_pattern(pattern)
        return self


class WordOutput(Strict):
    lemma: str = Field(description="One stable citation lemma for the selected lexical item")
    entry_type: Literal["word", "phrasal_verb", "idiom", "phrase", "expression"]
    aliases: list[str]
    related: list[WordRelationOutput]
    senses: list[SenseOutput] = Field(description="One or more senses of this same Word")

    @field_validator("lemma")
    @classmethod
    def check_lemma(cls, value: str) -> str:
        return validate_lemma(value)

    @model_validator(mode="after")
    def check(self):
        if not self.senses:
            raise ValueError("Word must contain at least one Sense")
        for alias in self.aliases:
            validate_lemma(alias)
        return self


def validate_evidence(
    generated: WordOutput,
    selected: SourceEntry,
    synsets: list[Synset],
) -> None:
    """An LLM may synthesize meanings, but cannot fabricate cited source rows."""
    try:
        if generated.entry_type != selected.entry_type:
            raise ValueError("entry type conflicts with selected source")
        # An unambiguous citation anchor must not be replaced by an unrelated item.
        if (
            match_key(selected.slug) != match_key(generated.lemma)
            and match_key(selected.display) != match_key(generated.lemma)
            and not any(
                token in selected.display for token in ("/", "(", ")", "someone", "something")
            )
        ):
            raise ValueError("lemma conflicts with selected source")
        allowed = {("cambridge", str(sense.id)) for sense in selected.senses} | {
            ("wordnet", item.key) for item in synsets
        }
        for sense in generated.senses:
            for ref in sense.references:
                ref_id = ref.source_ref
                if ref.source == "cambridge" and ref_id.lower().startswith("sense#"):
                    ref_id = ref_id[6:]
                if (ref.source, ref_id) not in allowed:
                    raise ValueError("reference was not supplied in source evidence")
    except ValueError as exc:
        raise InvalidOutputError(str(exc)) from exc
