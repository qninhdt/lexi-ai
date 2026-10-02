"""Word generation schemas and lexical/evidence validation."""

import unicodedata
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from lexi_ai.errors import InvalidOutputError
from lexi_ai.patterns import validate_pattern
from lexi_ai.references.cambridge import SourceEntry
from lexi_ai.references.wordnet import Synset
from lexi_ai.text import match_key, parse_marked_example, validate_lemma


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


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


class InventorySense(Strict):
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

    @model_validator(mode="after")
    def check_definition(self):
        if not self.definition.strip():
            raise ValueError("Sense definition must not be blank")
        return self


class SenseEnrichment(Strict):
    tier: Literal["core", "common", "less_common", "rare"]
    examples: list[str] = Field(description="Natural use; mark target with <t inf=...>...</t>")
    forms: list[FormOutput]
    patterns: list[str]
    collocations: list[str]
    relations: list[SenseRelationOutput]
    sources: list[str] = Field(description="Supplied local evidence IDs, e.g. c1, c2, w1")
    cefr_level: Literal["A1", "A2", "B1", "B2", "C1", "C2"]
    register_: Literal["formal", "informal", "slang", "literary", "specialist"] | None = Field(
        alias="register"
    )
    usage_note: str | None = None

    @model_validator(mode="after")
    def check(self):
        for content in self.examples:
            _, spans = parse_marked_example(content)
            if not spans:
                raise ValueError("example must mark the target")
        for pattern in self.patterns:
            validate_pattern(pattern)
        return self


class SenseOutput(InventorySense, SenseEnrichment):
    """Locally assembled fixed definition/POS and validated enrichment."""


class InventoryOutput(Strict):
    lemma: str = Field(description="One stable citation lemma for the selected lexical item")
    type: Literal["word", "phrasal_verb", "idiom", "phrase", "expression"]
    aliases: list[str]
    related: list[WordRelationOutput]
    senses: list[InventorySense] = Field(min_length=1)

    @field_validator("lemma")
    @classmethod
    def check_lemma(cls, value: str) -> str:
        return validate_lemma(value)

    @model_validator(mode="after")
    def check(self):
        for alias in self.aliases:
            validate_lemma(alias)
        keys = [
            (
                sense.pos,
                " ".join(
                    unicodedata.normalize(
                        "NFKC",
                        sense.definition,
                    ).split()
                ).casefold(),
            )
            for sense in self.senses
        ]
        if len(set(keys)) != len(keys):
            raise ValueError("exactly duplicate inventory Sense")
        return self


class WordOutput(InventoryOutput):
    """Complete content assembled in Python for publication, not an LLM task."""

    senses: list[SenseOutput] = Field(min_length=1)


def validate_inventory(generated: InventoryOutput, target: str) -> None:
    identities = {match_key(generated.lemma), *(match_key(a) for a in generated.aliases)}
    if match_key(validate_lemma(target)) not in identities:
        raise InvalidOutputError("lemma conflicts with supplied target")


def validate_evidence(
    generated: WordOutput,
    selected: SourceEntry,
    synsets: list[Synset],
    *,
    target: str,
) -> None:
    """An LLM may synthesize meanings, but cannot fabricate cited source rows."""
    try:
        validate_inventory(generated, target)
        allowed = {f"c{index}" for index in range(1, len(selected.senses) + 1)} | {
            f"w{index}" for index in range(1, len(synsets) + 1)
        }
        for sense in generated.senses:
            if len(set(sense.sources)) != len(sense.sources):
                raise ValueError("duplicate source reference")
            for ref_id in sense.sources:
                if ref_id not in allowed:
                    raise ValueError("reference was not supplied in source evidence")
    except ValueError as exc:
        raise InvalidOutputError(str(exc)) from exc
