"""Word generation schemas and lexical/evidence validation."""

from enum import Enum
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from lexi_ai.errors import InvalidOutputError
from lexi_ai.patterns import validate_pattern
from lexi_ai.references.cambridge import SourceEntry
from lexi_ai.references.wordnet import Synset
from lexi_ai.text import (
    answer_key,
    canonical_markup,
    match_key,
    parse_marked_example,
    strip_markup,
    validate_lemma,
)
from lexi_ai.vocab import (
    CEFRLevel,
    EntryType,
    Inflection,
    PartOfSpeech,
    Register,
    SenseRelationType,
    Tier,
    WordRelationType,
)


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @field_validator("*", mode="before")
    @classmethod
    def normalize_label(cls, value, info):
        annotation = cls.model_fields[info.field_name].annotation
        choices = [annotation, *get_args(annotation)]
        if isinstance(value, str):
            for choice in choices:
                members = (
                    list(choice)
                    if isinstance(choice, type) and issubclass(choice, Enum)
                    else [choice]
                )
                for member in members:
                    if isinstance(member, Enum) and value.casefold() == member.value.casefold():
                        return member.value
        return value


class FormOutput(Strict):
    surface: str
    inf: Inflection


class WordRelationOutput(Strict):
    lemma: str
    rel_type: Literal[WordRelationType.WORD_FAMILY, WordRelationType.CONFUSED_WITH]

    @model_validator(mode="after")
    def check(self):
        validate_lemma(self.lemma)
        return self


class SenseRelationOutput(Strict):
    lemma: str = Field(description="Stable citation lemma of the related lexical item.")
    rel_type: SenseRelationType
    gloss: str = Field(
        description="Short, discriminative meaning of the target lemma, not the source Sense."
    )

    @model_validator(mode="after")
    def check(self):
        validate_lemma(self.lemma)
        if parse_marked_example(self.gloss)[1]:
            raise ValueError("relation glosses must be plain text")
        return self


class InventorySense(Strict):
    definition: str = Field(
        min_length=1, max_length=16000, description="The single learner definition for this Sense"
    )
    pos: PartOfSpeech
    references: list[str] = Field(min_length=1, description="Supporting dictionary entry IDs")

    @model_validator(mode="after")
    def check_definition(self):
        if not self.definition.strip():
            raise ValueError("Sense definition must not be blank")
        return self


class SenseEnrichment(Strict):
    tier: Tier
    examples: list[str] = Field(
        min_length=1, description="Natural use; mark target with <t inf=...>...</t>"
    )
    forms: list[FormOutput]
    patterns: list[str]
    collocations: list[str]
    relations: list[SenseRelationOutput]
    cefr_level: CEFRLevel
    register_: Register | None = Field(alias="register")
    usage_note: str | None = None

    @field_validator("examples", mode="before")
    @classmethod
    def usable_examples(cls, values):
        if not isinstance(values, list):
            return values
        accepted, seen = [], set()
        for content in values:
            try:
                content = canonical_markup(content)
                if not parse_marked_example(content)[1]:
                    continue
                key = answer_key(strip_markup(content))
            except ValueError:
                continue
            if key not in seen:
                accepted.append(content)
                seen.add(key)
        if not accepted:
            raise ValueError("Sense has no usable examples")
        return accepted

    @model_validator(mode="after")
    def check(self):
        for pattern in self.patterns:
            validate_pattern(pattern)
        for text in [*self.collocations, self.usage_note or ""]:
            if parse_marked_example(text)[1]:
                raise ValueError("collocations and usage notes must be plain text")
        return self


class SenseOutput(InventorySense, SenseEnrichment):
    """Locally assembled fixed definition/POS and validated enrichment."""


class InventoryOutput(Strict):
    lemma: str = Field(description="One stable citation lemma for the selected lexical item")
    type: EntryType
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
        keys = [(sense.pos, answer_key(sense.definition)) for sense in self.senses]
        if len(set(keys)) != len(keys):
            raise ValueError("exactly duplicate inventory Sense")
        return self


class WordOutput(InventoryOutput):
    """Complete content assembled in Python for publication, not an LLM task."""

    senses: list[SenseOutput] = Field(min_length=1)


def validate_inventory(generated: InventoryOutput, allowed: set[str], target: str) -> None:
    identities = {match_key(generated.lemma), *(match_key(a) for a in generated.aliases)}
    if match_key(validate_lemma(target)) not in identities:
        raise InvalidOutputError("lemma conflicts with supplied target")
    for sense in generated.senses:
        if not sense.references:
            raise InvalidOutputError("Sense has no supporting references")
        if len(set(sense.references)) != len(sense.references):
            raise InvalidOutputError("duplicate source reference")
        if not set(sense.references) <= allowed:
            raise InvalidOutputError("reference was not supplied in source evidence")


def validate_evidence(
    generated: WordOutput,
    selected: SourceEntry,
    synsets: list[Synset],
    *,
    target: str,
) -> None:
    """An LLM may synthesize meanings, but cannot fabricate cited source rows."""
    try:
        allowed = {f"a{index}" for index in range(1, len(selected.senses) + 1)} | {
            f"b{index}" for index in range(1, len(synsets) + 1)
        }
        validate_inventory(generated, allowed, target)
    except ValueError as exc:
        raise InvalidOutputError(str(exc)) from exc
