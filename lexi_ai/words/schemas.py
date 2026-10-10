"""Word generation schemas and lexical/evidence validation."""

from enum import Enum
from typing import TypedDict, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from lexi_ai.errors import InvalidOutputError
from lexi_ai.patterns import validate_pattern
from lexi_ai.references.cambridge import SourceEntry
from lexi_ai.references.wordnet import Synset
from lexi_ai.text import (
    answer_key,
    canonical_markup,
    match_key,
    parse_form,
    parse_marked_example,
    strip_markup,
    validate_lemma,
)
from lexi_ai.vocab import (
    CEFRLevel,
    EntryType,
    PartOfSpeech,
    Register,
    Tier,
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


class WordRelations(TypedDict, total=False):
    __pydantic_config__ = ConfigDict(extra="forbid")
    WORD_FAMILY: list[str]
    CONFUSED_WITH: list[str]


class SenseRelations(TypedDict, total=False):
    __pydantic_config__ = ConfigDict(extra="forbid")
    SYNONYM: list[str]
    ANTONYM: list[str]
    HYPERNYM: list[str]
    HYPONYM: list[str]
    MERONYM: list[str]
    HOLONYM: list[str]


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
        min_length=1, description="Natural use; mark targets as [surface] or [surface|code]"
    )
    forms: list[str] = Field(description="Surface or surface|inflection-code; base has no code")
    patterns: list[str]
    collocations: list[str]
    relations: SenseRelations = Field(
        description="Related citation lemmas grouped by relation type; omit empty groups"
    )
    cefr_level: CEFRLevel
    register_: Register | None = Field(alias="register")

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
        for form in self.forms:
            parse_form(form)
        for lemmas in self.relations.values():
            for lemma in lemmas:
                if parse_marked_example(lemma)[1]:
                    raise ValueError("relation lemmas must be plain text")
                validate_lemma(lemma)
        for pattern in self.patterns:
            validate_pattern(pattern)
        for text in self.collocations:
            if parse_marked_example(text)[1]:
                raise ValueError("collocations must be plain text")
        return self


class EnrichmentItem(SenseEnrichment):
    sense_id: int = Field(ge=1, description="The supplied local Sense ID")


class EnrichmentBatch(Strict):
    senses: list[EnrichmentItem] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def unique_ids(self):
        if len({sense.sense_id for sense in self.senses}) != len(self.senses):
            raise ValueError("duplicate enrichment Sense ID")
        return self


class SenseOutput(InventorySense, SenseEnrichment):
    """Locally assembled fixed definition/POS and validated enrichment."""


class InventoryOutput(Strict):
    lemma: str = Field(description="One stable citation lemma for the selected lexical item")
    type: EntryType
    aliases: list[str]
    related: WordRelations = Field(
        description="Related citation lemmas grouped by relation type; omit empty groups"
    )
    senses: list[InventorySense] = Field(min_length=1)

    @field_validator("lemma")
    @classmethod
    def check_lemma(cls, value: str) -> str:
        return validate_lemma(value)

    @model_validator(mode="after")
    def check(self):
        for alias in self.aliases:
            validate_lemma(alias)
        for lemmas in self.related.values():
            for lemma in lemmas:
                if parse_marked_example(lemma)[1]:
                    raise ValueError("relation lemmas must be plain text")
                validate_lemma(lemma)
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
