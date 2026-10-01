"""Validate/unwrap target markup, then translate and cache the exact resulting text."""

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

from lexi_ai.errors import InvalidOutputError, InvalidResourceError, MissingProviderError
from lexi_ai.inference.prompting import render_prompt
from lexi_ai.text import content_hash, parse_marked_example

from . import storage as translations

_LANGUAGE = re.compile(r"[a-z]{2}(?:-[A-Z]{2})?")


class TranslationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content: str = Field(description="Translated text only; no explanation or formatting wrapper")

    @field_validator("content")
    @classmethod
    def check_content(cls, value: str) -> str:
        if not value.strip() or len(value) > 16000:
            raise ValueError("invalid translated content")
        return value


async def translate_text(db, llm, content: str, target_language: str) -> str:
    """Caller coordinates overlapping cache misses; one miss writes atomically."""
    if not isinstance(target_language, str) or not _LANGUAGE.fullmatch(target_language):
        raise InvalidResourceError("invalid target language")
    try:
        translatable, _ = parse_marked_example(content)
        fingerprint = content_hash(translatable)
    except ValueError as exc:
        raise InvalidResourceError("invalid translation text or target expression markup") from exc
    existing = await translations.by_key(db, fingerprint, target_language)
    if existing:
        return existing.content
    if llm is None:
        raise MissingProviderError("translation requires a structured LLM")
    instruction, data = render_prompt(
        "translation/prompts/translate.jinja",
        target_language=target_language,
        content=translatable,
    )
    try:
        output = TranslationOutput.model_validate(
            await llm.complete(
                instruction,
                data,
                TranslationOutput,
            )
        )
    except ValueError as exc:
        raise InvalidOutputError("invalid translation output") from exc
    await translations.insert(db, fingerprint, target_language, output.content)
    return output.content
