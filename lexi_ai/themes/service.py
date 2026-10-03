"""User-managed Theme metadata and once-per-Word themed content."""

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import delete, select, update
from sqlalchemy.exc import IntegrityError

from lexi_ai import schema as row
from lexi_ai.db.collections import collection
from lexi_ai.db.pagination import validate_page
from lexi_ai.errors import InvalidOutputError, InvalidResourceError, MissingProviderError
from lexi_ai.inference.prompting import render_prompt
from lexi_ai.models import Theme
from lexi_ai.schema import Theme as ThemeRow
from lexi_ai.text import parse_marked_example
from lexi_ai.vocab import GenerationState
from lexi_ai.words.storage import get_word, insert_contents

from .namespace import normalize_key


class ThemeParts(BaseModel):
    model_config = ConfigDict(extra="forbid")
    voice: str = Field(description="Whose voice or speaking perspective")
    diction: str = Field(description="Choice and manner of language")

    @model_validator(mode="after")
    def check(self):
        if any(not 1 <= len(text.strip()) <= 1000 for text in (self.voice, self.diction)):
            raise ValueError("invalid Theme voice/diction length")
        return self


class ThemedSense(BaseModel):
    model_config = ConfigDict(extra="forbid")
    definition: str = Field(
        min_length=1, max_length=16000, description="One themed definition of this same Sense"
    )
    examples: list[str] = Field(description="Requested tagged usage examples")

    @model_validator(mode="after")
    def check(self):
        if not self.definition.strip():
            raise ValueError("themed definition must not be blank")
        return self


class ThemedWord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    senses: list[ThemedSense]


def _dto(theme: ThemeRow) -> Theme:
    return Theme(theme.id, theme.key, theme.name, theme.voice, theme.diction)


async def create_theme(db, llm, key: str, name: str, concept: str) -> Theme:
    key = normalize_key(key)
    if not name.strip() or not concept.strip() or len(concept) > 4000:
        raise InvalidResourceError("invalid Theme name or concept")
    if await get_theme(db, key) is not None:
        raise InvalidResourceError("Theme key already exists")
    if llm is None:
        raise MissingProviderError("Theme creation requires a provider")
    instruction, data = render_prompt("themes/prompts/generate_theme.jinja", theme_prompt=concept)
    try:
        parts = ThemeParts.model_validate(await llm.complete(instruction, data, ThemeParts))
    except ValueError as exc:
        raise InvalidOutputError("invalid Theme output") from exc
    try:
        async with db.transaction() as session:
            record = ThemeRow(key=key, name=name, voice=parts.voice, diction=parts.diction)
            session.add(record)
            await session.flush()
            return _dto(record)
    except IntegrityError as exc:
        raise InvalidResourceError("Theme key already exists") from exc


async def get_theme(db, key: str) -> Theme | None:
    async with db.read() as connection:
        record = (
            await connection.execute(
                select(ThemeRow.__table__).where(ThemeRow.key == normalize_key(key))
            )
        ).first()
        return _dto(record) if record else None


async def list_themes(db, *, after_key: str | None = None, limit: int | None = None) -> list[Theme]:
    validate_page(limit)
    statement = select(ThemeRow.__table__).order_by(ThemeRow.key).limit(limit)
    if after_key is not None:
        statement = statement.where(ThemeRow.key > normalize_key(after_key))
    async with db.read() as connection:
        return [_dto(record) for record in (await connection.execute(statement)).all()]


async def update_theme(db, key: str, *, name=None, voice=None, diction=None) -> Theme:
    fields = {}
    for field, value in (("name", name), ("voice", voice), ("diction", diction)):
        if value is not None:
            if not isinstance(value, str) or not value.strip():
                raise InvalidResourceError("invalid Theme field")
            fields[field] = value
    if not fields:
        record = await get_theme(db, key)
        if record is None:
            raise InvalidResourceError("unknown Theme")
        return record
    async with db.transaction() as session:
        record = await session.scalar(
            update(ThemeRow)
            .where(ThemeRow.key == normalize_key(key))
            .values(**fields)
            .returning(ThemeRow)
        )
        if record is None:
            raise InvalidResourceError("unknown Theme")
        return _dto(record)


async def delete_theme(db, key: str) -> bool:
    async with db.transaction() as session:
        result = await session.execute(delete(ThemeRow).where(ThemeRow.key == normalize_key(key)))
        removed = bool(result.rowcount)
    if removed:
        for cache in (db.content_cache, db.question_cache):
            if cache is not None:
                cache.clear()
    return removed


async def ensure_word_theme(db, llm, word_id: int, key: str, example_count: int):
    """Caller serializes concurrent requests; each new namespace is an atomic set."""
    if type(example_count) is not int or example_count < 1:
        raise ValueError("example count must be a positive integer")
    existing = await get_word(db, word_id, theme_key=key)
    if existing is not None:
        return existing
    if llm is None:
        raise MissingProviderError("themed generation requires a provider")
    senses_projection = collection(
        row.Sense,
        {
            "id": row.Sense.id,
            "pos": row.Sense.pos,
            "tier": row.Sense.tier,
            "meaning_anchor": select(row.Definition.content)
            .where(
                row.Definition.sense_id == row.Sense.id,
                row.Definition.theme_id.is_(None),
            )
            .correlate(row.Sense.__table__)
            .scalar_subquery(),
        },
        row.Sense.word_id == row.Word.id,
        order_by=(row.Sense.id,),
        correlate=(row.Word.__table__,),
    )
    statement = (
        select(row.Word.lemma, ThemeRow.id, ThemeRow.voice, ThemeRow.diction, senses_projection)
        .select_from(row.Word)
        .join(ThemeRow, ThemeRow.key == normalize_key(key))
        .where(row.Word.id == word_id, row.Word.generation_state == GenerationState.DONE)
    )
    async with db.read() as connection:
        context = (await connection.execute(statement)).first()
    if context is None or not context[4] or any(not s["meaning_anchor"] for s in context[4]):
        raise InvalidResourceError("neutral Word has not been generated")
    lemma, theme_id, voice, diction, selected_senses = context
    senses = [{k: v for k, v in sense.items() if k != "id"} for sense in selected_senses]
    instruction, data = render_prompt(
        "themes/prompts/generate_themed_word.jinja",
        neutral_word={"word": lemma, "senses": senses},
        theme={"voice": voice, "diction": diction},
        generation_parameters={
            "examples_per_sense": example_count,
        },
    )
    try:
        content = ThemedWord.model_validate(await llm.complete(instruction, data, ThemedWord))
        if len(content.senses) != len(selected_senses):
            raise ValueError("not every sense has themed content")
        for group in content.senses:
            if len(group.examples) != example_count:
                raise ValueError("themed content cardinality differs from configured counts")
            for example in group.examples:
                if not parse_marked_example(example)[1]:
                    raise ValueError("themed example missing target")
    except ValueError as exc:
        raise InvalidOutputError("incomplete themed Word output") from exc
    async with db.transaction() as session:
        await insert_contents(
            session,
            [
                (sense["id"], group)
                for sense, group in zip(selected_senses, content.senses, strict=True)
            ],
            theme_id,
        )
    if db.content_cache is not None:
        db.content_cache.clear()
    return await get_word(db, word_id, theme_id)
