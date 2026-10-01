"""Append and retrieve trusted full Question artifacts by exact namespace."""

import json
from copy import deepcopy
from dataclasses import asdict, replace

from sqlalchemy import delete, literal, select

from lexi_ai import schema as row
from lexi_ai.db.bulk import insert_identified_rows
from lexi_ai.db.collections import JsonObject, collection
from lexi_ai.db.pagination import validate_page
from lexi_ai.db.random import random_position
from lexi_ai.errors import InvalidResourceError
from lexi_ai.models import Option, Question, Word
from lexi_ai.schema import Question as QuestionRow
from lexi_ai.themes.namespace import theme_scope
from lexi_ai.words.storage import aliases, sense_fields, sense_view


def _dto(row: QuestionRow) -> Question:
    payload = json.loads(row.payload)
    return Question(
        row.id,
        row.sense_id,
        row.theme_id,
        row.question_type,
        payload["content"],
        Option(**payload["correct"]),
        [Option(**option) for option in payload["distractors"]],
        payload["correct_alternatives"],
        payload["target_placement"],
    )


async def append(db, artifacts: list[Question]) -> list[Question]:
    if not artifacts:
        return []
    # Snapshot before the first await; returned artifacts must match saved JSON
    # and must not share mutable dialogue/options lists with caller-owned input.
    artifacts = deepcopy(artifacts)
    async with db.transaction(immediate=True) as session:
        values = []
        for artifact in artifacts:
            payload = {
                "content": artifact.content,
                "correct": asdict(artifact.correct),
                "distractors": [asdict(option) for option in artifact.distractors],
                "correct_alternatives": artifact.correct_alternatives,
                "target_placement": artifact.target_placement,
            }
            values.append(
                dict(
                    sense_id=artifact.sense_id,
                    theme_id=artifact.theme_id,
                    question_type=artifact.question_type,
                    payload=json.dumps(payload, ensure_ascii=False),
                )
            )
        identifiers = await insert_identified_rows(session, QuestionRow, values)
        return [
            replace(artifact, id=identifier)
            for artifact, identifier in zip(artifacts, identifiers, strict=True)
        ]


async def get(db, question_id: int) -> Question | None:
    async with db.read() as connection:
        record = (
            await connection.execute(select(QuestionRow).where(QuestionRow.id == question_id))
        ).first()
        return _dto(record) if record else None


async def list_for_sense(
    db,
    sense_id: int,
    question_type: str | None = None,
    theme_id: int | None = None,
    *,
    theme_key: str | None = None,
    after_id: int | None = None,
    limit: int | None = None,
) -> list[Question]:
    validate_page(limit, after_id)
    identifier, valid = theme_scope(theme_id, theme_key)
    stmt = select(QuestionRow).where(
        QuestionRow.sense_id == sense_id, QuestionRow.theme_id == identifier
    )
    ordering = [QuestionRow.theme_id]
    if question_type is not None:
        stmt = stmt.where(QuestionRow.question_type == question_type)
        ordering.append(QuestionRow.question_type)
    if after_id is not None:
        stmt = stmt.where(QuestionRow.id > after_id)
    stmt = stmt.order_by(*ordering, QuestionRow.id).limit(limit)
    return await _read_bank(db, stmt, valid)


async def _read_bank(db, statement, valid):
    # A left-join sentinel validates Theme even when the requested bank is empty.
    # It carries no Question and is never deserialized as an artifact.
    bank = statement.subquery("bank")
    singleton = select(literal(1)).subquery("namespace")
    statement = (
        select(valid.label("theme_valid"), *bank.c)
        .select_from(singleton)
        .outerjoin(bank, literal(True))
        .order_by(bank.c.id)
    )
    async with db.read() as connection:
        records = (await connection.execute(statement)).all()
    if not records[0].theme_valid:
        raise InvalidResourceError("unknown Theme")
    return [_dto(record) for record in records if record.id is not None]


async def retrieve(
    db,
    sense_id: int,
    question_type: str | None = None,
    theme_id: int | None = None,
    *,
    theme_key: str | None = None,
) -> Question | None:
    identifier, valid = theme_scope(theme_id, theme_key)
    maximum = QuestionRow.__table__.alias("last_question")
    slot = QuestionRow.position if question_type is None else QuestionRow.type_position
    max_slot = maximum.c.position if question_type is None else maximum.c.type_position
    scope = [QuestionRow.sense_id == sense_id, QuestionRow.theme_id == identifier]
    max_scope = [maximum.c.sense_id == sense_id, maximum.c.theme_id == identifier]
    if question_type is not None:
        scope.append(QuestionRow.question_type == question_type)
        max_scope.append(maximum.c.question_type == question_type)
    chosen = random_position(maximum, max_slot, *max_scope, scope_order=(maximum.c.theme_id,))
    records = await _read_bank(
        db, select(QuestionRow).where(*scope, slot == chosen).limit(1), valid
    )
    return records[0] if records else None


async def remove(db, question_id: int) -> bool:
    async with db.transaction() as session:
        result = await session.execute(delete(QuestionRow).where(QuestionRow.id == question_id))
        return bool(result.rowcount)


async def generation_context(db, sense_id, theme_id=None, *, theme_key=None):
    """Only the selected Sense and lexical facts; no graph or sibling Sense loads."""
    identifier, valid = theme_scope(theme_id, theme_key)
    fields = sense_fields(identifier, relations=False)
    payload = JsonObject(*(part for key, value in fields.items() for part in (literal(key), value)))
    # Bundle the independently projected identity and Theme into one statement.
    data = collection(
        row.Word.__table__.join(row.Sense.__table__, row.Sense.word_id == row.Word.id),
        {
            "lemma": row.Word.lemma,
            "entry_type": row.Word.entry_type,
            "word_id": row.Word.id,
            "aliases": aliases(row.Word.id, correlate=(row.Word.__table__,)),
            "sense": payload,
        },
        row.Sense.id == sense_id,
        row.Word.generation_state == "done",
        valid,
        order_by=(row.Sense.id,),
    )
    theme = (
        collection(
            row.Theme,
            {"id": row.Theme.id, "voice": row.Theme.voice, "diction": row.Theme.diction},
            row.Theme.id == identifier,
            order_by=(row.Theme.id,),
        )
        if theme_id is not None or theme_key is not None
        else literal(None)
    )
    async with db.read() as connection:
        theme_valid, values, themes = (await connection.execute(select(valid, data, theme))).one()
    if not theme_valid:
        raise InvalidResourceError("unknown Theme")
    if not values:
        raise InvalidResourceError("unknown or unpublished Sense")
    value = values[0]
    sense = sense_view(value["sense"])
    if sense.definition is None or (
        (theme_id is not None or theme_key is not None) and not sense.examples
    ):
        raise InvalidResourceError("Sense content unavailable in requested namespace")
    word = Word(
        value["word_id"],
        value["lemma"],
        value["entry_type"],
        "done",
        senses=[sense],
        aliases=[item["content"] for item in value["aliases"]],
    )
    return word, sense, themes[0] if themes else None
