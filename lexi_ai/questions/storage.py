"""Append and retrieve trusted full Question artifacts by exact namespace."""

import random
from copy import deepcopy
from dataclasses import asdict, replace

import orjson
from sqlalchemy import Integer, column, delete, func, literal, select, tuple_, values

from lexi_ai import schema as row
from lexi_ai.db.bulk import insert_identified_rows
from lexi_ai.db.collections import JsonObject, collection
from lexi_ai.db.pagination import validate_page
from lexi_ai.db.random import last_position, random_position
from lexi_ai.errors import InvalidResourceError, QuestionBankChangedError
from lexi_ai.models import Option, Question, Word
from lexi_ai.schema import Question as QuestionRow
from lexi_ai.themes.namespace import theme_scope
from lexi_ai.vocab import GenerationState, QuestionType
from lexi_ai.words.storage import aliases, sense_fields, sense_view


def _dto(row: QuestionRow) -> Question:
    payload = orjson.loads(row.payload)
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
                    payload=orjson.dumps(payload).decode(),
                )
            )
        identifiers = await insert_identified_rows(session, QuestionRow, values)
        return [
            replace(artifact, id=identifier)
            for artifact, identifier in zip(artifacts, identifiers, strict=True)
        ]


async def get(db, question_id: int) -> Question | None:
    records = await get_many(db, [question_id])
    return records[0] if records else None


async def get_many(db, question_ids: list[int]) -> list[Question]:
    """Hydrate only selected artifacts, preserving the supplied identity order."""
    identifiers = list(dict.fromkeys(question_ids))
    if not identifiers:
        return []
    cache = db.question_cache
    by_id = cache.get_many(identifiers, Question) if cache is not None else {}
    missing = [identifier for identifier in identifiers if identifier not in by_id]
    if not missing:
        return [by_id[identifier] for identifier in identifiers]
    async with db.read() as connection:
        for start in range(0, len(missing), 500):
            records = (
                await connection.execute(
                    select(QuestionRow.__table__).where(
                        QuestionRow.id.in_(missing[start : start + 500])
                    )
                )
            ).all()
            for record in records:
                artifact = _dto(record)
                by_id[record.id] = artifact
                if cache is not None:
                    cache.put(record.id, artifact)
    return [by_id[identifier] for identifier in identifiers if identifier in by_id]


async def list_for_sense(
    db,
    sense_id: int,
    question_type: QuestionType | None = None,
    theme_id: int | None = None,
    *,
    theme_key: str | None = None,
    after_id: int | None = None,
    limit: int | None = None,
) -> list[Question]:
    validate_page(limit, after_id)
    if question_type is not None:
        question_type = QuestionType(question_type)
    identifier, valid = theme_scope(theme_id, theme_key)
    stmt = select(QuestionRow.id).where(
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


async def list_for_senses(
    db,
    sense_ids: list[int],
    question_types: list[QuestionType] | None = None,
    theme_id: int | None = None,
    *,
    theme_key: str | None = None,
    limit_per_type: int = 8,
) -> list[Question]:
    if type(limit_per_type) is not int or limit_per_type < 1:
        raise ValueError("limit_per_type must be a positive integer")
    if question_types is not None:
        question_types = [QuestionType(kind) for kind in question_types]
    identifier, valid = theme_scope(theme_id, theme_key)
    if not sense_ids:
        async with db.read() as connection:
            record = (await connection.execute(select(valid))).first()
            if record and not record[0]:
                raise InvalidResourceError("unknown Theme")
        return []
    rn = (
        func.row_number()
        .over(
            partition_by=(QuestionRow.sense_id, QuestionRow.question_type),
            order_by=QuestionRow.id,
        )
        .label("rn")
    )
    inner = select(QuestionRow.id, QuestionRow.sense_id, QuestionRow.question_type, rn).where(
        QuestionRow.sense_id.in_(sense_ids),
        QuestionRow.theme_id == identifier,
    )
    if question_types is not None:
        inner = inner.where(QuestionRow.question_type.in_(question_types))
    subq = inner.subquery("ranked")
    cols = [subq.c.id, subq.c.sense_id, subq.c.question_type]
    ranked_stmt = select(*cols).where(subq.c.rn <= limit_per_type)
    return await _read_bank(db, ranked_stmt, valid, order_by=("sense_id", "question_type", "id"))


async def count_for_senses(
    db,
    sense_ids: list[int],
    question_types: list[QuestionType] | None = None,
    *,
    theme_key: str | None = None,
) -> dict[tuple[int, QuestionType], int]:
    if question_types is not None:
        question_types = [QuestionType(kind) for kind in question_types]
    identifier, valid = theme_scope(None, theme_key)
    grouped = (
        select(
            QuestionRow.sense_id, QuestionRow.question_type, func.count().label("question_count")
        )
        .where(QuestionRow.sense_id.in_(sense_ids), QuestionRow.theme_id == identifier)
        .group_by(QuestionRow.sense_id, QuestionRow.question_type)
    )
    if question_types is not None:
        grouped = grouped.where(QuestionRow.question_type.in_(question_types))
    bank = grouped.subquery("bank_counts")
    singleton = select(literal(1)).subquery("namespace")
    statement = (
        select(
            valid.label("theme_valid"), bank.c.sense_id, bank.c.question_type, bank.c.question_count
        )
        .select_from(singleton)
        .outerjoin(bank, literal(True))
    )
    async with db.read() as connection:
        records = (await connection.execute(statement)).all()
    if not records[0].theme_valid:
        raise InvalidResourceError("unknown Theme")
    return {
        (record.sense_id, QuestionType(record.question_type)): record.question_count
        for record in records
        if record.sense_id is not None
    }


async def _read_bank(db, statement, valid, *, order_by=("id",)):
    # A left-join sentinel validates Theme even when the requested bank is empty.
    # It carries no Question and is never deserialized as an artifact.
    bank = statement.subquery("bank")
    singleton = select(literal(1)).subquery("namespace")
    statement = (
        select(valid.label("theme_valid"), bank.c.id)
        .select_from(singleton)
        .outerjoin(bank, literal(True))
        .order_by(*(bank.c[name] for name in order_by))
    )
    async with db.read() as connection:
        records = (await connection.execute(statement)).all()
    if not records[0].theme_valid:
        raise InvalidResourceError("unknown Theme")
    return await get_many(db, [record.id for record in records if record.id is not None])


async def retrieve(
    db,
    sense_id: int,
    question_type: QuestionType | None = None,
    theme_id: int | None = None,
    *,
    theme_key: str | None = None,
) -> Question | None:
    if question_type is not None:
        question_type = QuestionType(question_type)
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
        db, select(QuestionRow.id).where(*scope, slot == chosen).limit(1), valid
    )
    return records[0] if records else None


async def remove(db, question_id: int) -> bool:
    async with db.transaction() as session:
        result = await session.execute(delete(QuestionRow).where(QuestionRow.id == question_id))
        removed = bool(result.rowcount)
    cache = db.question_cache
    if cache is not None:
        cache.discard(question_id)
    return removed


async def retrieve_many(db, requests, *, theme_key=None) -> list[Question]:
    """Sample dense bank slots in bulk, without loading unselected payloads."""
    requests = [(sense_id, QuestionType(kind), count) for sense_id, kind, count in requests]
    if any(type(count) is not int or count < 1 for _, _, count in requests):
        raise ValueError("Question quantity must be a positive integer")
    pairs = [(sense_id, kind) for sense_id, kind, _ in requests]
    if len(set(pairs)) != len(pairs):
        raise ValueError("request each Sense/type bank once")
    identifier, valid = theme_scope(None, theme_key)
    slots = []
    # The last dense slot is the bank size. Each scope is an indexed LIMIT 1,
    # not a full bank COUNT or an unselected payload read.
    async with db.read() as connection:
        if not requests:
            if not await connection.scalar(select(valid)):
                raise InvalidResourceError("unknown Theme")
            return []
        for start in range(0, len(requests), 500):
            banks = (
                values(
                    column("sense_id", Integer),
                    column("kind", QuestionRow.question_type.type),
                    column("quantity", Integer),
                )
                .data(requests[start : start + 500])
                .cte("requested_banks")
            )
            size = last_position(
                QuestionRow,
                QuestionRow.type_position,
                QuestionRow.sense_id == banks.c.sense_id,
                QuestionRow.question_type == banks.c.kind,
                QuestionRow.theme_id == identifier,
                scope_order=(QuestionRow.theme_id,),
                correlate=(banks,),
            )
            records = (await connection.execute(select(valid, *banks.c, size))).all()
            if not records[0][0]:
                raise InvalidResourceError("unknown Theme")
            for _, sense_id, kind, quantity, size in records:
                if (size or 0) < quantity:
                    raise QuestionBankChangedError("Question bank cannot satisfy the request")
                slots.extend(
                    (sense_id, kind, slot) for slot in random.sample(range(1, size + 1), quantity)
                )
    ids = []
    if slots:
        async with db.read() as connection:
            for start in range(0, len(slots), 500):
                ids.extend(
                    (
                        await connection.scalars(
                            select(QuestionRow.id).where(
                                QuestionRow.theme_id == identifier,
                                tuple_(
                                    QuestionRow.sense_id,
                                    QuestionRow.question_type,
                                    QuestionRow.type_position,
                                ).in_(slots[start : start + 500]),
                            )
                        )
                    ).all()
                )
    artifacts = await get_many(db, ids)
    if len(artifacts) != len(slots):
        raise QuestionBankChangedError("Question bank changed during retrieval")
    return artifacts


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
        row.Word.generation_state == GenerationState.DONE,
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
        GenerationState.DONE,
        senses=[sense],
        aliases=[item["content"] for item in value["aliases"]],
    )
    return word, sense, themes[0] if themes else None
