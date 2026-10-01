"""Word identity/publication writes and one-statement dictionary reads."""

from sqlalchemy import insert, literal, select
from sqlalchemy.ext.asyncio import AsyncSession

from lexi_ai import schema as row
from lexi_ai.db.bulk import insert_rows
from lexi_ai.db.collections import JsonObject, collection
from lexi_ai.errors import InvalidResourceError, WordCollisionError
from lexi_ai.models import Definition as DefinitionView
from lexi_ai.models import Example as ExampleView
from lexi_ai.models import Form
from lexi_ai.models import Sense as SenseView
from lexi_ai.models import SenseRelation as RelationView
from lexi_ai.models import Word as WordView
from lexi_ai.models import WordRelation as WordRelationView
from lexi_ai.relations.storage import definition_hash
from lexi_ai.schema import Word, WordAlias, WordSource
from lexi_ai.text import match_key
from lexi_ai.themes.namespace import theme_scope


async def consumed_word(session: AsyncSession, source_id: int, *, theme_key=None) -> int | None:
    consumed = (
        select(Word.id)
        .join(WordSource)
        .where(WordSource.source_id == source_id, Word.generation_state == "done")
        .scalar_subquery()
    )
    _, valid = theme_scope(theme_key=theme_key)
    word_id, theme_valid = (await session.execute(select(consumed, valid))).one()
    if not theme_valid:
        raise InvalidResourceError("unknown Theme")
    return word_id


async def publish_identity(session, source_id, lemma, entry_type, aliases) -> Word:
    key = match_key(lemma)
    word = await session.scalar(select(Word).where(Word.match_key == key))
    if word is not None and word.generation_state == "done":
        raise WordCollisionError("selected entry collides with a completed Word")
    association = await session.scalar(select(WordSource).where(WordSource.source_id == source_id))
    if association is not None and (word is None or association.word_id != word.id):
        raise WordCollisionError("source entry is already associated with another Word")
    if word is None:
        word = Word(lemma=lemma, match_key=key, entry_type=entry_type, generation_state="pending")
        session.add(word)
        await session.flush()
    else:
        word.lemma, word.entry_type = lemma, entry_type
    if association is None:
        session.add(WordSource(word_id=word.id, source_id=source_id))
    seen = {key}
    alias_rows = []
    for alias in aliases:
        alias_key = match_key(alias)
        if alias_key not in seen:
            alias_rows.append(dict(word_id=word.id, content=alias, match_key=alias_key))
            seen.add(alias_key)
    await session.flush()
    await insert_rows(session, WordAlias, alias_rows)
    return word


async def insert_contents(session, groups, theme_id=None):
    """Publish one relational Definition and requested Examples per Sense/namespace."""
    for table in (row.Definition, row.Example):
        rows = [
            dict(sense_id=sense_id, theme_id=theme_id, content=content)
            for sense_id, group in groups
            for content in ([group.definition] if table is row.Definition else group.examples)
        ]
        await insert_rows(session, table, rows)


async def target_words(session, lemmas):
    normalized = {}
    for lemma in lemmas:
        normalized.setdefault(match_key(lemma), lemma)
    if not normalized:
        return {}
    found = dict(
        (
            await session.execute(
                select(Word.match_key, Word.id).where(Word.match_key.in_(normalized))
            )
        ).all()
    )
    missing = [
        dict(match_key=key, lemma=lemma, generation_state="pending")
        for key, lemma in normalized.items()
        if key not in found
    ]
    if missing:
        found.update(
            (await session.execute(insert(Word).returning(Word.match_key, Word.id), missing)).all()
        )
    return found


def sense_fields(theme_id, *, relations=True):
    sense = row.Sense.__table__
    fields = {column.name: column for column in sense.c}
    definition = row.Definition.__table__
    fields["definition"] = (
        select(
            JsonObject(
                literal("id"),
                definition.c.id,
                literal("content"),
                definition.c.content,
                literal("theme_id"),
                definition.c.theme_id,
            )
        )
        .where(definition.c.sense_id == row.Sense.id, definition.c.theme_id == theme_id)
        .correlate(sense)
        .scalar_subquery()
    )
    for name, table, columns, scoped in (
        ("examples", row.Example, ("id", "content", "theme_id"), True),
        ("forms", row.SenseForm, ("surface", "inf"), False),
        ("patterns", row.SensePattern, ("content",), False),
        ("collocations", row.Collocation, ("content",), False),
    ):
        conditions = [table.sense_id == row.Sense.id]
        if scoped:
            conditions.append(table.theme_id == theme_id)
        fields[name] = collection(
            table,
            {name: getattr(table, name) for name in columns},
            *conditions,
            order_by=(table.id,),
            correlate=(sense,),
        )
    if relations:
        edge = row.SenseRelation.__table__
        target = Word.__table__.alias("relation_target")
        fields["relations"] = collection(
            edge.join(target, target.c.id == edge.c.to_word_id),
            {
                "rel_type": edge.c.rel_type,
                "to_word_id": target.c.id,
                "to_word_lemma": target.c.lemma,
                "to_sense_id": edge.c.to_sense_id,
                "target_hash": edge.c.target_hash,
                "attempted": edge.c.resolve_attempted_at,
            },
            edge.c.from_sense_id == row.Sense.id,
            order_by=(edge.c.id,),
            correlate=(sense,),
        )
    return fields


def sense_view(data, fingerprints=None):
    data = dict(data)
    data["definition"] = (
        DefinitionView(**data["definition"]) if data["definition"] is not None else None
    )
    data["examples"] = [ExampleView(**item) for item in data["examples"]]
    data["forms"] = [Form(**item) for item in data["forms"]]
    for name in ("patterns", "collocations"):
        data[name] = [item["content"] for item in data[name]]
    relations = []
    fingerprints = {} if fingerprints is None else fingerprints
    for edge in data.pop("relations", []):
        identifier = edge["to_sense_id"]
        if identifier is not None:
            fingerprint = fingerprints.get(identifier)
            state = (
                "resolved"
                if fingerprint is not None and fingerprint == edge["target_hash"]
                else "pending"
            )
        else:
            state = "unresolvable" if edge["attempted"] else "pending"
        relations.append(
            RelationView(
                edge["rel_type"],
                edge["to_word_id"],
                edge["to_word_lemma"],
                state,
                identifier if state == "resolved" else None,
            )
        )
    return SenseView(**data, relations=relations)


def target_evidence(*sense_conditions):
    """Return neutral text once per distinct relation target, in the reader snapshot."""
    referenced = (
        select(row.SenseRelation.to_sense_id.label("id"))
        .join(row.Sense, row.Sense.id == row.SenseRelation.from_sense_id)
        .where(*sense_conditions, row.SenseRelation.to_sense_id.is_not(None))
        .distinct()
        .subquery()
    )
    return collection(
        row.Definition.__table__.join(referenced, referenced.c.id == row.Definition.sense_id),
        {"id": row.Definition.sense_id, "content": row.Definition.content},
        row.Definition.theme_id.is_(None),
        order_by=(row.Definition.sense_id,),
    )


def evidence_fingerprints(evidence):
    return {item["id"]: definition_hash(item["content"]) for item in evidence}


def aliases(word_id, *, correlate=()):
    return collection(
        row.WordAlias,
        {"content": row.WordAlias.content},
        row.WordAlias.word_id == word_id,
        order_by=(row.WordAlias.id,),
        correlate=correlate,
    )


async def get_word(db, word_id, theme_id=None, *, theme_key=None):
    identifier, valid = theme_scope(theme_id, theme_key)
    scopes = [Word.id == word_id, Word.generation_state == "done"]
    scopes.append(select(row.Sense.id).where(row.Sense.word_id == Word.id).exists())
    if theme_id is not None or theme_key is not None:
        missing = (
            select(row.Sense.id)
            .where(
                row.Sense.word_id == Word.id,
                ~select(row.Definition.id)
                .where(
                    row.Definition.sense_id == row.Sense.id,
                    row.Definition.theme_id == identifier,
                )
                .exists()
                | ~select(row.Example.id)
                .where(
                    row.Example.sense_id == row.Sense.id,
                    row.Example.theme_id == identifier,
                )
                .exists(),
            )
            .exists()
        )
        scopes.append(~missing)
    target = Word.__table__.alias("word_relation_target")
    edge = row.WordRelation.__table__
    fields = {
        name: getattr(Word, name) for name in ("id", "lemma", "entry_type", "generation_state")
    }
    fields["aliases"] = aliases(Word.id, correlate=(Word.__table__,))
    fields["senses"] = collection(
        row.Sense,
        sense_fields(identifier),
        row.Sense.word_id == Word.id,
        order_by=(row.Sense.id,),
        correlate=(Word.__table__,),
    )
    fields["related"] = collection(
        edge.join(target, target.c.id == edge.c.to_word_id),
        {"rel_type": edge.c.rel_type, "to_word_id": target.c.id, "to_word_lemma": target.c.lemma},
        edge.c.from_word_id == Word.id,
        order_by=(edge.c.id,),
        correlate=(Word.__table__,),
    )
    payload = (
        select(JsonObject(*(part for k, v in fields.items() for part in (literal(k), v))))
        .where(*scopes, valid)
        .scalar_subquery()
    )
    async with db.read() as connection:
        theme_valid, data, evidence = (
            await connection.execute(
                select(valid, payload, target_evidence(row.Sense.word_id == word_id))
            )
        ).one()
    if not theme_valid:
        raise InvalidResourceError("unknown Theme")
    if data is None:
        return None
    fingerprints = evidence_fingerprints(evidence)
    data["senses"] = [sense_view(item, fingerprints) for item in data["senses"]]
    data["aliases"] = [item["content"] for item in data["aliases"]]
    data["related"] = [WordRelationView(**item) for item in data["related"]]
    return WordView(**data)


async def get_senses(db, ids, theme_id=None):
    if not ids:
        return []
    by_id = {}
    # Only requested IDs are queried; chunking bounds SQL parameter counts.
    unique_ids = list(dict.fromkeys(ids))
    fields = sense_fields(theme_id)
    async with db.read() as connection:
        for start in range(0, len(unique_ids), 500):
            scope = row.Sense.id.in_(unique_ids[start : start + 500])
            statement = select(
                collection(row.Sense, fields, scope, order_by=(row.Sense.id,)),
                target_evidence(scope),
            )
            senses, evidence = (await connection.execute(statement)).one()
            fingerprints = evidence_fingerprints(evidence)
            for data in senses:
                by_id[data["id"]] = sense_view(data, fingerprints)
    return [by_id[identifier] for identifier in ids if identifier in by_id]


async def meaning_inventory(db, *, word_id=None, owner_of=None):
    """One Word's complete neutral meaning inventory, without examples or graph data."""
    if (word_id is None) == (owner_of is None):
        raise ValueError("select a Word or the owner of a Sense")
    identifier = (
        word_id
        if word_id is not None
        else (select(row.Sense.word_id).where(row.Sense.id == owner_of).scalar_subquery())
    )
    meanings = collection(
        row.Sense.__table__.outerjoin(
            row.Definition.__table__,
            (row.Definition.sense_id == row.Sense.id) & row.Definition.theme_id.is_(None),
        ),
        {"id": row.Sense.id, "pos": row.Sense.pos, "definition": row.Definition.content},
        row.Sense.word_id == Word.id,
        order_by=(row.Sense.id,),
        correlate=(Word.__table__,),
    )
    async with db.read() as connection:
        value = (
            (
                await connection.execute(
                    select(Word.id, Word.lemma, meanings.label("senses")).where(
                        Word.id == identifier, Word.generation_state == "done"
                    )
                )
            )
            .mappings()
            .first()
        )
    if value is None:
        return None
    if any(sense["definition"] is None for sense in value["senses"]):
        raise InvalidResourceError("Word's neutral meaning inventory is incomplete")
    return dict(value)
