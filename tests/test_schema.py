from sqlalchemy import UniqueConstraint, inspect, select, text

from lexi_ai.db.session import Database
from lexi_ai.schema import Base, Definition, Example, Question, Sense, Theme, Word


def test_foreign_keys_have_leading_indexes_without_redundant_word_indexes():
    for table in Base.metadata.sorted_tables:
        keys = [tuple(column.name for column in table.primary_key.columns)]
        keys += [
            tuple(column.name for column in constraint.columns)
            for constraint in table.constraints
            if isinstance(constraint, UniqueConstraint)
        ]
        keys += [tuple(column.name for column in index.columns) for index in table.indexes]
        for foreign_key in table.foreign_keys:
            assert any(key and key[0] == foreign_key.parent.name for key in keys), (
                f"{table.name}.{foreign_key.parent.name} has no leading index"
            )
    assert not any(
        tuple(column.name for column in index.columns) == ("word_id",)
        for index in Base.metadata.tables["word_aliases"].indexes
    )
    assert any(
        tuple(column.name for column in index.columns) == ("match_key",)
        for index in Base.metadata.tables["word_aliases"].indexes
    )


async def test_theme_delete_cascades_without_neutral_relabel(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'generated.db'}")
    try:
        await db.create_schema(Base.metadata)
        async with db.transaction() as session:
            word = Word(
                lemma="glisten", match_key="glisten", entry_type="word", generation_state="done"
            )
            theme = Theme(key="pirate", name="Pirate", voice="captain", diction="nautical")
            session.add_all([word, theme])
            await session.flush()
            sense = Sense(word_id=word.id, pos="verb", tier="common")
            session.add(sense)
            await session.flush()
            session.add_all(
                [
                    Definition(sense_id=sense.id, content="shine"),
                    Definition(sense_id=sense.id, theme_id=theme.id, content="gleam, arrr"),
                    Example(sense_id=sense.id, theme_id=theme.id, content="It gleams."),
                    Question(
                        sense_id=sense.id,
                        theme_id=theme.id,
                        question_type="meaning_in_context",
                        payload="{}",
                    ),
                ]
            )
        async with db.transaction() as session:
            await session.delete(await session.get(Theme, theme.id))
        async with db.transaction() as session:
            assert len((await session.scalars(select(Definition))).all()) == 1
            assert (await session.scalars(select(Example))).all() == []
            assert (await session.scalars(select(Question))).all() == []
            await session.delete(await session.get(Word, word.id))
        async with db.transaction() as session:
            assert (await session.scalars(select(Sense))).all() == []
            assert (await session.scalars(select(Definition))).all() == []
    finally:
        await db.close()


async def test_schema_has_no_legacy_columns_or_tables(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'generated.db'}")
    try:
        await db.create_schema(Base.metadata)
        async with db.engine.connect() as connection:
            tables = await connection.run_sync(lambda conn: inspect(conn).get_table_names())
            assert not {"themed_senses", "assets"} & set(tables)
            columns = await connection.run_sync(
                lambda conn: {
                    table: {item["name"] for item in inspect(conn).get_columns(table)}
                    for table in tables
                }
            )
            assert "definition" not in columns["senses"]
            for table in tables:
                assert not {"model_id", "prompt_version", "file_path"} & columns[table]
            assert (
                "position" not in columns["definitions"] | columns["examples"] | columns["senses"]
            )
            indexes = await connection.run_sync(lambda conn: inspect(conn).get_indexes("senses"))
            assert any("word_id" in item["column_names"] for item in indexes)
            assert (await connection.execute(text("PRAGMA foreign_keys"))).scalar() == 1
        assert Base.metadata.tables["word_sources"].c.source_id.unique
    finally:
        await db.close()
