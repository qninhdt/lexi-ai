"""Tantivy projection/search backed by a disposable PostgreSQL dictionary."""

from sqlalchemy import event, insert, select

from lexi_ai import schema as row
from lexi_ai.text import match_key
from lexi_ai.words.search import Search, search


async def seed_words(db, names, *, aliases=(), forms=(), patterns=()):
    async with db.transaction() as session:
        words = {
            name: row.Word(
                lemma=name, match_key=match_key(name), entry_type="WORD", generation_state="DONE"
            )
            for name in names
        }
        session.add_all(words.values())
        await session.flush()
        senses = {
            name: row.Sense(word_id=word.id, pos="VERB", tier="CORE")
            for name, word in words.items()
        }
        session.add_all(senses.values())
        await session.flush()
        session.add_all(
            [
                row.WordAlias(word_id=words[owner].id, content=alias, match_key=match_key(alias))
                for owner, alias in aliases
            ]
        )
        session.add_all(
            [
                row.SenseForm(sense_id=senses[owner].id, surface=form, inf="PAST")
                for owner, form in forms
            ]
        )
        session.add_all(
            [
                row.SensePattern(sense_id=senses[owner].id, content=pattern)
                for owner, pattern in patterns
            ]
        )
    return words, senses


async def test_fuzzy_surfaces_do_not_depend_on_postgres_trigram_settings(pg_db):
    words, _ = await seed_words(
        pg_db,
        ["walk", "color", "walking stick"],
        aliases=[("color", "colour")],
        forms=[("walk", "walking")],
    )
    engine = Search(pg_db, None)
    for query, owner, surface in [
        ("walkingg", "walk", "walking"),
        ("colourg", "color", "colour"),
        ("walking stik", "walking stick", "walking stick"),
    ]:
        hits = (await engine.search(query)).words
        hit = next(item for item in hits if item.word_id == words[owner].id)
        assert (hit.match_kind, hit.matched_surface) == ("FUZZY", surface)


async def test_dedup_before_limit_shared_forms_and_fuzzy_fill(pg_db):
    names = ["walking"] + [f"walking tool {i:02d}" for i in range(40)] + ["walk"]
    words, senses = await seed_words(pg_db, names, forms=[("walk", "walking")])
    async with pg_db.transaction() as session:
        # More than the discarded design's 200 raw-row cap, all on one Word.
        await session.execute(
            insert(row.SenseForm),
            [
                {"sense_id": senses["walk"].id, "surface": "walking", "inf": "PAST"}
                for _ in range(250)
            ],
        )
    hits = (await search(pg_db, None, "walkingg")).words
    assert len(hits) == len({hit.word_id for hit in hits})
    assert words["walk"].id in {hit.word_id for hit in hits}
    hits = (await search(pg_db, None, "walking")).words
    assert (hits[0].word_id, hits[0].match_kind) == (words["walking"].id, "LEMMA")
    assert (hits[1].word_id, hits[1].match_kind) == (words["walk"].id, "FORM")

    # Eligible typo matches can fill the final limit without duplicate Word IDs.
    words, _ = await seed_words(pg_db, ["alpha"] + [f"alpha{i:02d}" for i in range(35)])
    hits = (await search(pg_db, None, "alpha")).words
    assert len(hits) == 30 and hits[0].word_id == words["alpha"].id


async def test_match_classes_have_the_accepted_order_and_no_public_score(pg_db):
    words, _ = await seed_words(
        pg_db,
        ["paint", "colour", "coat", "decorate", "paintbrush", "faint"],
        aliases=[("colour", "paint")],
        forms=[("coat", "paint")],
        patterns=[("decorate", "paint")],
    )
    hits = (await search(pg_db, None, "paint")).words
    assert [(hit.word_id, hit.match_kind) for hit in hits] == [
        (words[owner].id, kind)
        for owner, kind in [
            ("paint", "LEMMA"),
            ("colour", "ALIAS"),
            ("coat", "FORM"),
            ("decorate", "PATTERN"),
            ("paintbrush", "PREFIX"),
            ("faint", "FUZZY"),
        ]
    ]
    assert all(not hasattr(hit, "score") and not hasattr(hit, "similarity") for hit in hits)


async def test_exact_prefix_escape_unicode_pending_and_short_queries(pg_db):
    words, _ = await seed_words(
        pg_db,
        ["take off", "saw", "see", "café", "Straße", "a_b", "axb", "a%b", "a\\b"],
        aliases=[("take off", "lift off")],
        forms=[("take off", "took off"), ("see", "saw")],
    )
    for query, owner, kind in [
        ("  ＴＯＯＫ　ＯＦＦ  ", "take off", "FORM"),
        ("LIFT OFF", "take off", "ALIAS"),
        ("Cafe\u0301", "café", "LEMMA"),
        ("STRASSE", "Straße", "LEMMA"),
    ]:
        hit = (await search(pg_db, None, query)).words[0]
        assert (hit.word_id, hit.match_kind) == (words[owner].id, kind)
    hits = (await search(pg_db, None, "saw")).words
    assert [(hit.word_id, hit.match_kind) for hit in hits[:2]] == [
        (words["saw"].id, "LEMMA"),
        (words["see"].id, "FORM"),
    ]
    for query, owner in [
        ("a_", "a_b"),
        ("a%", "a%b"),
        ("a\\", "a\\b"),
        ("li", "take off"),
        ("to", "take off"),
    ]:
        hits = (await search(pg_db, None, query)).words
        assert [(hit.word_id, hit.match_kind) for hit in hits] == [(words[owner].id, "PREFIX")]
    async with pg_db.transaction() as session:
        (await session.get(row.Word, words["take off"].id)).generation_state = "PENDING"
    assert (await search(pg_db, None, "lift off")).words == []
    assert (await search(pg_db, None, "took off")).words == []


async def test_patterns_are_anchored_and_forms_are_sense_scoped(pg_db):
    words, senses = await seed_words(
        pg_db,
        ["take off"],
        forms=[("take off", "took off")],
        patterns=[("take off", "take {sth} off"), ("take off", "{sb} takes off")],
    )
    for query in ["took her coat off", "take the hat off", "John takes off"]:
        hit = (await search(pg_db, None, query)).words[0]
        assert (hit.word_id, hit.match_kind) == (words["take off"].id, "PATTERN")
    for query in ["took it away", "took it", "take the hat off tomorrow", "off take"]:
        assert not any(
            hit.match_kind == "PATTERN" for hit in (await search(pg_db, None, query)).words
        )
    async with pg_db.transaction() as session:
        other = row.Sense(word_id=words["take off"].id, pos="VERB", tier="COMMON")
        session.add(other)
        await session.flush()
        session.add(row.SenseForm(sense_id=other.id, surface="taken off", inf="PAST_PARTICIPLE"))
    assert not any(
        hit.match_kind == "PATTERN" for hit in (await search(pg_db, None, "taken it off")).words
    )


async def test_short_queries_skip_fuzzy_and_warm_search_issues_no_sql(pg_db):
    await seed_words(pg_db, ["cat", "cut", "catch"])
    statements = []

    def capture(_conn, _cursor, statement, _parameters, _context, _executemany):
        statements.append(statement)

    event.listen(pg_db.engine.sync_engine, "before_cursor_execute", capture)
    try:
        engine = Search(pg_db, None)
        assert {hit.lemma for hit in (await engine.search("ca")).words} == {"cat", "catch"}
        assert len(statements) == 5
        assert all("similarity" not in sql and "payload" not in sql for sql in statements)
        statements.clear()
        assert {hit.lemma for hit in (await engine.search("cot")).words} == {"cat", "cut"}
        assert statements == []
    finally:
        event.remove(pg_db.engine.sync_engine, "before_cursor_execute", capture)


async def test_forms_and_aliases_update_without_search_mirror_triggers(pg_db):
    words, senses = await seed_words(
        pg_db, ["walk"], aliases=[("walk", "stroll")], forms=[("walk", "walking")]
    )
    async with pg_db.transaction() as session:
        form = await session.scalar(
            select(row.SenseForm).where(
                row.SenseForm.sense_id == senses["walk"].id,
            )
        )
        form.surface = "walked"
        alias = await session.scalar(select(row.WordAlias))
        alias.content, alias.match_key = "amble", match_key("amble")
    assert (await search(pg_db, None, "walkedd")).words[0].matched_surface == "walked"
    assert (await search(pg_db, None, "ambl")).words[0].matched_surface == "amble"
    async with pg_db.transaction() as session:
        await session.delete(await session.get(row.Word, words["walk"].id))
    assert (await search(pg_db, None, "walkedd")).words == []
