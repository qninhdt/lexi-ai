"""Real PostgreSQL search semantics, not a Python similarity surrogate."""

from sqlalchemy import event, insert, select, text

from lexi_ai import schema as row
from lexi_ai.text import match_key
from lexi_ai.words.indexes import TRIGRAM_INDEXES
from lexi_ai.words.search import search


async def seed_words(db, names, *, aliases=(), forms=(), patterns=()):
    async with db.transaction() as session:
        words = {
            name: row.Word(
                lemma=name, match_key=match_key(name), entry_type="word", generation_state="done"
            )
            for name in names
        }
        session.add_all(words.values())
        await session.flush()
        senses = {
            name: row.Sense(word_id=word.id, pos="verb", tier="core")
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
                row.SenseForm(sense_id=senses[owner].id, surface=form, inf="past")
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


async def test_gin_sources_native_scores_and_transaction_local_threshold(pg_db):
    words, _ = await seed_words(
        pg_db,
        ["walk", "color", "walking stick"],
        aliases=[("color", "colour")],
        forms=[("walk", "walking")],
    )
    async with pg_db.transaction() as session:
        definitions = (
            await session.execute(
                text(
                    "SELECT indexname,indexdef FROM pg_indexes WHERE schemaname=current_schema() "
                    "AND indexname LIKE '%trigram'"
                )
            )
        ).all()
        assert {name for name, _ in definitions} == set(TRIGRAM_INDEXES.values())
        assert all(
            "USING gin" in definition and "gin_trgm_ops" in definition
            for _, definition in definitions
        )
        assert not any("gist" in definition for _, definition in definitions)
    for query, owner, surface in [
        ("walkingg", "walk", "walking"),
        ("colourg", "color", "colour"),
        ("walking stik", "walking stick", "walking stick"),
    ]:
        hits = (await search(pg_db, None, query)).words
        hit = next(item for item in hits if item.word_id == words[owner].id)
        assert (hit.match_kind, hit.matched_surface) == ("fuzzy", surface)

    async with pg_db.transaction() as session:
        await session.execute(text("SELECT set_config('pg_trgm.similarity_threshold','0.9',false)"))
        await session.execute(text("SELECT set_config('gin_fuzzy_search_limit','1',false)"))
        # The request's explicit 0.3 must override this pooled connection setting.
        from lexi_ai.words.search import _fuzzy

        matches = await _fuzzy(session, "walkingg", set(), 30)
        native_score = await session.scalar(text("SELECT public.similarity('walking','walkingg')"))
        assert (
            next(score for word, _, score, _, _ in matches if word.id == words["walk"].id)
            == native_score
        )
        assert await session.scalar(text("SHOW pg_trgm.similarity_threshold")) == "0.3"
        assert await session.scalar(text("SHOW gin_fuzzy_search_limit")) == "0"
    async with pg_db.transaction() as session:
        assert await session.scalar(text("SHOW pg_trgm.similarity_threshold")) == "0.9"
        assert await session.scalar(text("SHOW gin_fuzzy_search_limit")) == "1"
        await session.execute(text("SELECT set_config('pg_trgm.similarity_threshold','0.3',false)"))
        await session.execute(text("SELECT set_config('gin_fuzzy_search_limit','0',false)"))


async def test_dedup_before_limit_shared_forms_and_fuzzy_fill(pg_db):
    names = ["walking"] + [f"walking tool {i:02d}" for i in range(40)] + ["walk"]
    words, senses = await seed_words(pg_db, names, forms=[("walk", "walking")])
    async with pg_db.transaction() as session:
        # More than the discarded design's 200 raw-row cap, all on one Word.
        await session.execute(
            insert(row.SenseForm),
            [
                {"sense_id": senses["walk"].id, "surface": "walking", "inf": "past"}
                for _ in range(250)
            ],
        )
    hits = (await search(pg_db, None, "walkingg")).words
    assert len(hits) == 30
    assert len({hit.word_id for hit in hits}) == 30
    assert words["walk"].id in {hit.word_id for hit in hits}
    hits = (await search(pg_db, None, "walking")).words
    assert (hits[0].word_id, hits[0].match_kind) == (words["walking"].id, "lemma")
    assert (hits[1].word_id, hits[1].match_kind) == (words["walk"].id, "form")

    # Fuzzy final LIMIT excludes already-ranked Words; duplicates cannot steal
    # slots from valid lower-score suggestions that complete the result set.
    words, _ = await seed_words(pg_db, ["alpha"] + [f"alphx{i:02d}" for i in range(35)])
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
            ("paint", "lemma"),
            ("colour", "alias"),
            ("coat", "form"),
            ("decorate", "pattern"),
            ("paintbrush", "prefix"),
            ("faint", "fuzzy"),
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
        ("  ＴＯＯＫ　ＯＦＦ  ", "take off", "form"),
        ("LIFT OFF", "take off", "alias"),
        ("Cafe\u0301", "café", "lemma"),
        ("STRASSE", "Straße", "lemma"),
    ]:
        hit = (await search(pg_db, None, query)).words[0]
        assert (hit.word_id, hit.match_kind) == (words[owner].id, kind)
    hits = (await search(pg_db, None, "saw")).words
    assert [(hit.word_id, hit.match_kind) for hit in hits[:2]] == [
        (words["saw"].id, "lemma"),
        (words["see"].id, "form"),
    ]
    for query, owner in [
        ("a_", "a_b"),
        ("a%", "a%b"),
        ("a\\", "a\\b"),
        ("li", "take off"),
        ("to", "take off"),
    ]:
        hits = (await search(pg_db, None, query)).words
        assert [(hit.word_id, hit.match_kind) for hit in hits] == [(words[owner].id, "prefix")]
    async with pg_db.transaction() as session:
        (await session.get(row.Word, words["take off"].id)).generation_state = "pending"
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
        assert (hit.word_id, hit.match_kind) == (words["take off"].id, "pattern")
    for query in ["took it away", "took it", "take the hat off tomorrow", "off take"]:
        assert not any(
            hit.match_kind == "pattern" for hit in (await search(pg_db, None, query)).words
        )
    async with pg_db.transaction() as session:
        other = row.Sense(word_id=words["take off"].id, pos="verb", tier="common")
        session.add(other)
        await session.flush()
        session.add(row.SenseForm(sense_id=other.id, surface="taken off", inf="past_participle"))
    assert not any(
        hit.match_kind == "pattern" for hit in (await search(pg_db, None, "taken it off")).words
    )


async def test_sql_short_queries_skip_fuzzy_and_sources_are_not_scanned_in_python(pg_db):
    await seed_words(pg_db, ["cat", "cut", "catch"])
    statements = []

    def capture(_conn, _cursor, statement, _parameters, _context, _executemany):
        statements.append(statement)

    event.listen(pg_db.engine.sync_engine, "before_cursor_execute", capture)
    try:
        await search(pg_db, None, "ca")
        assert len(statements) == 1
        assert "similarity" not in statements[0]
        assert "sense_patterns" not in statements[0]
        statements.clear()
        await search(pg_db, None, "cot")
        fuzzy = next(
            statement
            for statement in statements
            if "UNION ALL" in statement and "similarity" in statement
        )
        assert fuzzy.count("UNION ALL") == 2
        assert "row_number() OVER (PARTITION BY" in fuzzy
        assert "OPERATOR(public.%%)" in fuzzy or "OPERATOR(public.%)" in fuzzy
        assert fuzzy.count("LIMIT") == 1
        assert not any("<->" in statement or "lower(" in statement for statement in statements)
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
