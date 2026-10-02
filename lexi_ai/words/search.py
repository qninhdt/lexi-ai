"""One source-neutral lexical search, with an optional requestable group."""

from sqlalchemy import and_, func, literal, or_, select, text, union_all

from lexi_ai import schema as row
from lexi_ai.config import MAX_QUERY_LENGTH
from lexi_ai.db.collections import collection
from lexi_ai.errors import InvalidResourceError
from lexi_ai.models import AvailableHit, SearchResult, WordHit
from lexi_ai.patterns import matches_pattern, surface_head_key
from lexi_ai.references.cambridge import encode_available_id
from lexi_ai.text import answer_key

from .indexes import TRIGRAM_SCHEMA

_LIMIT = 30
_PATTERN_PAGE = 200


def _prefix(session, column, prefix):
    if session.bind.dialect.name == "postgresql":
        escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        return column.like(escaped + "%", escape="\\")
    # SQLite's normalized keys use binary ordering. A range uses the ordinary
    # B-tree, unlike default case-insensitive LIKE over a binary index.
    for index in range(len(prefix) - 1, -1, -1):
        codepoint = ord(prefix[index])
        if codepoint < 0x10FFFF:
            successor = 0xE000 if codepoint == 0xD7FF else codepoint + 1
            upper = prefix[:index] + chr(successor)
            return and_(column >= prefix, column < upper)
    return column >= prefix


def _rank_key(pair):
    word_id, (rank, score, hit) = pair
    return rank, -score, hit.lemma, word_id


def _trim(ranked, limit=_LIMIT):
    if len(ranked) > limit:
        kept = sorted(ranked.items(), key=_rank_key)[:limit]
        ranked.clear()
        ranked.update(kept)


def _offer(ranked, word, rank, score, kind, surface):
    previous = ranked.get(word.id)
    if previous is None or (rank, -score, surface, kind) < (
        previous[0],
        -previous[1],
        previous[2].matched_surface,
        previous[2].match_kind,
    ):
        ranked[word.id] = (
            rank,
            score,
            WordHit(word.id, word.lemma, word.entry_type, kind, surface),
        )


async def _patterns(session, key, query, ranked, *, limit=_LIMIT):
    head = surface_head_key(key)
    inflected_senses = select(row.SenseForm.sense_id).where(row.SenseForm.head_key == head)
    eligible_pattern = or_(
        row.SensePattern.head_key == head,
        row.SensePattern.head_key.is_(None),
        row.SensePattern.sense_id.in_(inflected_senses),
    )
    exact_words = [identifier for identifier, (rank, _, _) in ranked.items() if rank < 3]
    patterns = (
        select(
            row.SensePattern.id,
            row.SensePattern.sense_id,
            row.SensePattern.content,
            row.Word.id.label("word_id"),
            row.Word.lemma,
            row.Word.entry_type,
        )
        .join(row.Sense, row.Sense.id == row.SensePattern.sense_id)
        .join(row.Word, row.Word.id == row.Sense.word_id)
        .where(
            row.Word.generation_state == "done", row.Word.id.not_in(exact_words), eligible_pattern
        )
        .order_by(row.SensePattern.id)
    )
    last_id = 0
    while True:
        page = (
            patterns.where(row.SensePattern.id > last_id)
            .limit(_PATTERN_PAGE)
            .cte("pattern_page")
            .prefix_with("MATERIALIZED", dialect="postgresql")
        )
        scopes = select(page.c.sense_id, page.c.lemma).distinct().cte("pattern_scopes")
        statement = select(
            collection(page, dict(page.c.items()), order_by=(page.c.id,)),
            collection(
                scopes,
                {
                    "sense_id": scopes.c.sense_id,
                    "lemma": scopes.c.lemma,
                    "forms": collection(
                        row.SenseForm,
                        {"surface": row.SenseForm.surface},
                        row.SenseForm.sense_id == scopes.c.sense_id,
                        order_by=(row.SenseForm.id,),
                        correlate=(scopes,),
                    ),
                },
                order_by=(scopes.c.sense_id,),
            ),
        )
        records, evidence = (await session.execute(statement)).one()
        if not records:
            break
        forms_by_sense = {}
        for item in evidence:
            word_head, _, tail = answer_key(item["lemma"]).partition(" ")
            licensed = [
                first
                for first, _, rest in (
                    answer_key(form["surface"]).partition(" ") for form in item["forms"]
                )
                if rest == tail
            ]
            forms_by_sense[item["sense_id"]] = {word_head: licensed}
        for pattern in records:
            if matches_pattern(
                pattern["content"], query, forms=forms_by_sense[pattern["sense_id"]]
            ):
                word = row.Word(
                    id=pattern["word_id"], lemma=pattern["lemma"], entry_type=pattern["entry_type"]
                )
                _offer(ranked, word, 3, 1.0, "pattern", pattern["content"])
        _trim(ranked, limit)
        last_id = records[-1]["id"]
        if len(records) < _PATTERN_PAGE:
            break


def _sources():
    """Three existing domain sources; no materialized search projection."""
    return (
        (row.Word.match_key, row.Word.lemma, row.Word.id, 0, "lemma", select(row.Word)),
        (
            row.WordAlias.match_key,
            row.WordAlias.content,
            row.WordAlias.word_id,
            1,
            "alias",
            select(row.WordAlias).join(row.Word, row.Word.id == row.WordAlias.word_id),
        ),
        (
            row.SenseForm.match_key,
            row.SenseForm.surface,
            row.Sense.word_id,
            2,
            "form",
            select(row.SenseForm).join(row.Sense).join(row.Word),
        ),
    )


def _matches_statement(branches, postgres, limit):
    matches = union_all(*branches).subquery()
    # Window dedup also works for lexical-only SQLite tests. PostgreSQL performs
    # both similarity scoring and dedup BEFORE the one final result limit.
    best = select(
        matches,
        func.row_number()
        .over(
            partition_by=matches.c.word_id,
            order_by=(
                matches.c.rank,
                matches.c.score.desc(),
                matches.c.surface.collate("C" if postgres else "BINARY"),
                matches.c.kind,
            ),
        )
        .label("choice"),
    ).subquery()
    return (
        select(row.Word, best.c.rank, best.c.score, best.c.kind, best.c.surface)
        .join(best, best.c.word_id == row.Word.id)
        .where(best.c.choice == 1)
        .order_by(
            best.c.rank,
            best.c.score.desc(),
            row.Word.lemma.collate("C" if postgres else "BINARY"),
            row.Word.id,
        )
        .limit(limit)
    )


async def _lexical(session, key, limit):
    branches = []
    for column, surface, word_id, rank, kind, source in _sources():
        for condition, priority, category, score in (
            (column == key, rank, kind, 1.0),
            (and_(_prefix(session, column, key), column != key), 4, "prefix", 0.0),
        ):
            branches.append(
                source.with_only_columns(
                    word_id.label("word_id"),
                    surface.label("surface"),
                    literal(priority).label("rank"),
                    literal(score).label("score"),
                    literal(category).label("kind"),
                ).where(row.Word.generation_state == "done", condition)
            )
    return (
        await session.execute(
            _matches_statement(
                branches,
                session.bind.dialect.name == "postgresql",
                limit,
            )
        )
    ).all()


async def _fuzzy(session, key, excluded, limit):
    schema = await session.scalar(TRIGRAM_SCHEMA)
    if schema is None:
        raise RuntimeError("pg_trgm is missing; apply the dictionary baseline migration")
    quoted = session.bind.dialect.identifier_preparer.quote_schema(schema)
    # SET LOCAL semantics: pool connections do not retain per-request settings.
    await session.execute(
        text(
            "SELECT set_config('pg_trgm.similarity_threshold','0.3',true), "
            "set_config('gin_fuzzy_search_limit','0',true)"
        )
    )
    branches = []
    for column, surface, word_id, _rank, _kind, source in _sources():
        branches.append(
            source.with_only_columns(
                word_id.label("word_id"),
                surface.label("surface"),
                literal(5).label("rank"),
                getattr(func, schema)
                .similarity(
                    column,
                    key,
                )
                .label("score"),
                literal("fuzzy").label("kind"),
            ).where(
                row.Word.generation_state == "done",
                row.Word.id.not_in(excluded),
                column.op(f"OPERATOR({quoted}.%)")(key),
            )
        )
    return (await session.execute(_matches_statement(branches, True, limit))).all()


async def search(
    db, cambridge, query: str, include_available: bool = False, *, limit=_LIMIT
) -> SearchResult:
    if not isinstance(query, str) or not query.strip() or len(query) > MAX_QUERY_LENGTH:
        raise InvalidResourceError("invalid lexical query")
    query = query.strip()
    key = answer_key(query)
    source_hits = []
    if include_available:
        if cambridge is None:
            raise InvalidResourceError("Cambridge source is not configured")
        source_hits = await cambridge.search(query)
    ranked: dict[int, tuple[int, float, WordHit]] = {}
    async with db.transaction() as session:
        for word, rank, score, kind, surface in await _lexical(session, key, limit):
            _offer(ranked, word, rank, score, kind, surface)
        if len(key) >= 3 and (
            len(ranked) < limit or any(rank >= 3 for rank, _, _ in ranked.values())
        ):
            await _patterns(session, key, query, ranked, limit=limit)
        if len(key) >= 3 and len(ranked) < limit and session.bind.dialect.name == "postgresql":
            for word, rank, score, kind, surface in await _fuzzy(session, key, set(ranked), limit):
                _offer(ranked, word, rank, score, kind, surface)
            _trim(ranked, limit)
        consumed = set()
        if source_hits:
            consumed = set(
                (
                    await session.scalars(
                        select(row.WordSource.source_id)
                        .join(row.Word, row.Word.id == row.WordSource.word_id)
                        .where(
                            row.Word.generation_state == "done",
                            row.WordSource.source_id.in_([hit.id for hit in source_hits]),
                        )
                    )
                ).all()
            )
    hits = [data[2] for _, data in sorted(ranked.items(), key=_rank_key)[:limit]]
    available = []
    if include_available:
        available = [
            AvailableHit(encode_available_id(hit.id), hit.display, hit.entry_type)
            for hit in source_hits
            if hit.id not in consumed
        ]
    return SearchResult(hits, available)
