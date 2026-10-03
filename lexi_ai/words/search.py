"""Shared embedded search for Cambridge and the published Lexi catalog."""

import asyncio
import time
from collections import defaultdict
from difflib import SequenceMatcher

import tantivy
from sqlalchemy import select

from lexi_ai import schema as row
from lexi_ai.cache import Cache
from lexi_ai.config import MAX_QUERY_LENGTH
from lexi_ai.db.session import SessionDatabase
from lexi_ai.errors import InvalidResourceError
from lexi_ai.models import AvailableHit, SearchResult, WordHit
from lexi_ai.patterns import matches_pattern, surface_head_key
from lexi_ai.references.cambridge import encode_available_id
from lexi_ai.text import answer_key
from lexi_ai.vocab import EntryType, GenerationState, MatchKind


class Search:
    def __init__(self, db, cambridge, result_cache_bytes=4 * 1024 * 1024, *, refresh_seconds=30):
        self.db, self.cambridge = db, cambridge
        self.results = Cache(result_cache_bytes)
        self.snapshot = None
        self.reference = None
        self.refresh_seconds = refresh_seconds
        self.refreshed_at = 0
        self._reload_lock = asyncio.Lock()

    async def reload(self, *, include_available=False, force=True):
        # Serialize process-local rebuilds; readers retain the previous immutable
        # snapshot until replacement. This is not a distributed/domain lock.
        async with self._reload_lock:
            if (
                not force
                and self.snapshot is not None
                and time.monotonic() - self.refreshed_at < self.refresh_seconds
                and (not include_available or self.reference is not None)
            ):
                return
            await self._reload(include_available=include_available)

    async def _reload(self, *, include_available):
        loaded_at = time.monotonic()
        reference = self.reference
        if include_available and reference is None:
            if self.cambridge is None:
                raise InvalidResourceError("Cambridge source is not configured")
            reference = await self.cambridge.projection()
        # One consistent publication snapshot. No definitions/examples/Questions.
        async with self.db.transaction() as session:
            if session.bind.dialect.name == "postgresql":
                # Host-owned transactions choose their own isolation level.
                if not isinstance(self.db, SessionDatabase):
                    await session.connection(
                        execution_options={"isolation_level": "REPEATABLE READ"}
                    )
            words = (
                await session.execute(
                    select(row.Word.id, row.Word.lemma, row.Word.entry_type).where(
                        row.Word.generation_state == GenerationState.DONE
                    )
                )
            ).all()
            ids = select(row.Word.id).where(row.Word.generation_state == GenerationState.DONE)
            aliases = (
                await session.execute(
                    select(row.WordAlias.word_id, row.WordAlias.content).where(
                        row.WordAlias.word_id.in_(ids)
                    )
                )
            ).all()
            forms = (
                await session.execute(
                    select(row.Sense.word_id, row.SenseForm.sense_id, row.SenseForm.surface)
                    .join(row.Sense, row.Sense.id == row.SenseForm.sense_id)
                    .where(row.Sense.word_id.in_(ids))
                )
            ).all()
            patterns = (
                await session.execute(
                    select(
                        row.Sense.word_id,
                        row.SensePattern.sense_id,
                        row.SensePattern.content,
                        row.SensePattern.head_key,
                    )
                    .join(row.Sense, row.Sense.id == row.SensePattern.sense_id)
                    .where(row.Sense.word_id.in_(ids))
                )
            ).all()
            consumed = set(
                (
                    await session.scalars(
                        select(row.WordSource.source_id).where(row.WordSource.word_id.in_(ids))
                    )
                ).all()
            )
        snapshot = await asyncio.to_thread(
            self._build, words, aliases, forms, patterns, consumed, reference or []
        )
        self.snapshot = snapshot
        self.reference = reference
        self.refreshed_at = loaded_at
        self.results.clear()

    @staticmethod
    def _build(words, aliases, forms, patterns, consumed, reference):
        builder = tantivy.SchemaBuilder()
        builder.add_text_field("identity", stored=True, tokenizer_name="raw")
        for name in ("source", "available", "lemma", "alias", "form", "surface"):
            builder.add_text_field(name, tokenizer_name="raw", index_option="basic")
        schema = builder.build()
        index = tantivy.Index(schema)
        entries = {}
        for identifier, lemma, kind in words:
            entries[f"LEXI:{identifier}"] = (identifier, lemma, kind, [(MatchKind.LEMMA, lemma)])
        for identifier, surface in aliases:
            entries[f"LEXI:{identifier}"][3].append((MatchKind.ALIAS, surface))
        sense_forms = defaultdict(list)
        for identifier, sense_id, surface in forms:
            entries[f"LEXI:{identifier}"][3].append((MatchKind.FORM, surface))
            sense_forms[sense_id].append(surface)
        for hit, surfaces in reference:
            entries[f"CAMBRIDGE:{hit.id}"] = (
                hit.id,
                hit.display,
                EntryType(hit.entry_type.upper()),
                [(MatchKind.LEMMA, surface) for surface in dict.fromkeys(surfaces)],
            )
        writer = index.writer(heap_size=15_000_000, num_threads=1)
        for identity, (_, _, _, surfaces) in entries.items():
            source = identity.partition(":")[0]
            doc = tantivy.Document(identity=identity, source=source)
            if source == "CAMBRIDGE" and entries[identity][0] not in consumed:
                doc.add_text("available", "yes")
            for kind, surface in set(surfaces):
                key = answer_key(surface)
                doc.add_text(kind.value.lower(), key)
                doc.add_text("surface", key)
            writer.add_document(doc)
        writer.commit()
        writer.wait_merging_threads()
        index.reload()
        by_head = defaultdict(list)
        for identifier, sense_id, pattern, head in patterns:
            lemma = entries[f"LEXI:{identifier}"][1]
            word_head, _, tail = answer_key(lemma).partition(" ")
            licensed = [
                first
                for first, _, rest in (
                    answer_key(form).partition(" ") for form in sense_forms[sense_id]
                )
                if rest == tail
            ]
            item = (identifier, pattern, {word_head: licensed})
            for candidate in {head, *(surface_head_key(form) for form in sense_forms[sense_id])}:
                by_head[candidate].append(item)
        return index, entries, by_head

    @staticmethod
    def _find(snapshot, query, source, limit):
        index, entries, patterns = snapshot
        key = answer_key(query)
        searcher = index.searcher()
        ranked = {}
        # Rust regex syntax: escape metacharacters only (not spaces, %, #, etc.).
        literal = "".join("\\" + char if char in r"\.^$|?*+(){}[]" else char for char in key)
        branches = [
            (rank, kind, tantivy.Query.term_query(index.schema, kind.value.lower(), key))
            for rank, kind in enumerate((MatchKind.LEMMA, MatchKind.ALIAS, MatchKind.FORM))
        ]
        branches.append(
            (
                4,
                MatchKind.PREFIX,
                tantivy.Query.regex_query(index.schema, "surface", literal + ".*"),
            )
        )
        if len(key) >= 3:
            branches.extend(
                [
                    (
                        5,
                        MatchKind.SUBSTRING,
                        tantivy.Query.regex_query(index.schema, "surface", ".*" + literal + ".*"),
                    ),
                    (
                        6,
                        MatchKind.FUZZY,
                        tantivy.Query.fuzzy_term_query(
                            index.schema, "surface", key, distance=1 if len(key) < 6 else 2
                        ),
                    ),
                ]
            )
        for rank, kind, branch in branches:
            clauses = [
                (tantivy.Occur.Must, tantivy.Query.term_query(index.schema, "source", source)),
                (tantivy.Occur.Must, branch),
            ]
            if source == "CAMBRIDGE":
                clauses.append(
                    (tantivy.Occur.Must, tantivy.Query.term_query(index.schema, "available", "yes"))
                )
            combined = tantivy.Query.boolean_query(clauses)
            # One document per identity, so aliases cannot consume the hit limit.
            for score, address in searcher.search(combined, limit=limit).hits:
                identity = searcher.doc(address)["identity"][0]
                identifier, lemma, entry_type, surfaces = entries[identity]
                if kind in (MatchKind.LEMMA, MatchKind.ALIAS, MatchKind.FORM):
                    surface = next(s for k, s in surfaces if k == kind and answer_key(s) == key)
                elif kind is MatchKind.PREFIX:
                    surface = next(s for _, s in surfaces if answer_key(s).startswith(key))
                elif kind is MatchKind.SUBSTRING:
                    surface = next(s for _, s in surfaces if key in answer_key(s))
                else:
                    # Select display evidence only among the engine's matched Word.
                    surface = max(
                        surfaces,
                        key=lambda pair: SequenceMatcher(None, key, answer_key(pair[1])).ratio(),
                    )[1]
                ranked.setdefault(identifier, (rank, -score, lemma, entry_type, kind, surface))
        if source == "LEXI" and len(key) >= 3:
            for identifier, pattern, forms in [
                *patterns.get(surface_head_key(key), []),
                *patterns.get(None, []),
            ]:
                if identifier in ranked and ranked[identifier][0] < 3:
                    continue
                if matches_pattern(pattern, query, forms=forms):
                    _, lemma, entry_type, _ = entries[f"LEXI:{identifier}"]
                    ranked[identifier] = (3, -1, lemma, entry_type, MatchKind.PATTERN, pattern)
        ordered = sorted(ranked.items(), key=lambda pair: (*pair[1][:3], pair[0]))[:limit]
        if source == "CAMBRIDGE":
            return [AvailableHit(encode_available_id(i), hit[2], hit[3]) for i, hit in ordered]
        return [WordHit(i, hit[2], hit[3], hit[4], hit[5]) for i, hit in ordered]

    async def search(self, query, include_available=False, *, limit=30):
        if not isinstance(query, str) or not query.strip() or len(query) > MAX_QUERY_LENGTH:
            raise InvalidResourceError("invalid lexical query")
        if type(limit) is not int or not 1 <= limit <= 100:
            raise InvalidResourceError("invalid search limit")
        await self.reload(
            include_available=include_available, force=isinstance(self.db, SessionDatabase)
        )
        snapshot = self.snapshot
        key = (answer_key(query), include_available, limit)
        if (cached := self.results.get(key, SearchResult)) is not None:
            return cached
        words = await asyncio.to_thread(self._find, snapshot, query, "LEXI", limit)
        available = (
            await asyncio.to_thread(self._find, snapshot, query, "CAMBRIDGE", limit)
            if include_available
            else []
        )
        result = SearchResult(words, available)
        if self.snapshot is snapshot:
            self.results.put(key, result)
        return result


async def search(db, cambridge, query, include_available=False, *, limit=30):
    """Standalone native read; Lexicon owns the reusable process index."""
    engine = db.search_index or Search(db, cambridge)
    return await engine.search(query, include_available, limit=limit)
