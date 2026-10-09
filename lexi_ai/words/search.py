"""One startup-built Tantivy index, updated when a Word is published."""

import asyncio
import json
from collections import defaultdict

import tantivy
from sqlalchemy import select

from lexi_ai import schema as row
from lexi_ai.config import MAX_QUERY_LENGTH
from lexi_ai.errors import InvalidResourceError
from lexi_ai.models import ReferenceHit, SearchResult, WordHit
from lexi_ai.patterns import matches_pattern, surface_head_key
from lexi_ai.references.cambridge import encode_reference_id
from lexi_ai.text import answer_key
from lexi_ai.vocab import EntryType, GenerationState, MatchKind


class Search:
    def __init__(self, db, cambridge):
        self.db, self.cambridge = db, cambridge
        self.index = self.writer = self.snapshot = None
        self._write_lock = asyncio.Lock()

    async def _projection(self, word_id=None):
        ids = select(row.Word.id).where(row.Word.generation_state == GenerationState.DONE)
        if word_id is not None:
            ids = ids.where(row.Word.id == word_id)
        async with self.db.read() as session:
            words = (
                await session.execute(
                    select(row.Word.id, row.Word.lemma, row.Word.entry_type).where(
                        row.Word.id.in_(ids)
                    )
                )
            ).all()
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
        return words, aliases, forms, patterns, consumed

    @staticmethod
    def _documents(words, aliases, forms, patterns, consumed, reference):
        entries = {
            f"LEXI:{identifier}": (identifier, lemma, kind, [(0, lemma)])
            for identifier, lemma, kind in words
        }
        for identifier, surface in aliases:
            entries[f"LEXI:{identifier}"][3].append((1, surface))
        sense_forms = defaultdict(list)
        for identifier, sense_id, surface in forms:
            entries[f"LEXI:{identifier}"][3].append((2, surface))
            sense_forms[sense_id].append(surface)
        for hit, surfaces, reference_forms in reference:
            if hit.id not in consumed:
                entries[f"CAMBRIDGE:{hit.id}"] = (
                    hit.id,
                    hit.display,
                    EntryType(hit.entry_type.upper()),
                    [
                        *((0 if i < 2 else 1, surface) for i, surface in enumerate(surfaces)),
                        *((2, form) for form in reference_forms),
                    ],
                )

        def document(identity, rank, surface):
            identifier, label, kind, _ = entries[identity]
            source = identity.partition(":")[0]
            doc = tantivy.Document(
                identity=identity,
                source=source,
                label=label,
                entry_type=str(kind),
                matched_surface=surface,
                order=(
                    f"{int(source != 'LEXI')}{rank}{answer_key(label)}\0"
                    f"{identifier:020d}\0{answer_key(surface)}"
                ),
            )
            doc.add_unsigned("identifier", identifier)
            doc.add_unsigned("surface_rank", rank)
            return doc

        for identity, (_, _, _, surfaces) in entries.items():
            for rank, surface in sorted(set(surfaces)):
                doc = document(identity, rank, surface)
                doc.add_text("surface", answer_key(surface))
                yield doc
        for identifier, sense_id, pattern, head in patterns:
            identity = f"LEXI:{identifier}"
            lemma = entries[identity][1]
            word_head, _, tail = answer_key(lemma).partition(" ")
            licensed = [
                first
                for first, _, rest in (
                    answer_key(form).partition(" ") for form in sense_forms[sense_id]
                )
                if rest == tail
            ]
            doc = document(identity, 3, pattern)
            doc.add_text("forms", json.dumps({word_head: licensed}))
            for candidate in {head, *(surface_head_key(form) for form in sense_forms[sense_id])}:
                doc.add_text("head", candidate or "\0")
            yield doc

    @staticmethod
    def _build(words, aliases, forms, patterns, consumed, reference):
        builder = tantivy.SchemaBuilder()
        for name in ("identity", "source", "surface", "head"):
            builder.add_text_field(
                name, stored=name == "identity", tokenizer_name="raw", index_option="basic"
            )
        for name in ("label", "entry_type", "matched_surface", "forms"):
            builder.add_text_field(name, stored=True, tokenizer_name="raw", index_option="basic")
        builder.add_text_field("order", fast=True, tokenizer_name="raw")
        builder.add_unsigned_field("identifier", stored=True)
        builder.add_unsigned_field("surface_rank", stored=True)
        index = tantivy.Index(builder.build())
        writer = index.writer(heap_size=15_000_000, num_threads=1)
        for doc in Search._documents(words, aliases, forms, patterns, consumed, reference):
            writer.add_document(doc)
        writer.commit()
        index.reload()
        return index, writer, (index, index.searcher())

    async def start(self):
        async with self._write_lock:
            if self.snapshot is not None:
                return
            reference = await self.cambridge.projection() if self.cambridge is not None else []
            projection = await self._projection()
            self.index, self.writer, self.snapshot = await asyncio.to_thread(
                self._build, *projection, reference
            )

    async def update(self, word_id):
        self._ready()
        async with self._write_lock:
            projection = await self._projection(word_id)

            def publish():
                self.writer.delete_documents_by_term("identity", f"LEXI:{word_id}")
                for identifier in projection[4]:
                    self.writer.delete_documents_by_term("identity", f"CAMBRIDGE:{identifier}")
                for doc in self._documents(*projection, []):
                    self.writer.add_document(doc)
                self.writer.commit()
                self.index.reload()
                return self.index, self.index.searcher()

            publication = asyncio.create_task(asyncio.to_thread(publish))
            try:
                self.snapshot = await asyncio.shield(publication)
            except asyncio.CancelledError:
                # Finish the native write before shutdown closes its writer.
                self.snapshot = await publication
                raise

    def _ready(self):
        if self.snapshot is None:
            raise RuntimeError("Search is not started; call Lexicon.start() first")

    @staticmethod
    def _find(snapshot, query, source, limit):
        index, searcher = snapshot
        schema, key = index.schema, answer_key(query)
        ranked = {}

        def candidate(doc, tier):
            identifier, label = doc["identifier"][0], doc["label"][0]
            generated = doc["identity"][0].startswith("LEXI:")
            kind, surface = tuple(MatchKind)[tier], doc["matched_surface"][0]
            hit = (
                WordHit(identifier, label, EntryType(doc["entry_type"][0]), kind, surface)
                if generated
                else ReferenceHit(
                    encode_reference_id(identifier),
                    label,
                    EntryType(doc["entry_type"][0]),
                    kind,
                    surface,
                )
            )
            order = (
                tier,
                int(not generated),
                doc["surface_rank"][0],
                -1.0,
                answer_key(label),
                identifier,
            )
            identity = doc["identity"][0]
            if identity not in ranked or order < ranked[identity][0]:
                ranked[identity] = (order, hit)

        def filtered(branch):
            if source is None:
                return branch
            return tantivy.Query.boolean_query(
                [
                    (tantivy.Occur.Must, branch),
                    (tantivy.Occur.Must, tantivy.Query.term_query(schema, "source", source)),
                ]
            )

        literal = "".join("\\" + char if char in r"\.^$|?*+(){}[]" else char for char in key)
        for tier in range(4 if len(key) >= 3 else 3):
            if tier == 1 and source != "CAMBRIDGE":
                heads = tantivy.Query.boolean_query(
                    [
                        (tantivy.Occur.Should, tantivy.Query.term_query(schema, "head", head))
                        for head in {surface_head_key(key) or "\0", "\0"}
                    ]
                )
                total = searcher.search(heads, limit=1).count
                if total and sum(order[1] == 0 for order, _ in ranked.values()) < limit:
                    for _, address in searcher.search(heads, limit=total, count=False).hits:
                        doc = searcher.doc(address)
                        if doc["identity"][0] not in ranked and matches_pattern(
                            doc["matched_surface"][0], query, forms=json.loads(doc["forms"][0])
                        ):
                            candidate(doc, 0)
            if tier and len(ranked) >= limit:
                break
            if tier == 0:
                branch = tantivy.Query.term_query(schema, "surface", key)
            elif tier == 1:
                branch = tantivy.Query.regex_query(schema, "surface", literal + ".*")
            elif tier == 2:
                branch = tantivy.Query.regex_query(schema, "surface", ".*" + literal + ".*")
            else:
                branch = tantivy.Query.fuzzy_term_query(
                    schema, "surface", key, distance=1 if len(key) < 6 else 2
                )
            # Raw surface scores tie within each tier; a constant score avoids a redundant probe.
            combined = filtered(tantivy.Query.const_score_query(branch, 1.0))
            fetched, seen, size = 0, set(), limit
            while len(seen) < limit:
                hits = searcher.search(
                    combined,
                    limit=size,
                    count=False,
                    order_by_field="order",
                    order=tantivy.Order.Asc,
                ).hits
                for _, address in hits[fetched:]:
                    doc = searcher.doc(address)
                    identity = doc["identity"][0]
                    if identity in seen:
                        continue
                    seen.add(identity)
                    candidate(doc, tier)
                    if len(seen) >= limit:
                        break
                if len(hits) < size:
                    break
                fetched, size = size, size * 2
        return sorted(ranked.values(), key=lambda item: item[0])[:limit]

    async def search(self, query, include_reference=False, *, limit=30):
        if not isinstance(query, str) or not query.strip() or len(query) > MAX_QUERY_LENGTH:
            raise InvalidResourceError("invalid lexical query")
        if type(limit) is not int or not 1 <= limit <= 100:
            raise InvalidResourceError("invalid search limit")
        self._ready()
        snapshot = self.snapshot
        candidates = await asyncio.to_thread(
            self._find, snapshot, query, None if include_reference else "LEXI", limit
        )
        return SearchResult([hit for _, hit in candidates])

    async def close(self):
        async with self._write_lock:
            self.snapshot = None
            writer, self.writer = self.writer, None
            self.index = None
            if writer is not None:
                await asyncio.to_thread(writer.wait_merging_threads)


async def search(db, query, include_reference=False, *, limit=30):
    if db.search_index is None:
        raise RuntimeError("Search is not started; call Lexicon.start() first")
    return await db.search_index.search(query, include_reference, limit=limit)
