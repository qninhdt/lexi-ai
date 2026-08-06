"""Read-only Cambridge source access (Phase 3).

Opens the Cambridge SQLite file with a ``file:...?mode=ro`` URI so writes are
impossible. All sqlite calls are synchronous and wrapped in
``asyncio.to_thread`` to stay non-blocking on the event loop.

Cambridge schema (verified against ``./data``):
  words(id, word[slug], display_form, entry_type, status)
  entries(id, word_id, pos, ...)
  senses(id, entry_id, sense_order, guideword, definition, cefr_level,
         domain, labels, phrase_title)
  examples(id, sense_id, example, is_extra)
  sense_synonyms(sense_id, synonym, is_antonym)
  word_alternatives(word_id, alternative_word, alternative_type)
"""

import asyncio
import re
import sqlite3
import threading
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field

from lexi_ai.normalize import match_key

# A Cambridge display slot ``X/Y`` where BOTH sides are placeholder words is an
# alternation *inside one lemma* (``look after someone/something``), not two
# lemmas. We expand it in place (one full-phrase key per alternative) so natural
# input like ``look after somebody`` resolves. Splitting the whole display on
# ``/`` instead is wrong: it produces stub fragments ("something") that collapse
# hundreds of unrelated idioms onto one match_key.
_PLACEHOLDER_WORD = r"someone|something|somebody|sb|sth|oneself|yourself"
_PLACEHOLDER_SLOT_RE = re.compile(
    rf"\b({_PLACEHOLDER_WORD})/({_PLACEHOLDER_WORD})\b", re.IGNORECASE
)


def _surface_keys(slug: str, display_form: str) -> set[str]:
    """All normalized surface keys a Cambridge row should resolve from.

    ``slug`` with hyphens as spaces, the full ``display_form``, and — when the
    display carries a placeholder slot ``X/Y`` — one variant per alternative with
    the slot substituted in place. Every candidate is ``match_key``-folded.
    """
    keys = {match_key(slug.replace("-", " "))}
    keys |= expand_display_keys(display_form)
    return {k for k in keys if k}


def expand_display_keys(display_form: str) -> set[str]:
    """Match keys a display form should be found under (headword or generated norm).

    The folded display, plus — when it carries a placeholder slot ``X/Y`` — one
    key per alternative with the slot substituted in place. This matters because a
    generated word's ``norm`` ("look after {sb}") folds to a *single*-placeholder
    key while the Cambridge display ("look after someone/something") folds to a
    two-placeholder key; the expansion bridges the two so a generated word is found
    from its display.
    """
    if not display_form:
        return set()
    keys = {match_key(display_form)}
    slot = _PLACEHOLDER_SLOT_RE.search(display_form)
    if slot is not None:
        for alternative in (slot.group(1), slot.group(2)):
            variant = display_form[: slot.start()] + alternative + display_form[slot.end() :]
            keys.add(match_key(variant))
    return {k for k in keys if k}


@dataclass
class CamSense:
    definition: str
    guideword: str | None
    cefr_level: str | None
    pos: str | None
    phrase_title: str | None
    domain: str | None
    cambridge_sense_id: int
    examples: list[str] = field(default_factory=list)
    # Cambridge IPA, per POS (entries are POS-grouped), hard-anchored into the
    # generation prompt so the LLM copies it (LLMs hallucinate IPA badly). None
    # when Cambridge lacks it. Semantic relations (synonyms/antonyms) are NO
    # LONGER anchored — the LLM generates relations itself (selective anchoring).
    ipa_uk: str | None = None
    ipa_us: str | None = None


@dataclass
class CambridgeEntry:
    word_id: int
    word_slug: str
    display_form: str
    entry_type: str | None
    senses: list[CamSense] = field(default_factory=list)
    alternatives: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class CamRef:
    """Lightweight Cambridge word reference from surface resolution (no senses)."""

    word_id: int
    display_form: str
    entry_type: str | None


class CambridgeSource:
    """Read-only reader over the Cambridge SQLite file."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        # Deployment mounts one checksum-verified artifact without SQLite WAL/
        # SHM sidecars. ``immutable=1`` avoids looking for or mutating those
        # sidecars; service readiness has already verified this file's SHA-256.
        self._uri = f"file:{db_path}?mode=ro&immutable=1"
        # Lazily-built {match_key: word_id} index for the fallback lookup, so a
        # direct-miss fetch is O(1) instead of a full-table Python scan. Built
        # once per process; guarded because _fetch_sync runs in worker threads.
        self._key_index: dict[str, int] | None = None
        self._index_lock = threading.Lock()
        # Lazily-built {surface_key: [word_id, ...]} index for resolve_exact.
        # Unlike _key_index (fallback, first-writer-wins single id), this keeps
        # ALL ids per key so genuine homographs surface as multiple candidates.
        self._surface_index: dict[str, list[int]] | None = None
        self._surface_meta: dict[int, tuple[str, str | None]] | None = None
        self._surface_lock = threading.Lock()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._uri, uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    # --- sync bodies (run in a thread) ------------------------------------

    def _fetch_sync(self, key: str) -> CambridgeEntry | None:
        conn = self._connect()
        try:
            # Resolve by slug, display_form, or normalized match_key.
            row = conn.execute(
                "SELECT id, word, display_form, entry_type FROM words "
                "WHERE word = ? OR display_form = ? LIMIT 1",
                (key, key),
            ).fetchone()
            if row is None:
                row = self._match_by_key(conn, key)
            if row is None:
                return None
            return self._build_entry(conn, row)
        finally:
            conn.close()

    def _fetch_by_id_sync(self, word_id: int) -> CambridgeEntry | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT id, word, display_form, entry_type FROM words WHERE id = ?",
                (word_id,),
            ).fetchone()
            if row is None:
                return None
            return self._build_entry(conn, row)
        finally:
            conn.close()

    def _build_entry(self, conn: sqlite3.Connection, row: sqlite3.Row) -> CambridgeEntry:
        entry = CambridgeEntry(
            word_id=row["id"],
            word_slug=row["word"],
            display_form=row["display_form"] or row["word"],
            entry_type=row["entry_type"],
        )
        entry.senses = self._fetch_senses(conn, row["id"])
        entry.alternatives = [
            (r["alternative_word"], r["alternative_type"])
            for r in conn.execute(
                "SELECT alternative_word, alternative_type "
                "FROM word_alternatives WHERE word_id = ?",
                (row["id"],),
            ).fetchall()
        ]
        return entry

    def _match_by_key(self, conn: sqlite3.Connection, key: str) -> sqlite3.Row | None:
        """Fallback: find a row whose display_form/slug shares the match_key.

        Uses a cached {match_key: word_id} index (built once), so this is an O(1)
        dict lookup rather than a full-table NFKD+regex scan on every miss.
        """
        index = self._get_key_index(conn)
        word_id = index.get(match_key(key))
        if word_id is None:
            return None
        return conn.execute(
            "SELECT id, word, display_form, entry_type FROM words WHERE id = ?",
            (word_id,),
        ).fetchone()

    def _get_key_index(self, conn: sqlite3.Connection) -> dict[str, int]:
        if self._key_index is not None:
            return self._key_index
        with self._index_lock:
            if self._key_index is None:
                index: dict[str, int] = {}
                for r in conn.execute("SELECT id, word, display_form FROM words"):
                    surface = r["display_form"] or r["word"]
                    # First writer wins on a collision — mirrors the old scan,
                    # which returned the first matching row.
                    index.setdefault(match_key(surface), r["id"])
                self._key_index = index
        return self._key_index

    def _get_surface_index(
        self, conn: sqlite3.Connection
    ) -> tuple[dict[str, list[int]], dict[int, tuple[str, str | None]]]:
        """Build (once) the {surface_key: [word_id, ...]} + {word_id: (display,
        entry_type)} indexes used by resolve_exact. Keeps every id per key so
        genuine homographs return multiple candidates."""
        if self._surface_index is not None and self._surface_meta is not None:
            return self._surface_index, self._surface_meta
        with self._surface_lock:
            if self._surface_index is None or self._surface_meta is None:
                index: dict[str, list[int]] = {}
                meta: dict[int, tuple[str, str | None]] = {}
                for r in conn.execute("SELECT id, word, display_form, entry_type FROM words"):
                    display = r["display_form"] or r["word"]
                    meta[r["id"]] = (display, r["entry_type"])
                    for key in _surface_keys(r["word"], display):
                        ids = index.setdefault(key, [])
                        if r["id"] not in ids:
                            ids.append(r["id"])
                self._surface_index = index
                self._surface_meta = meta
        return self._surface_index, self._surface_meta

    def _resolve_exact_sync(self, raw: str) -> list[CamRef]:
        conn = self._connect()
        try:
            index, meta = self._get_surface_index(conn)
            ids = index.get(match_key(raw))
            if not ids:
                return []
            return [CamRef(word_id=i, display_form=meta[i][0], entry_type=meta[i][1]) for i in ids]
        finally:
            conn.close()

    def _rank_similar_sync(self, raw: str, limit: int) -> list[tuple[CamRef, float]]:
        """Fuzzy-rank Cambridge words by similarity of raw to their surface keys.

        Scores each distinct surface key with difflib's ratio, keeps the best
        score per word_id, and returns the top ``limit`` (CamRef, score) pairs,
        best first. Deterministic, no LLM.
        """
        from difflib import SequenceMatcher

        conn = self._connect()
        try:
            index, meta = self._get_surface_index(conn)
            target = match_key(raw)
            matcher = SequenceMatcher(a=target)
            best: dict[int, float] = {}
            for key, ids in index.items():
                matcher.set_seq2(key)
                # Cheap length-ratio upper bound gate before the real ratio.
                if matcher.real_quick_ratio() < 0.6 or matcher.quick_ratio() < 0.6:
                    continue
                score = matcher.ratio()
                if score < 0.6:
                    continue
                for i in ids:
                    if score > best.get(i, 0.0):
                        best[i] = score
            top = sorted(best.items(), key=lambda kv: (-kv[1], meta[kv[0]][0]))[:limit]
            return [
                (CamRef(word_id=i, display_form=meta[i][0], entry_type=meta[i][1]), score)
                for i, score in top
            ]
        finally:
            conn.close()

    def _first_definition_sync(self, word_ids: Iterable[int]) -> dict[int, str]:
        """First sense definition per word_id (a gloss for disambiguation).

        One query for the whole batch, not one per id. This runs on the search
        path — a `search()` asking for ten candidates issued ten round trips
        against the reference database, each opening its own statement, and the
        cost scaled with the result count rather than staying flat.

        The window function does what `LIMIT 1` did per word: rank each word's
        senses by the same `(entry_order, sense_order)` and keep the first. SQLite
        has supported window functions since 3.25 (2018), and this file already
        requires a modern SQLite for its FTS5 usage.
        """
        ids = list(word_ids)
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT word_id, definition FROM ("
                "  SELECT e.word_id AS word_id, s.definition AS definition, "
                "         ROW_NUMBER() OVER ("
                "           PARTITION BY e.word_id ORDER BY e.entry_order, s.sense_order"
                "         ) AS rank "
                "  FROM entries e JOIN senses s ON s.entry_id = e.id "
                f"  WHERE e.word_id IN ({placeholders})"
                ") WHERE rank = 1",
                ids,
            ).fetchall()
            return {
                row["word_id"]: row["definition"]
                for row in rows
                if row["definition"]
            }
        finally:
            conn.close()

    def _fetch_senses(self, conn: sqlite3.Connection, word_id: int) -> list[CamSense]:
        senses: list[CamSense] = []
        sense_rows = conn.execute(
            "SELECT s.id, s.definition, s.guideword, s.cefr_level, s.domain, "
            "       s.phrase_title, e.pos, e.pronunciation_uk, e.pronunciation_us "
            "FROM entries e JOIN senses s ON s.entry_id = e.id "
            "WHERE e.word_id = ? "
            "ORDER BY e.entry_order, s.sense_order",
            (word_id,),
        ).fetchall()
        for s in sense_rows:
            examples = [
                r["example"]
                for r in conn.execute(
                    "SELECT example FROM examples WHERE sense_id = ? ORDER BY example_order",
                    (s["id"],),
                ).fetchall()
            ]
            # Semantic relations (sense_synonyms) are deliberately NOT loaded — the
            # LLM generates relations itself now (selective anchoring). Only IPA is
            # hard-anchored, threaded through the SAME entries join (per POS).
            senses.append(
                CamSense(
                    definition=s["definition"],
                    guideword=s["guideword"],
                    cefr_level=s["cefr_level"],
                    pos=s["pos"],
                    phrase_title=s["phrase_title"],
                    domain=s["domain"],
                    cambridge_sense_id=s["id"],
                    examples=examples,
                    ipa_uk=s["pronunciation_uk"],
                    ipa_us=s["pronunciation_us"],
                )
            )
        return senses

    def _iter_surface_forms_sync(self) -> list[tuple[str, str | None]]:
        conn = self._connect()
        try:
            return [
                (r["display_form"] or r["word"], r["entry_type"])
                for r in conn.execute("SELECT word, display_form, entry_type FROM words").fetchall()
            ]
        finally:
            conn.close()

    def _standalone_keys_sync(self) -> set[str]:
        conn = self._connect()
        try:
            keys: set[str] = set()
            for r in conn.execute("SELECT word, display_form FROM words"):
                keys.add(match_key(r["display_form"] or r["word"]))
            return keys
        finally:
            conn.close()

    def _iter_phrase_titles_sync(self) -> list[tuple[str, str | None, str | None]]:
        """Distinct (phrase_title, host_display_form, host_entry_type).

        One host per phrase_title (the first entry that carries it); enough to
        seed a stub and link it to its host word.
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT s.phrase_title AS pt, "
                "       COALESCE(w.display_form, w.word) AS host, "
                "       w.entry_type AS host_type "
                "FROM senses s "
                "JOIN entries e ON s.entry_id = e.id "
                "JOIN words w ON e.word_id = w.id "
                "WHERE s.phrase_title IS NOT NULL AND s.phrase_title != '' "
                "GROUP BY s.phrase_title"
            ).fetchall()
            return [(r["pt"], r["host"], r["host_type"]) for r in rows]
        finally:
            conn.close()

    # --- async surface ----------------------------------------------------

    async def fetch(self, key: str) -> CambridgeEntry | None:
        """Fetch a full Cambridge entry by slug / display form / match_key."""
        return await asyncio.to_thread(self._fetch_sync, key)

    async def fetch_by_id(self, word_id: int) -> CambridgeEntry | None:
        """Fetch a full Cambridge entry by its word_id (from a resolved Suggestion)."""
        return await asyncio.to_thread(self._fetch_by_id_sync, word_id)

    async def resolve_exact(self, raw: str) -> list[CamRef]:
        """Resolve a raw string to Cambridge word(s) by normalized surface form.

        Returns every word whose surface keys (slug, display, expanded placeholder
        slots) fold to ``match_key(raw)``. Empty when nothing matches exactly;
        multiple only for genuine homograph keys. No LLM, no generation.
        """
        return await asyncio.to_thread(self._resolve_exact_sync, raw)

    async def rank_similar(self, raw: str, limit: int = 10) -> list[tuple[CamRef, float]]:
        """Fuzzy-rank Cambridge words by similarity to ``raw`` (best first)."""
        return await asyncio.to_thread(self._rank_similar_sync, raw, limit)

    async def first_definitions(self, word_ids: Iterable[int]) -> dict[int, str]:
        """First sense definition per word_id, for suggestion glosses."""
        return await asyncio.to_thread(self._first_definition_sync, list(word_ids))

    async def standalone_keys(self) -> set[str]:
        """All match_keys that exist as standalone Cambridge ``words`` rows."""
        return await asyncio.to_thread(self._standalone_keys_sync)

    async def phrase_titles(self) -> list[tuple[str, str | None, str | None]]:
        """Distinct (phrase_title, host_display_form, host_entry_type)."""
        return await asyncio.to_thread(self._iter_phrase_titles_sync)

    async def candidates(
        self, generated_done_keys: Iterable[str]
    ) -> AsyncIterator[tuple[str, str | None]]:
        """Yield Cambridge (surface_form, entry_type) not yet generated.

        "Not yet generated" (decision #14) = a Cambridge word whose match_key
        has no ``done`` row in the generated DB. Diff is by match_key in-app
        (no cross-DB join).
        """
        done = set(generated_done_keys)
        forms = await asyncio.to_thread(self._iter_surface_forms_sync)
        for surface, entry_type in forms:
            if match_key(surface) not in done:
                yield surface, entry_type
