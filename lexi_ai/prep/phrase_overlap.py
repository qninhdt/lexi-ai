"""Phrase-overlap data-prep (Phase 7).

Cambridge stores many multi-word units twice: as an inline ``phrase_title`` on a
host word's sense AND (often) as a standalone ``words`` row. This one-off prep
classifies every distinct ``phrase_title`` by ``match_key``:

- overlap  (a standalone ``words`` row shares the key): link the host word to
  that unit as ``part_of_phrasal_family`` — do not regenerate it.
- orphan   (no standalone row): seed a ``pending`` stub in the generated DB so
  it enters the lazy generation queue. Orphans are invisible to the Phase 3
  candidate scan (which reads Cambridge ``words`` only), so this is the ONLY way
  they get generated (consistent with two-DB decision #14).

Actual sense generation stays lazy (Phases 4-6). This only seeds queue/link
state, is idempotent, reads Cambridge read-only, and batches over the ~13.7k
rows.
"""

from dataclasses import dataclass

from lexi_ai.normalize import match_key
from lexi_ai.persistence.repository import Repository
from lexi_ai.references.cambridge import CambridgeSource


@dataclass
class PhraseOverlapReport:
    total: int = 0
    overlap: int = 0
    orphan: int = 0
    stubs_seeded: int = 0
    links_queued: int = 0


class PhraseOverlapPrep:
    def __init__(self, cambridge: CambridgeSource, repository: Repository):
        self._cambridge = cambridge
        self._repo = repository

    async def run(self, batch_size: int = 500) -> PhraseOverlapReport:
        standalone = await self._cambridge.standalone_keys()
        phrase_rows = await self._cambridge.phrase_titles()

        report = PhraseOverlapReport()
        # Dedup by match_key across phrase_titles (folding may merge variants).
        seen: set[str] = set()
        batch: list[tuple[str, str | None, str | None, bool]] = []

        for phrase_title, host, host_type in phrase_rows:
            key = match_key(phrase_title)
            if key in seen:
                continue
            seen.add(key)
            is_overlap = key in standalone
            report.total += 1
            if is_overlap:
                report.overlap += 1
                report.links_queued += 1
            else:
                report.orphan += 1
                report.stubs_seeded += 1
            batch.append((phrase_title, host, host_type, is_overlap))

            if len(batch) >= batch_size:
                await self._flush_batch(batch)
                batch.clear()

        if batch:
            await self._flush_batch(batch)

        return report

    async def _flush_batch(self, batch: list[tuple[str, str | None, str | None, bool]]) -> None:
        async with self._repo.session() as session:
            for phrase_title, host, host_type, is_overlap in batch:
                await self._repo.seed_phrase_unit(
                    session,
                    phrase_title=phrase_title,
                    host_display=host,
                    entry_type=host_type,
                    is_overlap=is_overlap,
                )
