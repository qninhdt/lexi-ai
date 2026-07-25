"""Writing one generation result: claim, publish, and record failure.

These three steps deliberately use THREE different transactions, and collapsing
them would each break something specific:

* The claim commits before any provider call. Competing workers detect ownership
  by reading the epoch, so an uncommitted claim is invisible and fences nothing.
* The publish is one atomic transaction over every unit of the result.
* Error recording runs after the publish transaction already rolled back, so it
  needs an independent session; the rolled-back one cannot write.
"""

from collections.abc import Callable

from lexi_ai.domain.errors import StaleGenerationError
from lexi_ai.domain.models import GenerationFence, WordRecord
from lexi_ai.generation.schemas import GeneratedResult
from lexi_ai.infrastructure.db.uow import SqlAlchemyUnitOfWork


class GenerationWriter:
    """Persist generation results through a unit of work per step."""

    def __init__(self, uow_factory: Callable[[], SqlAlchemyUnitOfWork]) -> None:
        self._uow_factory = uow_factory

    async def claim(self, norm: str) -> GenerationFence:
        """Take ownership of a word's next generation epoch and COMMIT it."""
        async with self._uow_factory() as uow:
            fence = await uow.words.claim_next_epoch(norm)
            await uow.commit()
            return fence

    async def publish(
        self,
        result: GeneratedResult,
        cambridge_word_id: int | None = None,
        cambridge_cefr: dict[str, str] | None = None,
        fence: GenerationFence | None = None,
    ) -> list[WordRecord]:
        """Persist every unit of one generation call in a single transaction.

        Units are written in two passes because sibling units link to each other:
        the first pass gives every word a real id, the second adds the links and
        flips each word to done.

        On failure the transaction rolls back and each unit is marked errored in a
        separate transaction, then the original error is re-raised. A superseded
        claim is reported as-is and never marks an error, because the word now
        belongs to a newer generation.
        """
        cefr_map = cambridge_cefr or {}
        try:
            async with self._uow_factory() as uow:
                if fence is not None and not await uow.words.fence_is_current(fence):
                    raise StaleGenerationError("generation claim was superseded")
                word_ids: list[int] = []
                for entry in result.units:
                    word_id = await uow.words.upsert_core(entry, cambridge_word_id)
                    await uow.words.sync_aliases(word_id, entry.aliases)
                    await uow.senses.sync(word_id, entry.senses, cefr_map)
                    await uow.tags.sync(word_id, entry.topics)
                    word_ids.append(word_id)
                await uow.flush()
                for word_id, entry in zip(word_ids, result.units, strict=True):
                    await uow.words.link_related(word_id, entry.related)
                    await uow.words.mark_done(word_id)
                await uow.flush()
                records = await uow.words.records(word_ids)
                await uow.commit()
                return records
        except Exception as exc:  # noqa: BLE001 - recorded then re-raised
            if not isinstance(exc, StaleGenerationError):
                await self._record_error(result, str(exc), fence=fence)
            raise

    async def _record_error(
        self, result: GeneratedResult, message: str, *, fence: GenerationFence | None
    ) -> None:
        """Best-effort: stamp the failed units on an INDEPENDENT session."""
        try:
            async with self._uow_factory() as uow:
                await uow.words.mark_error(
                    [entry.norm for entry in result.units], message, fence=fence
                )
                await uow.commit()
        except Exception:  # noqa: BLE001 - error recording must never mask the cause
            pass
