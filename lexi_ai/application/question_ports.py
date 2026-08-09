"""Adapter that resolves a sense to its owning entry."""

from collections.abc import Awaitable, Callable

from sqlalchemy.exc import NoResultFound

from lexi_ai.domain.ports import UnitOfWork
from lexi_ai.read_models import Entry


class SenseEntryLoader:
    """Resolves a sense to its owning entry, for provider-free exposure cards."""

    def __init__(
        self,
        uow_factory: Callable[[], UnitOfWork],
        read_entry: Callable[[int], Awaitable[Entry]],
    ) -> None:
        self._uow = uow_factory
        self._read_entry = read_entry

    async def load_entry(self, sense_id: int) -> Entry | None:
        """The sense's owning entry, or ``None`` when the sense is gone."""
        async with self._uow() as uow:
            try:
                word_id = await uow.senses.word_id_for(sense_id)
            except NoResultFound:
                return None
        return await self._read_entry(word_id)
