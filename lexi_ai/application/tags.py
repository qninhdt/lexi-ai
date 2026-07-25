"""Curating the topic vocabulary the model authors.

Tags are created as a side effect of generation, so these are the operations that
let an owner clean up afterwards. None of them calls a provider.
"""

from collections.abc import Callable, Sequence

from lexi_ai.domain.ports import UnitOfWork


class TagService:
    """Tag curation use cases over the unit of work."""

    def __init__(self, uow_factory: Callable[[], UnitOfWork]) -> None:
        self._uow = uow_factory

    async def rename(self, tag: str, *, name: str | None = None, title: str | None = None) -> bool:
        """Change a tag's display text. The dedup key is immutable, so this never
        merges or re-keys — see :meth:`merge` for that."""
        async with self._uow() as uow:
            renamed = await uow.tags.rename(tag, name=name, title=title)
            await uow.commit()
            return renamed

    async def delete(self, tag: str) -> bool:
        """Delete a tag. Tagged words survive and simply lose this one topic."""
        async with self._uow() as uow:
            deleted = await uow.tags.delete(tag)
            await uow.commit()
            return deleted

    async def merge(self, sources: Sequence[str], into: str) -> int:
        """Fold source tags into a destination that must already exist.

        Returns how many word-tag links were re-pointed; a word already carrying
        both tags contributes none, because that link would collide.
        """
        async with self._uow() as uow:
            moved = await uow.tags.merge(list(sources), into)
            await uow.commit()
            return moved
