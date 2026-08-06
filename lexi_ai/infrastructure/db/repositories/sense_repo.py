"""The ``senses`` aggregate: senses and everything hanging off one sense.

Async safety: relationship collections are never touched on a persistent object,
because that triggers a lazy load outside greenlet context. Children are cleared
with Core deletes and re-inserted with explicit foreign keys.

Relation disambiguation lives in `sense_relation_repo.py` and is mixed back in
below. It was the one part of this file with a vocabulary entirely of its own —
pending edges, candidates, judged decisions — and moving it costs no indirection
because the port surface is unchanged.
"""

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, cast

from sqlalchemy import CursorResult, delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from lexi_ai.infrastructure.db.repositories.asset_repo import AssetRepository
from lexi_ai.constants import WSD_CANDIDATE_CAP, canonical_cambridge_ref
from lexi_ai.domain.models import (
    ResolveCandidate,
    ResolveDecision,
    ResolveOutcome,
    ResolveTask,
    SemanticSenseRow,
    SenseEmbeddingNeed,
    ThemingSense,
)
from lexi_ai.generation.schemas import ExampleGenContext
from lexi_ai.infrastructure.db.asset_gc import collect_word_assets
from lexi_ai.infrastructure.db.models import (
    Collocation,
    Example,
    Sense,
    SenseForm,
    SenseReference,
    SenseRelation,
    Word,
    _utcnow,
)
from lexi_ai.infrastructure.db.repositories.sense_relation_repo import (
    SenseRelationResolutionMixin,
)
from lexi_ai.infrastructure.db.repositories.word_repo import SqlWordRepo
from lexi_ai.infrastructure.db.sanitize import (
    MAX_COLLOCATION,
    MAX_DEFINITION,
    MAX_DOMAIN,
    MAX_EXAMPLE,
    MAX_GLOSS,
    MAX_GUIDEWORD,
    MAX_IPA,
    MAX_SOURCE_REF,
    MAX_SURFACE,
    MAX_USAGE_NOTE,
    clean,
    clean_opt,
)

if TYPE_CHECKING:
    from lexi_ai.generation.schemas import GeneratedSense, GeneratedSenseRelation


class SqlSenseRepo(SenseRelationResolutionMixin):
    """Session-bound implementation of :class:`lexi_ai.domain.ports.SenseRepo`.

    The relation-resolution half of that port comes from the mixin, so this class
    still satisfies it whole; see `sense_relation_repo.py` for why the split is a
    file boundary rather than an interface one.
    """

    def __init__(
        self,
        session: AsyncSession,
        words: SqlWordRepo,
        assets: AssetRepository | None = None,
    ) -> None:
        self._session = session
        self._words = words
        self._assets = assets

    async def sync(
        self, word_id: int, senses: Iterable["GeneratedSense"], cefr_map: dict[str, str]
    ) -> None:
        """Replace a word's senses and every child row beneath them.

        Two invalidation steps must run BEFORE the old senses vanish:

        * Inbound resolved edges pointing at a disappearing sense are demoted back
          to pending. The foreign key clears ``to_sense_id`` on delete, but that
          alone leaves ``resolve_attempted_at`` set, which reads as unresolvable and
          strands the edge outside the queue forever.
        * Inbound edges that are currently unresolvable are re-queued, so a better
          regeneration gets a fresh attempt instead of a permanent false negative.

        Cached assets are collected first too, since the child ids vanish with the
        delete.
        """
        await collect_word_assets(self._session, self._assets, word_id)
        old_sense_ids = list(
            (await self._session.execute(select(Sense.id).where(Sense.word_id == word_id)))
            .scalars()
            .all()
        )
        await self._demote_edges_for_senses(old_sense_ids)
        await self._requeue_unresolvable_inbound(word_id)
        # Deleting senses cascades to references, examples, collocations, and forms.
        await self._session.execute(delete(Sense).where(Sense.word_id == word_id))
        # Two passes with one flush between them, rather than a flush per sense.
        #
        # Children and relation edges need their parent's `sense.id`, which only a
        # flush assigns, so *a* flush is unavoidable — but one flush for the whole
        # batch assigns every id at once. The per-sense version paid a round trip
        # per sense, and a regenerated word carries a dozen: the cost scaled with
        # the size of the entry for no reason but the order of these statements.
        built = [
            (self._build_sense(word_id, order, generated, cefr_map), generated)
            for order, generated in enumerate(senses)
        ]
        self._session.add_all([sense for sense, _ in built])
        await self._session.flush()
        for sense, generated in built:
            self._add_children(sense.id, generated)
            await self._link_sense_relations(sense.id, word_id, generated.relations)
        await self._session.flush()

    async def word_id_for(self, sense_id: int) -> int:
        """Owning word id. Raises when the sense does not exist."""
        return (
            await self._session.execute(select(Sense.word_id).where(Sense.id == sense_id))
        ).scalar_one()

    async def needing_embedding(
        self,
        word_ids: list[int] | None = None,
        limit: int | None = None,
        after_sense_id: int | None = None,
    ) -> list[SenseEmbeddingNeed]:
        """Done senses that are candidates for embedding, oldest id first.

        This store no longer knows which senses carry a vector — the index does —
        so every done sense is a candidate and the caller subtracts what the index
        already holds. Restrict to ``word_ids`` right after generation; omit them
        for a global backfill.

        ``after_sense_id`` resumes past an id already examined, which is what lets
        a caller honour a ``limit`` without loading the whole table. Because the
        already-embedded set lives in the index rather than here, a plain
        ``LIMIT`` could return a page that is entirely embedded already and read
        as "nothing left to do"; paging with this cursor lets the caller keep
        asking until it has enough genuinely-unembedded rows.
        """
        stmt = (
            select(Sense.id, Word.norm, Sense.definition)
            .join(Word, Word.id == Sense.word_id)
            .where(Word.status == "done")
            .order_by(Sense.id)
        )
        if word_ids is not None:
            if not word_ids:
                return []
            stmt = stmt.where(Sense.word_id.in_(word_ids))
        if after_sense_id is not None:
            stmt = stmt.where(Sense.id > after_sense_id)
        if limit is not None:
            stmt = stmt.limit(limit)
        rows = await self._session.execute(stmt)
        return [SenseEmbeddingNeed(sid, norm, definition) for sid, norm, definition in rows]

    async def semantic_rows(self, sense_ids: Sequence[int]) -> list[SemanticSenseRow]:
        """Presentation rows for ranked sense ids, in the order requested.

        Unknown ids are skipped, which is what makes a stale vector harmless: an id
        left behind by a delete or a regeneration drops out of the results instead
        of raising.
        """
        if not sense_ids:
            return []
        rows = await self._session.execute(
            select(
                Sense.id,
                Sense.word_id,
                Word.norm,
                Word.entry_type,
                Sense.definition,
                Sense.tier,
            )
            .join(Word, Word.id == Sense.word_id)
            .where(Word.status == "done", Sense.id.in_(list(sense_ids)))
        )
        by_id = {row[0]: SemanticSenseRow(*row) for row in rows}
        return [by_id[sense_id] for sense_id in sense_ids if sense_id in by_id]

    async def live_sense_ids(self) -> set[int]:
        """Every existing sense id, for pruning vectors whose sense is gone."""
        rows = await self._session.execute(select(Sense.id))
        return {sense_id for (sense_id,) in rows}

    async def ids_for_word(self, word_id: int) -> list[int]:
        """Every sense id of one word, whatever the word's status.

        Read before a delete, so the caller can forget the word's vectors after the
        rows are gone. Status-agnostic on purpose: a pending word can still own
        senses from an earlier successful generation.
        """
        rows = await self._session.execute(
            select(Sense.id).where(Sense.word_id == word_id).order_by(Sense.id)
        )
        return [sense_id for (sense_id,) in rows]

    async def example_context(self, sense_id: int) -> tuple[ExampleGenContext, list[str]] | None:
        """The context and existing examples a targeted generator needs for one sense.

        Existing example texts come back alongside the context rather than inside it,
        so the context stays a pure fact carrier while the caller can still feed the
        texts back for soft de-duplication. Returns ``None`` for an unknown sense.
        """
        row = (
            await self._session.execute(
                select(Sense.definition, Sense.pos, Sense.guideword, Sense.tier).where(
                    Sense.id == sense_id
                )
            )
        ).first()
        if row is None:
            return None
        forms = (
            await self._session.execute(
                select(SenseForm.inf, SenseForm.surface)
                .where(SenseForm.sense_id == sense_id)
                .order_by(SenseForm.form_order)
            )
        ).all()
        examples = (
            (
                await self._session.execute(
                    select(Example.text)
                    .where(Example.sense_id == sense_id)
                    .order_by(Example.example_order)
                )
            )
            .scalars()
            .all()
        )
        context = ExampleGenContext(
            definition=row[0],
            pos=row[1],
            guideword=row[2],
            tier=row[3],
            forms=[(inf, surface) for inf, surface in forms],
        )
        return context, list(examples)

    async def append_examples(self, sense_id: int, texts: Sequence[str]) -> int:
        """Append cleaned, non-empty texts after the current highest order.

        Never deletes or overwrites existing examples, unlike the whole-word path.
        """
        current_max = (
            await self._session.execute(
                select(func.max(Example.example_order)).where(Example.sense_id == sense_id)
            )
        ).scalar_one_or_none()
        order = (current_max + 1) if current_max is not None else 0
        inserted = 0
        for text in texts:
            cleaned = clean(text, MAX_EXAMPLE)
            if not cleaned:
                continue
            self._session.add(Example(sense_id=sense_id, text=cleaned, example_order=order))
            order += 1
            inserted += 1
        await self._session.flush()
        return inserted

    async def for_theming(self, word_id: int) -> list[ThemingSense]:
        """A word's senses in deterministic prompt order.

        The caller passes these ids back when persisting the themed result in the
        same order it numbered them, so themed index ``i`` maps to sense ``i``.
        """
        rows = await self._session.execute(
            select(Sense.id, Sense.definition, Sense.pos, Sense.guideword, Sense.tier)
            .where(Sense.word_id == word_id)
            .order_by(Sense.sense_order, Sense.id)
        )
        return [ThemingSense(*row) for row in rows]

    def _build_sense(
        self, word_id: int, order: int, generated: "GeneratedSense", cefr_map: dict[str, str]
    ) -> Sense:
        """Map one generated sense onto a row, cleaning every free-text field.

        ``grammar`` is a set of schema-validated tokens; the column type validates
        them again on write and handles the encoding, so no caller joins by hand.
        ``register`` and ``connotation`` are validated enum values or ``None``.
        """
        return Sense(
            word_id=word_id,
            definition=clean(generated.definition, MAX_DEFINITION),
            tier=generated.tier,
            sense_order=order,
            pos=generated.pos,
            cefr_level=self._resolve_cefr(generated, cefr_map),
            guideword=clean_opt(generated.guideword, MAX_GUIDEWORD),
            grammar=list(generated.grammar),
            register=generated.register,
            connotation=generated.connotation,
            ipa_uk=clean_opt(generated.ipa_uk, MAX_IPA),
            ipa_us=clean_opt(generated.ipa_us, MAX_IPA),
            domain=clean_opt(generated.domain, MAX_DOMAIN),
            usage_note=clean_opt(generated.usage_note, MAX_USAGE_NOTE),
        )

    def _add_children(self, sense_id: int, generated: "GeneratedSense") -> None:
        """Insert the ordered child rows of one sense.

        Every text field is untrusted model output, so each is cleaned; a value that
        cleans away entirely is skipped rather than stored blank.
        """
        for reference in generated.references:
            self._session.add(
                SenseReference(
                    sense_id=sense_id,
                    source=reference.source,
                    source_ref=clean(reference.source_ref, MAX_SOURCE_REF),
                )
            )
        for order, example in enumerate(generated.examples):
            text = clean(example, MAX_EXAMPLE)
            if text:
                self._session.add(Example(sense_id=sense_id, text=text, example_order=order))
        for order, collocation in enumerate(generated.collocations):
            text = clean(collocation, MAX_COLLOCATION)
            if text:
                self._session.add(
                    Collocation(sense_id=sense_id, text=text, collocation_order=order)
                )
        for order, form in enumerate(generated.forms):
            surface = clean(form.surface, MAX_SURFACE)
            if surface:
                self._session.add(
                    SenseForm(sense_id=sense_id, inf=form.inf, surface=surface, form_order=order)
                )

    @staticmethod
    def _resolve_cefr(generated: "GeneratedSense", cefr_map: dict[str, str]) -> str | None:
        """Cambridge-first CEFR: a Cambridge reference wins over the model's guess.

        Both the reference id and the map keys are canonicalized, so the labelled and
        bare forms collapse to one key and the rule cannot fall through when the model
        echoes the form it was shown.
        """
        for reference in generated.references:
            if reference.source == "cambridge":
                key = canonical_cambridge_ref(reference.source_ref)
                if key in cefr_map:
                    return cefr_map[key]
        return generated.cefr_level
