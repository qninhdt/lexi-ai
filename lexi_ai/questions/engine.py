"""The question engine — a dispatcher over format plugins.

The engine owns no format logic and no persistence rule. It assembles a
:class:`QuestionContext` of capabilities and delegates to the plugin the registry
names for a format; a plugin persists itself (via ``ctx.store``) if it wants to,
invisibly to the engine. There is deliberately no ``if ...: insert`` here and no
branch on which backend a plugin uses — "who persists" and "rule vs llm" are
plugin concerns the engine cannot see.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import replace

from lexi_ai.llm import StructuredLLM
from lexi_ai.questions.base import REGISTRY, QuestionContext, QuestionFormat, TtsPort
from lexi_ai.questions.distractors import DistractorProvider
from lexi_ai.questions.repository import QuestionRepository
from lexi_ai.read_models import BatchResult, Entry, Question, Score


class QuestionEngine:
    """Generate, list, get, delete, and grade questions for done words."""

    def __init__(
        self,
        repo: QuestionRepository,
        distractors: DistractorProvider,
        llm: StructuredLLM | None = None,
        judge_llm: StructuredLLM | None = None,
        tts: TtsPort | None = None,
    ):
        self._repo = repo
        self._distractors = distractors
        self._llm = llm
        self._judge = judge_llm
        self._tts = tts
        self._plugins: dict[str, QuestionFormat] = {}

    # --- generate / grade (dispatch to plugins) ---------------------------

    async def generate(
        self,
        entry: Entry,
        formats: Sequence[str] | None = None,
        n: int = 1,
        *,
        persist: bool = False,
    ) -> list[Question]:
        """Generate up to ``n`` questions per requested format for a done word.

        ``n`` is a best-effort MAX: a plugin returns fewer when source material is
        thin, and never fabricates. In v1 every plugin targets the entry's core
        sense and returns at most one question, so ``n`` only gates ``n <= 0``;
        per-sense fan-out (scaling toward ``n``) is a later enhancement. A plugin
        that persists does so itself; the returned list mixes persisted questions
        (carrying a DB id) and ephemeral ones (``id=None``) transparently.
        """
        if entry.status != "done":
            raise ValueError(
                f"cannot generate questions for a {entry.status!r} word — generate the word first"
            )
        ctx = self._ctx(entry)
        requested = list(formats) if formats is not None else list(REGISTRY)
        out: list[Question] = []
        for fmt in requested:
            generated = await self._plugin(fmt).generate(ctx, n)
            # Library callers may keep ephemeral questions, but a service-issued
            # learner question must have an immutable id for later authoritative
            # lookup and grading.  Plugins that already persist stay untouched.
            if persist:
                if self._repo is None:
                    raise RuntimeError("question persistence is not configured")
                generated = [
                    question if question.id is not None else await self._repo.insert(question)
                    for question in generated
                ]
            out.extend(generated)
        return out

    async def generate_for_sense(
        self,
        entry: Entry,
        sense_id: int,
        formats: Sequence[str] | None = None,
        n: int = 1,
        *,
        persist: bool = False,
    ) -> list[Question]:
        """Generate questions bound to exactly one persisted sense.

        Format plugins deliberately consume an entry's core sense.  Reordering a
        detached read model keeps that plugin contract intact while making the
        selected sense explicit at the service boundary.  Whole-word formats are
        excluded because Pycil's review queue must retain one sense provenance.
        """
        selected = next((sense for sense in entry.senses if sense.sense_id == sense_id), None)
        if selected is None:
            raise ValueError("sense does not belong to entry")
        scoped = replace(entry, senses=[selected, *(s for s in entry.senses if s is not selected)])
        questions = await self.generate(scoped, formats, n, persist=persist)
        return [question for question in questions if question.sense_id == sense_id]

    async def grade(self, question: Question, answer: object) -> Score:
        """Grade an answer to any question — persisted or freshly generated.

        Pure dispatch by ``question.format`` to the plugin's ``grade``; needs no DB
        (a client can grade a just-generated ephemeral question).
        """
        return await self._plugin(question.format).grade(self._ctx(None), question, answer)

    async def grade_many(
        self, pairs: list[tuple[Question, object]], *, concurrency: int = 5
    ) -> list[BatchResult]:
        """Batch :meth:`grade` — one :class:`BatchResult` per ``(question, answer)``
        pair, in order, up to ``concurrency`` in flight. Grading may hit an LLM
        judge (fallible per item), so one pair's failure never aborts the rest."""
        if not pairs:
            return []
        sem = asyncio.Semaphore(concurrency)

        async def _one(pair: tuple[Question, object]) -> Score:
            question, answer = pair
            async with sem:
                return await self.grade(question, answer)

        raw = await asyncio.gather(*(_one(p) for p in pairs), return_exceptions=True)
        return [
            BatchResult(key=pair, error=str(r))
            if isinstance(r, Exception)
            else BatchResult(key=pair, value=r)
            for pair, r in zip(pairs, raw, strict=True)
        ]

    # --- CRUD passthrough (persisted questions) ---------------------------

    async def list(self, word_id: int, fmt: str | None = None) -> list[Question]:
        """Persisted questions for a word, optionally filtered by format. FREE."""
        return await self._repo.list_for_word(word_id, fmt)

    async def list_for_sense(self, sense_id: int, fmt: str | None = None) -> list[Question]:
        """Persisted, single-sense questions suitable for learner delivery."""
        return await self._repo.list_for_sense(sense_id, fmt)

    async def get(self, question_id: int) -> Question | None:
        """A persisted question by id, or ``None``. FREE."""
        return await self._repo.get(question_id)

    async def delete(self, question_id: int) -> bool:
        """Delete a persisted question; return whether a row was removed."""
        return await self._repo.delete(question_id)

    # --- internals --------------------------------------------------------

    def _ctx(self, entry: Entry | None) -> QuestionContext:
        """Assemble the capability bag handed to every plugin this call."""
        return QuestionContext(
            entry=entry,
            distractors=self._distractors,
            llm=self._llm,
            judge=self._judge,
            store=self._repo,
            tts=self._tts,
        )

    def _plugin(self, fmt: str) -> QuestionFormat:
        """Resolve (and cache) the plugin instance for a format id."""
        plugin = self._plugins.get(fmt)
        if plugin is None:
            if fmt not in REGISTRY:
                raise KeyError(f"no plugin registered for format {fmt!r}")
            plugin = REGISTRY[fmt].make_plugin()
            self._plugins[fmt] = plugin
        return plugin
