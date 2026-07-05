"""The question engine — a dispatcher over format plugins.

The engine owns no format logic and no persistence rule. It assembles a
:class:`QuestionContext` of capabilities and delegates to the plugin the registry
names for a format; a plugin persists itself (via ``ctx.store``) if it wants to,
invisibly to the engine. There is deliberately no ``if ...: insert`` here and no
branch on which backend a plugin uses — "who persists" and "rule vs llm" are
plugin concerns the engine cannot see.
"""

import asyncio
from collections.abc import Sequence

from langchain_core.runnables import Runnable

from lexi_ai.questions.base import REGISTRY, QuestionContext, QuestionFormat
from lexi_ai.questions.distractors import DistractorProvider
from lexi_ai.questions.repository import QuestionRepository
from lexi_ai.read_models import BatchResult, Entry, Question, Score


class QuestionEngine:
    """Generate, list, get, delete, and grade questions for done words."""

    def __init__(
        self,
        repo: QuestionRepository,
        distractors: DistractorProvider,
        llm: Runnable | None = None,
        judge_llm: Runnable | None = None,
    ):
        self._repo = repo
        self._distractors = distractors
        self._llm = llm
        self._judge = judge_llm
        self._plugins: dict[str, QuestionFormat] = {}

    # --- generate / grade (dispatch to plugins) ---------------------------

    async def generate(
        self,
        entry: Entry,
        formats: Sequence[str] | None = None,
        n: int = 1,
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
            out.extend(await self._plugin(fmt).generate(ctx, n))
        return out

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
