"""Trial 12 — the question engine (generate + grade + persist).

Turns a done dictionary word into vocabulary questions across four formats, then
grades answers. The three axes are wired through ``answer_kind`` and the engine is
a pure dispatcher: each format is a self-contained plugin that owns its own
generation, grading, and persistence decision.

  * definition_mcq   rule  -> which word means <definition>?          (ephemeral)
  * cloze            rule  -> fill the blank in an example sentence    (ephemeral)
  * contextual_mcq   llm   -> MCQ from a novel context                 (PERSISTS)
  * use_in_sentence  rule  -> write a sentence; graded by an llm rubric (ephemeral)

Only ``contextual_mcq`` persists (it calls the store itself), so a re-run lists it
back at zero token cost. Like the sibling trials the FIRST lookup of a word spends
tokens; re-runs read the local cache.

    uv run python examples/12_question_engine.py [word]

Needs the LLM env (LEXI_LLM_*). Prints a notice and exits if the word can't be
resolved.
"""

import asyncio
import sys

from _common import aclose, build_lexicon, get_demo_settings, lookup

from lexi_ai.questions.distractors import DistractorProvider
from lexi_ai.questions.engine import QuestionEngine
from lexi_ai.questions.repository import QuestionRepository
from lexi_ai.questions.schemas import GeneratedMCQ, Judgment

DEFAULT_WORD = "eloquent"

SEED_FORMATS = ["definition_mcq", "cloze", "contextual_mcq", "use_in_sentence"]


def _build_question_engine(lex) -> QuestionEngine:
    """A QuestionEngine wired to the proxy-compatible runnables.

    Mirrors ``_common.build_structured_llm``: the proxy honors
    ``method="function_calling"`` + a ``reasoning_effort`` reliably, unlike the
    library-default ``json_schema`` method. Model/URL/key/temperature all come
    from env — nothing about the model is hardcoded.
    """
    from langchain_openai import ChatOpenAI
    from pydantic import SecretStr

    settings = get_demo_settings()

    def bind(schema):
        llm = ChatOpenAI(
            base_url=settings.llm_base_url,
            api_key=SecretStr(settings.llm_api_key),
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            reasoning_effort=settings.llm_reasoning_effort,
        )
        return llm.with_structured_output(schema, method="function_calling")

    return QuestionEngine(
        QuestionRepository(lex._session_factory),
        DistractorProvider(lex._repo, lex._embedder),
        llm=bind(GeneratedMCQ),
        judge_llm=bind(Judgment),
    )


def _print_question(i: int, q) -> None:
    print(f"  {i}. [{q.format} · {q.answer_kind}] id={q.id}")
    payload = q.payload
    if "options" in payload:
        print(f"     {payload['stem']}")
        for j, opt in enumerate(payload["options"]):
            mark = "*" if j == payload["correct_index"] else " "
            print(f"       {mark} ({j}) {opt}")
    elif "stem_with_blank" in payload:
        print(f"     {payload['stem_with_blank']}")
        print(f"       answer: {payload['answer_norm']}")
    else:  # use_in_sentence
        print(f"     {payload['prompt']}")
        print(f"       rubric: {payload['rubric']}")


async def main() -> None:
    word = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_WORD
    lex = build_lexicon()
    await lex.init()
    try:
        print(f"=== generating dictionary entry for {word!r} ===")
        entry = await lookup(lex, word)
        if entry is None:
            print(f"◇ {word!r} not in Cambridge — try another word.")
            return
        engine = _build_question_engine(lex)

        print(f"\n=== generating one question per format for {entry.display!r} ===")
        questions = await engine.generate(entry, formats=SEED_FORMATS, n=1)
        by_format = {q.format: q for q in questions}
        for i, fmt in enumerate(SEED_FORMATS, 1):
            q = by_format.get(fmt)
            if q is None:
                print(f"  {i}. [{fmt}] (no question — thin source material, best-effort)")
                continue
            _print_question(i, q)

        # Grade a correct and a wrong answer on the definition MCQ, if present.
        mcq = by_format.get("definition_mcq")
        if mcq is not None:
            correct = mcq.payload["correct_index"]
            wrong = (correct + 1) % len(mcq.payload["options"])
            good = await engine.grade(mcq, correct)
            bad = await engine.grade(mcq, wrong)
            print("\n=== grading the definition MCQ ===")
            print(f"  answer {correct} (correct): correct={good.correct} score={good.score}")
            print(f"  answer {wrong} (wrong):   correct={bad.correct} score={bad.score}")

        # Grade the free-text question through the llm rubric judge, if present.
        uis = by_format.get("use_in_sentence")
        if uis is not None:
            sample = f"The {entry.display} speaker held the whole room in rapt attention."
            verdict = await engine.grade(uis, sample)
            print("\n=== grading a free-text answer (llm rubric judge) ===")
            print(f"  answer: {sample!r}")
            print(f"  correct={verdict.correct} score={verdict.score} — {verdict.feedback}")

        # Show 0-token reuse: the persisted contextual_mcq lists back from the DB.
        persisted = await engine.list(entry.word_id, fmt="contextual_mcq")
        print(f"\n=== persisted questions for {entry.display!r} (0-token reuse) ===")
        print(f"  contextual_mcq rows in DB: {len(persisted)}")
        for q in persisted:
            print(f"    id={q.id}: {q.payload['stem']}")
    finally:
        await aclose(lex)


if __name__ == "__main__":
    asyncio.run(main())
