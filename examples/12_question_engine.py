"""Trial 12 — prepare, retrieve, and evaluate vocabulary questions.

The public question API separates plugin identity (``type_id``) from the UI
contract (``render_kind``). Level 0 is exposure; levels 1-4 are assessments.
Preparation is best-effort and persisted, retrieval is exact and never generates,
and evaluation returns either ``graded`` or ``pending``.

Note what this script CANNOT do: pick the right answer. A ``PresentedQuestion``
carries no correct answer by design, so the demo submits a fixed guess and lets
grading disclose the truth through ``Evaluation.reveal``. That asymmetry is the
answer-safety property, not a limitation of the example.

    uv run python examples/12_question_engine.py [word]
"""

import asyncio
import sys

from _common import aclose, build_lexicon, lookup

from lexi_ai import (
    AnswerSubmission,
    ChoiceResponse,
    Flashcard,
    FreeText,
    PrepareDemand,
    PresentedQuestion,
    SingleChoice,
    TextResponse,
    TextSpan,
)

DEFAULT_WORD = "eloquent"


def _describe_render(question: PresentedQuestion) -> str:
    """One line of whatever the learner would actually be shown."""
    render = question.render
    if isinstance(render, SingleChoice):
        return f"{render.stem}  options={list(render.options)}"
    if isinstance(render, TextSpan):
        bank = f"  bank={list(render.word_bank)}" if render.word_bank else ""
        return f"{render.stem_with_blank}{bank}"
    if isinstance(render, FreeText):
        return render.prompt
    if isinstance(render, Flashcard):
        return f"{render.word}: {render.definition}"
    return repr(render)


def _guess_for(question: PresentedQuestion, word: str) -> AnswerSubmission:
    """A deliberately uninformed submission — the answer is not knowable here."""
    render = question.render
    if isinstance(render, SingleChoice):
        response = ChoiceResponse(selected_index=0)  # first option, blind
    elif isinstance(render, TextSpan):
        response = TextResponse(text=word)
    else:
        response = TextResponse(text=f"The {word} speaker held the whole room in rapt attention.")
    return AnswerSubmission(question_id=question.question_id, response=response)


async def main() -> None:
    word = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_WORD
    lex = build_lexicon()
    await lex.init()
    try:
        entry = await lookup(lex, word)
        if entry is None:
            print(f"◇ {word!r} not in Cambridge — try another word.")
            return

        worker = lex.engine()
        descriptors = worker.question_types()
        print("=== registered question types ===")
        for descriptor in descriptors:
            levels = ",".join(str(level) for level in sorted(descriptor.difficulty_levels))
            print(
                f"  {descriptor.type_id}: render={descriptor.render_kind.value} "
                f"interaction={descriptor.interaction} levels={levels}"
            )

        sense_id = entry.senses[0].sense_id
        if sense_id is None:
            print("◇ entry has no persisted sense id")
            return

        exposure = await worker.retrieve_exposure(sense_id)
        print(f"\n=== level 0 exposure ({exposure.type_id}) ===")
        print(f"  {_describe_render(exposure)}")

        # The PUBLIC demand DTO: its sense_id is a string (the internal
        # questions.base.QuestionDemand takes an int and is not exported).
        demands = [PrepareDemand(str(sense_id), level, 1) for level in range(1, 5)]
        report = await worker.prepare_questions(entry.word_id, demands)
        print("\n=== prepared assessment counts ===")
        for key, count in sorted(report.produced.items()):
            print(f"  sense={key[0]} level={key[1]}: {count}")

        print("\n=== exact retrieval + evaluation (blind submission) ===")
        for descriptor in descriptors:
            if descriptor.interaction != "assessment":
                continue
            level = min(descriptor.difficulty_levels)
            question = await worker.retrieve_question(
                sense_id, level, frozenset(), descriptor.type_id
            )
            if question is None:
                print(f"  {descriptor.type_id}@{level}: unavailable (best-effort)")
                continue
            print(f"  {question.type_id}@{level} shows: {_describe_render(question)}")
            evaluation = await worker.evaluate_answer(
                int(question.question_id), _guess_for(question, entry.display)
            )
            if evaluation is None:
                print("      no evaluation returned")
                continue
            # The reveal is the ONLY place the answer appears, and only after grading.
            print(
                f"      status={evaluation.status} correct={evaluation.correct} "
                f"score={evaluation.score} reveal={evaluation.reveal}"
            )
    finally:
        await aclose(lex)


if __name__ == "__main__":
    asyncio.run(main())
