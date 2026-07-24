"""Trial 12 — prepare, retrieve, and evaluate vocabulary questions.

The public question API separates plugin identity (``type_id``) from the UI
contract (``render_format``). Level 0 is exposure; levels 1-4 are assessments.
Preparation is best-effort and persisted, retrieval is exact and never generates,
and evaluation returns either ``graded`` or ``pending``.

    uv run python examples/12_question_engine.py [word]
"""

import asyncio
import sys

from _common import aclose, build_lexicon, lookup

from lexi_ai import QuestionDemand

DEFAULT_WORD = "eloquent"


def _answer_for(question, word: str) -> object:
    if question.render_format == "single_choice":
        return question.payload["correct_index"]
    if question.render_format == "text_span":
        return question.payload["answer_norm"]
    return f"The {word} speaker held the whole room in rapt attention."


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
        reader = lex.reader()
        descriptors = worker.question_types()
        print("=== registered question types ===")
        for descriptor in descriptors:
            levels = ",".join(str(level) for level in sorted(descriptor.supported_levels))
            print(
                f"  {descriptor.type_id}: render={descriptor.render_format} "
                f"mode={descriptor.interaction_mode} levels={levels}"
            )

        sense_id = entry.senses[0].sense_id
        if sense_id is None:
            print("◇ entry has no persisted sense id")
            return

        exposure = await reader.retrieve_exposure(sense_id)
        print(f"\n=== level 0 exposure ({exposure.type_id}) ===")
        print(f"  {exposure.payload['word']}: {exposure.payload['definition']}")

        demands = [QuestionDemand(sense_id, level, 1) for level in range(1, 5)]
        report = await worker.prepare_questions(entry.word_id, demands)
        print("\n=== prepared assessment counts ===")
        for key, count in sorted(report.produced.items()):
            print(f"  sense={key[0]} level={key[1]}: {count}")

        print("\n=== exact retrieval + evaluation ===")
        for descriptor in descriptors:
            if descriptor.interaction_mode != "assessment":
                continue
            level = min(descriptor.supported_levels)
            question = await reader.retrieve_question(
                sense_id,
                level,
                frozenset(),
                descriptor.type_id,
            )
            if question is None:
                print(f"  {descriptor.type_id}@{level}: unavailable (best-effort)")
                continue
            evaluation = await reader.evaluate_answer(
                question.question_id,
                _answer_for(question, entry.display),
            )
            print(
                f"  {question.type_id}@{level}: render={question.render_format} "
                f"status={evaluation.status} verdict={evaluation.verdict} "
                f"score={evaluation.score}"
            )
    finally:
        await aclose(lex)


if __name__ == "__main__":
    asyncio.run(main())
