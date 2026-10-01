"""Compare saved-answer fast paths with independent free-text diagnostics."""

import argparse
import asyncio
from dataclasses import asdict
from pprint import pprint

from _config import add_config_arguments, create_lexicon


async def main(args):
    lexicon = create_lexicon(args)
    try:
        question = await lexicon.get_question(args.question_id)
        if question is None or not question.supports("single_word"):
            raise SystemExit("Choose a saved definition/context/cloze-to-word Question.")
        print("Prompt:")
        pprint(question.content)

        # This is a trusted developer demo, not a learner-facing endpoint.
        print("Saved correct option ID, no provider request:")
        pprint(
            asdict(await lexicon.grade_answer(question.id, "single_choice", question.correct.id))
        )
        if question.distractors:
            print("Saved distractor ID, no provider request:")
            pprint(
                asdict(
                    await lexicon.grade_answer(
                        question.id, "single_choice", question.distractors[0].id
                    )
                )
            )
        print("Exact saved single-word answer, no provider request:")
        pprint(
            asdict(await lexicon.grade_answer(question.id, "single_word", question.correct.content))
        )

        for answer in args.answer:
            print("Submitted answer:", repr(answer))
            pprint(asdict(await lexicon.grade_answer(question.id, "single_word", answer)))
        print("task_fit and spelling_error are independent; sense_id is dictionary mapping.")
        print("A missing dictionary mapping does not turn task_fit into false.")
    finally:
        await lexicon.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    parser.add_argument("question_id", type=int)
    parser.add_argument(
        "--answer", action="append", default=[], help="Repeat for your correct/typo/wrong samples"
    )
    asyncio.run(main(parser.parse_args()))
