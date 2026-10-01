"""Identify intended owner-Word meaning before evaluating accuracy and coverage."""

import argparse
import asyncio
from dataclasses import asdict
from pprint import pprint

from _config import add_config_arguments, create_lexicon


async def main(args):
    lexicon = create_lexicon(args)
    try:
        question = await lexicon.get_question(args.question_id)
        if question is None or question.question_type != "word_to_definition":
            raise SystemExit("Choose a saved word_to_definition Question from example 03.")
        print("Prompt:")
        pprint(question.content)

        for answer in args.answer:
            print("Submitted definition:", repr(answer))
            grade = await lexicon.grade_answer(question.id, "short_answer", answer)
            pprint(asdict(grade))
            if grade.sense_id is None:
                print("No identifiable intended Sense; accuracy and coverage remain null.")
            else:
                print("Identified neutral Sense (may differ from the Question's original Sense):")
                for sense in await lexicon.get_senses([grade.sense_id]):
                    pprint(
                        {
                            "id": sense.id,
                            "definition": asdict(sense.definition) if sense.definition else None,
                        }
                    )
        print("These are model diagnostics, not predetermined expected verdicts.")
    finally:
        await lexicon.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    parser.add_argument("question_id", type=int)
    parser.add_argument(
        "--answer", action="append", required=True, help="Repeat for several proposed definitions"
    )
    asyncio.run(main(parser.parse_args()))
