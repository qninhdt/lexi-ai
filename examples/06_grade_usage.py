"""Gate on target use, then diagnose usage against the saved meaning anchor."""

import argparse
import asyncio
from dataclasses import asdict
from pprint import pprint

from _config import add_config_arguments, create_lexicon


async def main(args):
    lexicon = create_lexicon(args)
    try:
        question = await lexicon.get_question(args.question_id)
        if question is None or question.question_type != "word_to_usage":
            raise SystemExit("Choose a saved word_to_usage Question from example 03.")
        print("Saved Word/meaning anchor:")
        pprint(question.content)

        for answer in args.answer:
            print("Submitted sentence:", repr(answer))
            grade = await lexicon.grade_answer(question.id, "short_answer", answer)
            pprint(asdict(grade))
            if not grade.used:
                print("Target not used; all five usage diagnostics remain null.")
            else:
                print("Meaning, form, construction, collocation and appropriacy are separate.")
        print("The saved meaning anchor is used, not a later dictionary rewrite.")
        print("These are model diagnostics, not predetermined expected verdicts.")
    finally:
        await lexicon.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    parser.add_argument("question_id", type=int)
    parser.add_argument(
        "--answer", action="append", required=True, help="Repeat for natural/error/non-use samples"
    )
    asyncio.run(main(parser.parse_args()))
