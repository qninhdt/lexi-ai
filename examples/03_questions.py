"""Generate on request, then read a saved Question bank without exposing answers."""

import argparse
import asyncio
import random
from pprint import pprint

from _config import add_config_arguments, create_lexicon

QUESTION_TYPES = (
    "definition_to_word",
    "context_to_word",
    "cloze_to_word",
    "word_to_definition",
    "word_to_usage",
    "dialogue_completion",
    "meaning_in_context",
)


async def main(args):
    lexicon = create_lexicon(args)
    try:
        if args.generate_count is not None:
            generated = await lexicon.generate_questions(
                args.sense_id,
                args.question_type,
                args.generate_count,
                distractor_count=args.distractor_count,
                theme=args.theme,
                target_placement=args.target_placement,
            )
            print("Appended Question IDs:", [question.id for question in generated])

        page = await lexicon.list_questions(
            args.sense_id,
            args.question_type,
            theme=args.theme,
            limit=args.limit,
            after_id=args.after_id,
        )
        print("Stored page IDs:", [question.id for question in page])
        if page:
            saved = await lexicon.get_question(page[0].id)
            print("Read by ID:", saved.id if saved is not None else None)
            print("Next page cursor (--after-id):", page[-1].id)

        question = await lexicon.retrieve_question(
            args.sense_id, args.question_type, theme=args.theme
        )
        if question is None:
            print("No saved Questions in this exact namespace; reads do not generate.")
            return

        # Full artifacts are server-only. Shuffle before exposing just IDs/content.
        options = [question.correct, *question.distractors]
        random.shuffle(options)
        print("Learner-facing multiple-choice payload (no answers or explanations):")
        pprint(
            {
                "id": question.id,
                "content": question.content,
                "options": [{"id": option.id, "content": option.content} for option in options],
            }
        )
        print(
            "Supported formats:",
            [
                fmt
                for fmt in ("single_choice", "single_word", "short_answer")
                if question.supports(fmt)
            ],
        )
        print("Use this Question ID with grading examples 04–06.")
    finally:
        await lexicon.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    parser.add_argument("sense_id", type=int)
    parser.add_argument("question_type", choices=QUESTION_TYPES)
    parser.add_argument("--theme", help="Omit for the neutral namespace")
    parser.add_argument("--generate-count", type=int, choices=range(1, 21))
    parser.add_argument("--distractor-count", type=int, default=3, choices=range(3, 21))
    parser.add_argument("--target-placement", choices=("dialogue", "options"))
    parser.add_argument("--limit", type=int, default=10, choices=range(1, 501))
    parser.add_argument("--after-id", type=int)
    args = parser.parse_args()
    if args.target_placement is not None and args.question_type != "dialogue_completion":
        parser.error("--target-placement applies only to dialogue_completion")
    if args.target_placement is not None and args.generate_count is None:
        parser.error("--target-placement requires --generate-count")
    asyncio.run(main(args))
