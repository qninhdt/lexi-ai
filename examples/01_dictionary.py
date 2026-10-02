"""Search first; generate only an explicitly selected Cambridge entry."""

import argparse
import asyncio
from dataclasses import asdict
from pprint import pprint

from _config import add_config_arguments, create_lexicon


async def main(args):
    lexicon = create_lexicon(args)
    try:
        matches = await lexicon.search(args.query, include_available=True)
        print("Stored dictionary hits (word_id) and Cambridge entries (available_id):")
        pprint(asdict(matches))

        if args.available_id is not None:
            # The caller picks the handle; never implicitly generate the first hit.
            if not args.target:
                raise SystemExit("Generation requires --target for the selected lexical item.")
            word = await lexicon.generate(
                args.available_id, target=args.target, example_count=args.example_count
            )
        elif args.word_id is not None:
            word = await lexicon.get_word(args.word_id)
        else:
            print("Select --available-id to generate, or --word-id to read a stored Word.")
            return

        if word is None:
            raise SystemExit("No stored Word for that ID.")
        print("Word, definitions, examples, forms, patterns and relations:")
        pprint(asdict(word))
        print("Read the same neutral Word again without generation:")
        pprint(asdict(await lexicon.get_word(word.id)))
        print("Batch-read its neutral Senses:")
        for sense in await lexicon.get_senses([sense.id for sense in word.senses]):
            pprint(asdict(sense))
    finally:
        await lexicon.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    parser.add_argument("query")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--available-id", help="Exact handle selected from search output")
    selection.add_argument("--word-id", type=int, help="Read an existing generated Word")
    parser.add_argument("--target", help="Lexical item to generate for the selected entry")
    parser.add_argument("--example-count", type=int, default=3, help="Examples per new Sense")
    asyncio.run(main(parser.parse_args()))
