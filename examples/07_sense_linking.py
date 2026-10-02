"""Inspect pending targets, optionally generate selected entries, then resolve a page."""

import argparse
import asyncio
from dataclasses import asdict
from pprint import pprint

from _config import add_config_arguments, create_lexicon


async def main(args):
    lexicon = create_lexicon(args)
    try:
        word = await lexicon.get_word(args.word_id)
        if word is None:
            raise SystemExit("Choose a generated source Word ID from example 01.")
        print("Source relations before Sense Linking:")
        pending_targets = {}
        for sense in word.senses:
            for relation in sense.relations:
                pprint({"source_sense_id": sense.id, **asdict(relation)})
                if relation.resolution_state == "pending":
                    pending_targets[relation.to_word_id] = relation.to_word_lemma

        for target_id, lemma in pending_targets.items():
            print("Pending target:", target_id, lemma)
            target = await lexicon.get_word(target_id)
            if target is None or target.generation_state != "done":
                print("Select a Cambridge available_id explicitly if this target needs generation:")
                pprint(asdict(await lexicon.search(lemma, include_available=True)))

        for available_id, target_text in args.target_entry:
            target = await lexicon.generate(
                available_id, target=target_text, example_count=args.example_count
            )
            print("Generated/reused selected target:", target.id, target.lemma)

        if not args.resolve:
            print("Inspection complete. Use --resolve to process one eligible global page.")
            return

        # Public API has no source-Word filter. This can process OTHER Words' edges too.
        print("Resolving a GLOBAL page, not only this source Word's relations.")
        results = await lexicon.resolve_relations(batch_size=args.batch_size)
        for result in results:
            pprint(asdict(result))
        if not results:
            print("No eligible pending edges in this page; unavailable targets stay pending.")
        print("resolved/unresolvable are decisions; error stays pending; noop means not applied.")
        print("Source relations after the call:")
        refreshed = await lexicon.get_word(word.id)
        if refreshed is None:
            print("Source Word is no longer available.")
            return
        for sense in refreshed.senses:
            for relation in sense.relations:
                pprint({"source_sense_id": sense.id, **asdict(relation)})
    finally:
        await lexicon.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    parser.add_argument("word_id", type=int, help="Source Word to inspect, NOT a resolution filter")
    parser.add_argument(
        "--target-entry",
        action="append",
        nargs=2,
        metavar=("AVAILABLE_ID", "TARGET"),
        default=[],
        help="Selected Cambridge ID and lexical item; repeat for multiple entries",
    )
    parser.add_argument(
        "--resolve", action="store_true", help="Opt in to global relation decisions"
    )
    parser.add_argument("--batch-size", type=int, default=20, choices=range(1, 51))
    parser.add_argument(
        "--example-count", type=int, default=3, help="Examples per new target Sense"
    )
    asyncio.run(main(parser.parse_args()))
