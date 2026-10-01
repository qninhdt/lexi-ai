"""Generate and compare exact neutral/themed namespaces for a selected entry."""

import argparse
import asyncio
from dataclasses import asdict
from pprint import pprint

from _config import add_config_arguments, create_lexicon


async def main(args):
    lexicon = create_lexicon(args)
    try:
        theme = await lexicon.get_theme(args.theme)
        if theme is None:
            if not args.name or not args.concept:
                raise SystemExit("A new Theme requires both --name and --concept.")
            theme = await lexicon.create_theme(args.theme, args.name, args.concept)
        print("Theme metadata:")
        pprint(asdict(theme))

        themed = await lexicon.generate(
            args.available_id, theme=theme.key, example_count=args.example_count
        )
        print("Neutral namespace:")
        pprint(asdict(await lexicon.get_word(themed.id)))
        print("Themed namespace:")
        pprint(asdict(themed))
        print("Read stored themed content without regenerating it:")
        pprint(asdict(await lexicon.get_word(themed.id, theme=theme.key)))

        if args.update_name is not None:
            print("Explicit metadata update; saved Word content is not rewritten:")
            pprint(asdict(await lexicon.update_theme(theme.key, name=args.update_name)))
            pprint(asdict(await lexicon.get_word(themed.id, theme=theme.key)))

        print("First page of Themes, ordered by key:")
        for item in await lexicon.list_themes(limit=10):
            pprint(asdict(item))
    finally:
        await lexicon.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    parser.add_argument("available_id", help="Selected Cambridge handle from example 01")
    parser.add_argument("--theme", required=True, help="Exact Theme key")
    parser.add_argument("--name", help="Display name when creating a new Theme")
    parser.add_argument("--concept", help="Style concept when creating a new Theme")
    parser.add_argument("--update-name", help="Opt in to changing existing Theme metadata")
    parser.add_argument("--example-count", type=int, default=3, help="Examples per new Sense")
    asyncio.run(main(parser.parse_args()))
