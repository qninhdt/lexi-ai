"""Translate exact text, reuse saved translations, and inspect a bounded cache page."""

import argparse
import asyncio
from dataclasses import asdict
from pprint import pprint

from _config import add_config_arguments, create_lexicon


async def main(args):
    lexicon = create_lexicon(args)
    try:
        print("Plain input:", repr(args.text))
        print(await lexicon.translate_text(args.text, args.language))
        print("Repeat the same text/language (saved cache entry):")
        print(await lexicon.translate_text(args.text, args.language))

        if args.tagged is not None:
            print("Tagged input:", repr(args.tagged))
            print(await lexicon.translate_text(args.tagged, args.language))
            print("Shares the plain cache key ONLY if exact unwrapped text is identical.")
        if args.whitespace_variant:
            text = " " + args.text
            print("Whitespace variant (different cache key):", repr(text))
            print(await lexicon.translate_text(text, args.language))
        if args.other_language is not None:
            print("Other language (different cache key):", args.other_language)
            print(await lexicon.translate_text(args.text, args.other_language))

        print("Saved translations page (includes previous runs):")
        page = await lexicon.list_translations(limit=args.limit, after_id=args.after_id)
        for translation in page:
            pprint(asdict(translation))
        if page:
            print("Next page cursor (--after-id):", page[-1].id)
        if args.translation_id is not None:
            saved = await lexicon.get_translation(args.translation_id)
            print("Read by ID, without translation generation:")
            pprint(asdict(saved) if saved is not None else None)
        print("Equal translated wording does not imply equal cache keys.")
    finally:
        await lexicon.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(parser)
    parser.add_argument("--text", required=True, help="Exact plain input; whitespace is preserved")
    parser.add_argument("--language", default="vi")
    parser.add_argument("--tagged", help='Same text with valid <t inf="base">target</t> markup')
    parser.add_argument("--whitespace-variant", action="store_true")
    parser.add_argument("--other-language", help="Opt in to a second target language")
    parser.add_argument("--translation-id", type=int)
    parser.add_argument("--limit", type=int, default=10, choices=range(1, 501))
    parser.add_argument("--after-id", type=int)
    asyncio.run(main(parser.parse_args()))
