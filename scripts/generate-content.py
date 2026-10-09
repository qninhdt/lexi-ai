"""Generate Words and optional Question banks concurrently; resume from SQLite."""

import argparse
import asyncio
import sys
from pathlib import Path

from tqdm import tqdm

from lexi_ai import FIXED_STEM_QUESTION_TYPES, QuestionType, cli, migrations
from lexi_ai.cli.config import DEFAULT_ENV_FILE
from lexi_ai.errors import InvalidResourceError, LexiconError


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--word-list", type=Path, required=True)
    result.add_argument("--output", type=Path, default=Path("data/content.sqlite"))
    result.add_argument("--reference-path", type=Path, default=Path("data/reference.sqlite"))
    result.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    result.add_argument(
        "--words-only", action="store_true", help="Generate Words without Questions"
    )
    result.add_argument("--limit", type=int, help="Process only the first N unique input words")
    result.add_argument("--word-concurrency", type=int, default=32)
    result.add_argument("--count", type=int, default=8, help="Questions per Sense/context type")
    result.add_argument("--max-concurrency", type=int, default=128)
    result.set_defaults(word=None, db_schema=None, threshold=0.8, decision_fallback_model=None)
    return result


async def run(args):
    words = cli.input_words(args)
    if min(args.max_concurrency, args.word_concurrency) < 1 or not 1 <= args.count <= 20:
        raise ValueError("concurrency must be positive and count must be 1–20")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        words = words[: args.limit]
    if args.output.resolve() == args.reference_path.resolve():
        raise ValueError("output must differ from the reference database")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.db_url = cli.target_url(str(args.output))
    await asyncio.to_thread(migrations.upgrade_to_head, args.db_url)
    lexicon = cli.create_lexicon(args)
    quotas = {kind: 1 if kind in FIXED_STEM_QUESTION_TYPES else args.count for kind in QuestionType}
    failures = 0
    Path(str(args.output) + ".sha256").unlink(missing_ok=True)
    try:
        await lexicon.import_reference(args.reference_path)
        await lexicon.start()
        targets = iter(words)
        banks = {}
        with tqdm(
            total=len(words), desc="Words" if args.words_only else "Word + Questions", unit="word"
        ) as progress:

            async def worker():
                nonlocal failures
                for target in targets:
                    try:
                        word = await lexicon.generate_word(target)
                        if not args.words_only:
                            # Alias/form inputs can resolve to the same Word.
                            async with banks.setdefault(word.id, asyncio.Lock()):
                                ids = [sense.id for sense in word.senses]
                                counts = await lexicon.count_questions_for_senses(ids)
                                async with asyncio.TaskGroup() as group:
                                    for sense_id in ids:
                                        for kind in QuestionType:
                                            missing = quotas[kind] - counts.get((sense_id, kind), 0)
                                            if missing > 0:
                                                group.create_task(
                                                    lexicon.generate_questions(
                                                        sense_id,
                                                        kind,
                                                        missing,
                                                        distractor_count=(
                                                            5
                                                            if kind in FIXED_STEM_QUESTION_TYPES
                                                            else 3
                                                        ),
                                                    )
                                                )
                                counts = await lexicon.count_questions_for_senses(ids)
                                if any(
                                    counts.get((sid, kind), 0) < quotas[kind]
                                    for sid in ids
                                    for kind in QuestionType
                                ):
                                    raise InvalidResourceError("question bank remains incomplete")
                        tqdm.write(
                            f"{target}: {word.lemma} (id={word.id}, {len(word.senses)} senses)"
                        )
                    except Exception as error:
                        failures += 1
                        message = (
                            str(error)
                            if isinstance(error, LexiconError)
                            else (f"{type(error).__name__}: generation failed")
                        )
                        tqdm.write(f"{target}: {message}", file=sys.stderr)
                    finally:
                        progress.update(1)
                        progress.set_postfix(failures=failures)

            async with asyncio.TaskGroup() as group:
                for _ in range(min(args.word_concurrency, len(words))):
                    group.create_task(worker())
    finally:
        await lexicon.close()
        cli.finalize(args.db_url)
    return int(failures > 0)


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        print("Interrupted; run the same command to resume committed content.", file=sys.stderr)
        return 130
    except Exception as error:
        # Provider/DB exceptions can contain credentials; keep transport details private.
        print(
            f"{type(error).__name__}: failed; check input and provider settings.", file=sys.stderr
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
