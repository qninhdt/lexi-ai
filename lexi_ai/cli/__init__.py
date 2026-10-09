"""Installed one-shot commands over the native dictionary API."""

import argparse
import asyncio
import sqlite3
import sys
from contextlib import closing
from dataclasses import asdict, is_dataclass
from importlib.metadata import version
from pathlib import Path

import orjson
from sqlalchemy.engine import make_url

from lexi_ai import (
    FIXED_STEM_QUESTION_TYPES,
    DecisionMode,
    Lexicon,
    QuestionType,
    ResponseFormat,
    TargetPlacement,
    migrations,
)
from lexi_ai.config import database_schema_name
from lexi_ai.datasets import checksum
from lexi_ai.errors import LexiconError
from lexi_ai.text import validate_lemma

from .config import DEFAULT_ENV_FILE, create_lexicon


def target_url(target, schema=None):
    """One destination grammar for commands: SQLite path or PostgreSQL URL."""
    database_schema_name(schema)
    if not target:
        raise ValueError("destination must not be empty")
    if "://" not in target:
        if schema:
            raise ValueError("named schema requires PostgreSQL")
        return "sqlite+aiosqlite:///" + str(Path(target).expanduser().resolve())
    url = make_url(target)
    if url.drivername in {"postgresql", "postgresql+asyncpg"}:
        return url.set(drivername="postgresql+asyncpg").render_as_string(hide_password=False)
    if url.drivername == "sqlite+aiosqlite" and not schema:
        if not url.database or url.database == ":memory:":
            raise ValueError("SQLite output must be a persistent file")
        return url.set(database=str(Path(url.database).expanduser().resolve())).render_as_string()
    raise ValueError("expected a SQLite path or PostgreSQL URL")


def allocation(value):
    try:
        sense, kind, count = value.split(":")
        if int(sense) < 1 or int(count) < 1:
            raise ValueError
        return int(sense), QuestionType(kind), int(count)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected SENSE_ID:TYPE:COUNT") from error


def pair(value):
    try:
        kind, fmt = value.split(":")
        return QuestionType(kind), ResponseFormat(fmt)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected TYPE:FORMAT") from error


def parser():
    root = argparse.ArgumentParser(
        prog="lexi", description="Generate and reuse vocabulary content."
    )
    root.add_argument("--version", action="version", version=version("lexi-ai"))
    groups = root.add_subparsers(dest="group", required=True)

    def commands(name, actions):
        sub = groups.add_parser(name).add_subparsers(dest="action", required=True)
        return {action: sub.add_parser(action) for action in actions}

    def common(p, *, destination="db", required=True):
        p.add_argument("--" + destination, required=required, help="SQLite path or PostgreSQL URL")
        p.add_argument("--db-schema")
        p.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
        p.add_argument("--threshold", type=float, default=0.8)
        p.add_argument("--decision-fallback-model")
        p.add_argument("--max-concurrency", type=int, default=128)

    def theme(p):
        p.add_argument("--theme")

    def kinds(p):
        p.add_argument("--type", dest="types", choices=list(QuestionType), action="append")

    def paging(p, key="after-id"):
        p.add_argument("--" + key, type=str if key == "after-key" else int)
        p.add_argument("--limit", type=int)

    def mode(p):
        p.add_argument("--mode", choices=list(DecisionMode), default=DecisionMode.LLM_FALLBACK)
        p.add_argument("--with-usage", action="store_true")

    db = commands("db", ("init", "head", "current"))
    for action in ("init", "current"):
        common(db[action])
    db["init"].add_argument("--reference-path", type=Path)
    imp = groups.add_parser("import", help="Import SQLite into PostgreSQL; conflicts roll back.")
    imp.add_argument("source", nargs="?", type=Path)
    common(imp, destination="into")
    imp.add_argument("--kind", choices=("content", "reference"), default="content")
    imp.add_argument("--words-only", action="store_true")
    imp.add_argument("--reference-path", type=Path)
    ref = commands("reference", ("validate",))["validate"]
    common(ref)
    ref.add_argument("reference_id")
    ref.add_argument("--reference-path", type=Path)

    gen = commands("generate", ("word", "question", "theme", "translation"))
    for p in gen.values():
        common(p, destination="output")
        p.add_argument("--with-usage", action="store_true")
    gen["word"].add_argument("--reference-path", type=Path)
    gen["word"].add_argument("word", nargs="?")
    gen["word"].add_argument("--word-list", type=Path)
    theme(gen["word"])
    theme(gen["question"])
    gen["word"].add_argument("--example-count", type=int, default=5)
    gen["question"].add_argument("sense_id", type=int)
    gen["question"].add_argument("--type", choices=list(QuestionType), required=True)
    gen["question"].add_argument(
        "--count", type=int, help="New questions (default 1 for fixed stems, otherwise 8)"
    )
    gen["question"].add_argument(
        "--distractor-count", type=int, help="Distractors (default 5 for fixed stems, otherwise 3)"
    )
    gen["question"].add_argument("--target-placement", choices=list(TargetPlacement))
    gen["theme"].add_argument("key")
    gen["theme"].add_argument("--name", required=True)
    gen["theme"].add_argument("--concept", required=True)
    gen["translation"].add_argument("--text", required=True, help="Text or - for stdin")
    gen["translation"].add_argument("--language", required=True)

    word = commands("word", ("search", "get"))
    for p in word.values():
        common(p)
    word["search"].add_argument("query")
    word["search"].add_argument("--include-reference", action="store_true")
    word["search"].add_argument("--reference-path", type=Path)
    word["get"].add_argument("id", type=int)
    theme(word["get"])
    senses = commands("sense", ("get", "previews"))
    for p in senses.values():
        common(p)
        p.add_argument("ids", nargs="+", type=int)
    themes = commands("theme", ("get", "list", "update", "delete"))
    for action, p in themes.items():
        common(p)
        if action != "list":
            p.add_argument("key")
    paging(themes["list"], "after-key")
    for field in ("name", "voice", "diction"):
        themes["update"].add_argument("--" + field)

    questions = commands("question", ("get", "list", "count", "retrieve", "delete", "grade"))
    for p in questions.values():
        common(p)
    for action in ("get", "list", "count"):
        questions[action].add_argument("ids", nargs="+", type=int)
    for action in ("list", "count"):
        theme(questions[action])
        kinds(questions[action])
    paging(questions["list"])
    questions["list"].add_argument("--limit-per-type", type=int, default=8)
    retrieve = questions["retrieve"]
    retrieve.add_argument("sense_id", nargs="?", type=int)
    retrieve.add_argument("--request", type=allocation, action="append")
    retrieve.add_argument("--type", choices=list(QuestionType))
    theme(retrieve)
    for action in ("delete", "grade"):
        questions[action].add_argument("id", type=int)
    grade = questions["grade"]
    grade.add_argument("--format", choices=list(ResponseFormat), required=True)
    grade.add_argument("--answer", required=True, help="Answer or - for stdin")
    grade.add_argument("--allowed-pair", type=pair, action="append")
    mode(grade)
    relation = commands("relation", ("resolve",))["resolve"]
    common(relation)
    mode(relation)
    relation.add_argument(
        "--batch-size", type=int, default=20, help="Global eligible relation page"
    )
    translations = commands("translation", ("get", "list", "delete", "purge"))
    for action, p in translations.items():
        common(p)
        if action in ("get", "delete"):
            p.add_argument("id", type=int)
    paging(translations["list"])
    return root


def input_words(args):
    if bool(args.word) == bool(args.word_list):
        raise ValueError("provide exactly one word or --word-list")
    lines = (
        args.word_list.read_text(encoding="utf-8-sig").splitlines()
        if args.word_list
        else [args.word]
    )
    words = list(dict.fromkeys(validate_lemma(line.strip()) for line in lines if line.strip()))
    if not words:
        raise ValueError("word input must not be empty")
    return words


def finalize(url):
    path = Path(make_url(url).database)
    with closing(sqlite3.connect(path)) as connection:
        row = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if row[0]:
            raise RuntimeError("SQLite checkpoint busy")
        if connection.execute("PRAGMA journal_mode=DELETE").fetchone() != ("delete",):
            raise RuntimeError("SQLite journal finalization failed")
        if connection.execute("PRAGMA quick_check").fetchone() != ("ok",):
            raise RuntimeError("SQLite integrity check failed")
    Path(str(path) + ".sha256").write_text(checksum(path) + "\n")


async def dispatch(lexicon, a):
    if a.group == "generate":
        if a.action == "word":
            words, failures = [], []
            targets = iter(a.words)

            async def worker():
                for target in targets:
                    try:
                        words.append(
                            await lexicon.generate_word(
                                target,
                                theme=a.theme,
                                example_count=a.example_count,
                                with_usage=a.with_usage,
                            )
                        )
                    except Exception as error:
                        failures.append(
                            {
                                "target": target,
                                "error": str(error)
                                if isinstance(error, LexiconError)
                                else type(error).__name__,
                            }
                        )

            async with asyncio.TaskGroup() as tasks:
                for _ in range(min(a.max_concurrency, len(a.words))):
                    tasks.create_task(worker())
            return {"words": words, "failures": failures}
        if a.action == "question":
            fixed = a.type in FIXED_STEM_QUESTION_TYPES
            return await lexicon.generate_questions(
                a.sense_id,
                a.type,
                a.count if a.count is not None else (1 if fixed else 8),
                distractor_count=(
                    a.distractor_count if a.distractor_count is not None else (5 if fixed else 3)
                ),
                theme=a.theme,
                target_placement=a.target_placement,
                with_usage=a.with_usage,
            )
        if a.action == "theme":
            return await lexicon.create_theme(a.key, a.name, a.concept, with_usage=a.with_usage)
        return await lexicon.translate_text(a.text, a.language, with_usage=a.with_usage)
    if a.group == "import":
        if a.kind == "reference":
            return await lexicon.import_reference(a.source)
        await lexicon.import_reference(a.reference_path)
        return await lexicon.import_content(a.source, questions=not a.words_only)
    if a.group == "db":
        return await lexicon.import_reference(a.reference_path)
    if a.group == "reference":
        await lexicon.validate_reference(a.reference_id)
        return {"valid": True}
    if a.group == "word":
        if a.action == "search":
            return await lexicon.search(a.query, include_reference=a.include_reference)
        return await lexicon.get_word(a.id, theme=a.theme)
    if a.group == "sense":
        if a.action == "get":
            return await lexicon.get_senses(a.ids)
        return await lexicon.get_sense_previews(a.ids)
    if a.group == "theme":
        if a.action == "get":
            return await lexicon.get_theme(a.key)
        if a.action == "list":
            return await lexicon.list_themes(after_key=a.after_key, limit=a.limit)
        if a.action == "update":
            return await lexicon.update_theme(a.key, name=a.name, voice=a.voice, diction=a.diction)
        return await lexicon.delete_theme(a.key)
    if a.group == "question":
        if a.action == "get":
            return (
                await lexicon.get_question(a.ids[0])
                if len(a.ids) == 1
                else await lexicon.get_questions(a.ids)
            )
        if a.action == "list":
            if len(a.ids) == 1:
                return await lexicon.list_questions(
                    a.ids[0],
                    QuestionType(a.types[0]) if a.types else None,
                    theme=a.theme,
                    after_id=a.after_id,
                    limit=a.limit,
                )
            return await lexicon.list_questions_for_senses(
                a.ids, a.types, theme=a.theme, limit_per_type=a.limit_per_type
            )
        if a.action == "count":
            counts = await lexicon.count_questions_for_senses(a.ids, a.types, theme=a.theme)
            return [
                {"sense_id": sense, "question_type": kind, "count": count}
                for (sense, kind), count in sorted(counts.items())
            ]
        if a.action == "retrieve":
            if a.request:
                return await lexicon.retrieve_questions(a.request, theme=a.theme)
            return await lexicon.retrieve_question(a.sense_id, a.type, theme=a.theme)
        if a.action == "delete":
            return await lexicon.delete_question(a.id)
        return await lexicon.grade_answer(
            a.id,
            a.format,
            a.answer,
            mode=a.mode,
            with_usage=a.with_usage,
            allowed_pairs=set(a.allowed_pair) if a.allowed_pair else None,
        )
    if a.group == "relation":
        return await lexicon.resolve_relations(a.batch_size, mode=a.mode, with_usage=a.with_usage)
    if a.action == "get":
        return await lexicon.get_translation(a.id)
    if a.action == "list":
        return await lexicon.list_translations(after_id=a.after_id, limit=a.limit)
    if a.action == "delete":
        return await lexicon.delete_translation(a.id)
    return await lexicon.purge_translations()


async def run(args):
    if args.group == "db" and args.action == "head":
        return migrations.inspect_head()
    args.db_url = target_url(
        getattr(args, "output", None) or getattr(args, "into", None) or getattr(args, "db", None),
        args.db_schema,
    )
    if args.group == "import" and not args.db_url.startswith("postgresql"):
        raise ValueError("import destination must be PostgreSQL")
    if args.group == "db" and args.action == "current":
        return await asyncio.to_thread(
            migrations.inspect_current, args.db_url, db_schema=args.db_schema
        )
    setup = args.group in {"generate", "import"} or (args.group == "db" and args.action == "init")
    if setup:
        if args.db_url.startswith("sqlite"):
            output = Path(make_url(args.db_url).database)
            output.parent.mkdir(parents=True, exist_ok=True)
            if args.group == "generate":
                Path(str(output) + ".sha256").unlink(missing_ok=True)
        await asyncio.to_thread(migrations.upgrade_to_head, args.db_url, db_schema=args.db_schema)
    providers = (
        args.group == "generate"
        or args.group == "relation"
        or (args.group == "question" and args.action == "grade")
    )
    lexicon = create_lexicon(args) if providers else Lexicon(args.db_url, db_schema=args.db_schema)
    try:
        if args.group == "generate" and args.action == "word":
            await lexicon.import_reference(args.reference_path)
        elif args.db_url.startswith("sqlite") and (
            args.group == "reference"
            or (
                args.group == "word"
                and args.action == "search"
                and (args.include_reference or args.reference_path)
            )
        ):
            await lexicon.import_reference(args.reference_path)
        if (
            (args.group == "word" and args.action == "search")
            or (args.group == "question" and args.action == "grade")
            or (args.group == "generate" and args.action == "word")
        ):
            await lexicon.start()
        result = await dispatch(lexicon, args)
    finally:
        await lexicon.close()
    if args.group == "generate" and args.db_url.startswith("sqlite"):
        finalize(args.db_url)
    return result


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        if hasattr(args, "max_concurrency") and args.max_concurrency < 1:
            raise ValueError("max concurrency must be positive")
        if args.group == "generate" and args.action == "word":
            args.words = input_words(args)
            if args.example_count < 1:
                raise ValueError("example count must be positive")
        if args.group == "generate" and args.action == "question":
            if (
                args.sense_id < 1
                or (args.count is not None and not 1 <= args.count <= 20)
                or (args.distractor_count is not None and not 3 <= args.distractor_count <= 20)
            ):
                raise ValueError(
                    "Sense ID must be positive, count 1..20 and distractor count 3..20"
                )
        if args.group == "question" and args.action == "retrieve":
            if bool(args.sense_id) == bool(args.request):
                raise ValueError("provide a Sense ID or --request allocations")
            if args.request and args.type:
                raise ValueError("--type applies only to a single Sense")
        if args.group == "question" and args.action == "list":
            if len(args.ids) == 1 and args.types and len(args.types) > 1:
                raise ValueError("single-Sense list accepts one --type")
            if len(args.ids) > 1 and (args.after_id is not None or args.limit is not None):
                raise ValueError("batch list uses --limit-per-type, not cursor/limit")
        if (
            args.group == "theme"
            and args.action == "update"
            and not any(getattr(args, field) is not None for field in ("name", "voice", "diction"))
        ):
            raise ValueError("supply a theme field to update")
        if (
            args.group == "import"
            and args.kind == "reference"
            and (args.words_only or args.reference_path)
        ):
            raise ValueError("reference import does not accept content options")
        for name in ("text", "answer"):
            if getattr(args, name, None) == "-":
                setattr(args, name, sys.stdin.read())
        if hasattr(args, "db_schema"):
            target_url(
                getattr(args, "output", None)
                or getattr(args, "into", None)
                or getattr(args, "db", None),
                args.db_schema,
            )
    except (ValueError, OSError) as error:
        p.error(str(error))
    try:
        result = asyncio.run(run(args))
        failures = result.get("failures", []) if isinstance(result, dict) else []
        for failure in failures:
            print(f"{failure['target']}: {failure['error']}", file=sys.stderr)
        print(
            orjson.dumps(
                result, default=lambda value: asdict(value) if is_dataclass(value) else str(value)
            ).decode()
        )
        return 1 if failures else 0
    except KeyboardInterrupt:
        print("Interrupted; committed progress is preserved.", file=sys.stderr)
        return 130
    except Exception as error:
        # DB/provider exceptions can include connection strings or credentials.
        print(
            f"{type(error).__name__}: operation failed; check input, database and provider config.",
            file=sys.stderr,
        )
        return 1
