"""Disposable 100k-Word PostgreSQL search probe using a read-only Cambridge corpus.

Run: LEXI_TEST_PG_URL=... uv run python tests/benchmark_postgres_search.py \
    --source data --output /tmp/opencode/lexi-search-benchmark.json

This is a retrieval probe, not generated dictionary ingestion or a CI latency SLO.
Only the newly created schema is populated and dropped. No provider is called.
"""

import argparse
import asyncio
import json
import math
import os
import platform
import sqlite3
import statistics
import time
import uuid
from pathlib import Path

import asyncpg
from sqlalchemy import event

from lexi_ai.db.session import Database
from lexi_ai.patterns import surface_head_key
from lexi_ai.text import answer_key, match_key
from lexi_ai.vocab import INFLECTIONS
from lexi_ai.words.search import search


def corpus(path, size):
    words, aliases, forms = [], [], []
    keys, source_ids = set(), {}
    with sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True) as source:
        for source_id, lemma in source.execute(
            "SELECT id,coalesce(display_form,word) FROM words WHERE status='done' ORDER BY id"
        ):
            try:
                key = match_key(lemma)
            except ValueError:
                continue
            if key in keys or len(key) > 512:
                continue
            keys.add(key)
            words.append((source_id, lemma, key, "word", "done"))
        if len(words) < size:
            raise ValueError(f"source has {len(words)} eligible normalized Words, need {size}")
        # Spread the sample over the full alphabet instead of losing late words.
        words = [words[index * len(words) // size] for index in range(size)]
        source_ids = {word[0]: index + 1 for index, word in enumerate(words)}
        words = [(source_ids[w[0]], *w[1:]) for w in words]
        seen = set()
        for source_id, alias in source.execute(
            "SELECT word_id,alternative_word FROM word_alternatives ORDER BY word_id"
        ):
            if source_id not in source_ids:
                continue
            try:
                key = match_key(alias)
            except ValueError:
                continue
            identity = source_ids[source_id], key
            if identity in seen or len(key) > 512:
                continue
            seen.add(identity)
            aliases.append((len(aliases) + 1, identity[0], alias, key))
        for source_id, form, inf in source.execute(
            "SELECT e.word_id,f.inflected_form,f.form_type FROM entry_inflections f "
            "JOIN entries e ON e.id=f.entry_id ORDER BY f.id"
        ):
            if source_id not in source_ids:
                continue
            key = answer_key(form)
            if len(key) > 512:
                continue
            forms.append(
                (
                    len(forms) + 1,
                    source_ids[source_id],
                    form,
                    key,
                    surface_head_key(form),
                    inf if inf in INFLECTIONS else "base",
                )
            )
    if len(words) != size:
        raise ValueError(f"source has {len(words)} eligible normalized Words, need {size}")
    return words, aliases, forms


def nodes(plan):
    yield plan
    for child in plan.get("Plans", []):
        yield from nodes(child)


async def benchmark(args):
    url = os.environ["LEXI_TEST_PG_URL"]
    schema = f"lexi_benchmark_{uuid.uuid4().hex[:12]}"
    # asyncpg accepts the same URL after removing SQLAlchemy's driver suffix.
    admin = await asyncpg.connect(url.replace("postgresql+asyncpg://", "postgresql://", 1))
    db = None
    report = {
        "schema": schema,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "iterations": args.iterations,
        "scope": "single-client warm-cache retrieval, no generation/providers",
        "source": str(Path(args.source).resolve()),
        "cases": [],
    }
    try:
        await admin.execute(f'CREATE SCHEMA "{schema}"')
        await admin.execute(f'SET search_path TO "{schema}"')
        db = Database(url, schema=schema)
        from lexi_ai.schema import Base

        await db.create_schema(Base.metadata)
        words, aliases, forms = corpus(args.source, args.words)
        report["postgres"] = await admin.fetchval("SELECT version()")
        report["counts"] = {
            "words": len(words),
            "aliases": len(aliases),
            "forms": len(forms),
            "senses": len(words),
            "patterns": 0,
        }
        start = time.perf_counter()
        async with admin.transaction():
            await admin.copy_records_to_table(
                "words",
                records=words,
                columns=["id", "lemma", "match_key", "entry_type", "generation_state"],
            )
            await admin.copy_records_to_table(
                "senses",
                records=[(w[0], w[0], "noun", "core") for w in words],
                columns=["id", "word_id", "pos", "tier"],
            )
            await admin.copy_records_to_table(
                "word_aliases",
                records=aliases,
                columns=["id", "word_id", "content", "match_key"],
            )
            await admin.copy_records_to_table(
                "sense_forms",
                records=forms,
                columns=["id", "sense_id", "surface", "match_key", "head_key", "inf"],
            )
        # Finish GIN pending-list maintenance and refresh planner statistics.
        for table in ["words", "senses", "word_aliases", "sense_forms", "sense_patterns"]:
            await admin.execute(f"VACUUM (ANALYZE) {table}")
        report["seed_seconds"] = time.perf_counter() - start
        await admin.execute("SELECT set_config('pg_trgm.similarity_threshold','0.3',false)")
        report["settings"] = {
            setting: await admin.fetchval(f"SHOW {setting}")
            for setting in (
                "shared_buffers",
                "work_mem",
                "gin_fuzzy_search_limit",
                "enable_seqscan",
            )
        }
        cases = [
            ("exact_lemma", "bank"),
            ("prefix_lemma", "trans"),
            ("exact_alias", "walking papers"),
            ("exact_form", "children"),
            ("fuzzy_lemma", "dictionry"),
            ("fuzzy_alias", "walking paperss"),
            ("fuzzy_form", "childrenn"),
            ("short_broad", "a"),
            ("two_character", "ta"),
            ("common_typo", "bankk"),
            ("no_match", "zzqxjjvv"),
        ]
        for label, query in cases:
            statements = []

            def capture(
                _conn, _cursor, statement, parameters, _context, _many, statements=statements
            ):
                if "UNION ALL" in statement:
                    statements.append((statement, parameters))

            event.listen(db.engine.sync_engine, "before_cursor_execute", capture)
            try:
                result = await search(db, None, query)
            finally:
                event.remove(db.engine.sync_engine, "before_cursor_execute", capture)
            samples = []
            for _ in range(args.iterations):
                start = time.perf_counter()
                await search(db, None, query)
                samples.append((time.perf_counter() - start) * 1000)
            samples.sort()
            plans = []
            for statement, parameters in statements:
                raw = await admin.fetchval(
                    "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + statement,
                    *parameters,
                )
                plan = json.loads(raw)[0]
                plans.append(
                    {
                        "kind": "fuzzy" if "similarity(" in statement else "lexical",
                        "sql": statement,
                        "parameters": list(parameters),
                        "explain": plan,
                        "indexes": sorted(
                            {n["Index Name"] for n in nodes(plan["Plan"]) if "Index Name" in n}
                        ),
                        "sequential_scans": [
                            {
                                key: n.get(key)
                                for key in [
                                    "Relation Name",
                                    "Actual Rows",
                                    "Rows Removed by Filter",
                                    "Actual Loops",
                                ]
                            }
                            for n in nodes(plan["Plan"])
                            if n["Node Type"] == "Seq Scan"
                        ],
                    }
                )
            case = {
                "label": label,
                "query": query,
                "hits": len(result.words),
                "median_ms": round(statistics.median(samples), 3),
                "p95_ms": round(samples[math.ceil(len(samples) * 0.95) - 1], 3),
                "max_ms": round(max(samples), 3),
                "top": [vars(hit) for hit in result.words[:3]],
                "plans": plans,
            }
            report["cases"].append(case)
            print(label, query, case["hits"], case["median_ms"], case["p95_ms"], flush=True)
        report["source_plans"] = []
        for table, query in [
            ("words", "dictionry"),
            ("word_aliases", "walking paperss"),
            ("word_aliases", "walkingg"),
            ("sense_forms", "childrenn"),
        ]:
            published = " AND generation_state='done'" if table == "words" else ""
            raw = await admin.fetchval(
                f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) SELECT id FROM {table} "
                f"WHERE match_key OPERATOR(public.%) $1{published}",
                query,
            )
            plan = json.loads(raw)[0]
            report["source_plans"].append({"table": table, "query": query, "explain": plan})
        report["index_sizes"] = [
            dict(record)
            for record in await admin.fetch(
                "SELECT indexrelname,pg_relation_size(indexrelid) bytes FROM pg_stat_user_indexes "
                "WHERE schemaname=current_schema() AND indexrelname LIKE '%trigram'"
            )
        ]
    finally:
        if db is not None:
            await db.close()
        await admin.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await admin.close()
    report["schema_removed"] = True
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(f"Full plans and measurements: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--words", type=int, default=100_000)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if args.words < 1 or args.iterations < 1:
        parser.error("words and iterations must be positive")
    asyncio.run(benchmark(args))
