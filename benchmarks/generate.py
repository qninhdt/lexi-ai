"""Generate sources and learner answers, then label answers with the same LLM."""

import argparse
import asyncio
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from lexi_ai import Lexicon, LLMConfig
from lexi_ai.cli.config import DEFAULT_ENV_FILE, load_provider_values, provider_options
from lexi_ai.inference.decision import DecisionModel
from lexi_ai.references.cambridge import Cambridge, encode_reference_id

from .run import TASKS
from .synthetic import (
    GenerationConfig,
    answer_types,
    grade_answers,
    question_plan,
    synthesis_context,
    synthesize,
)
from .workers import map_workers, validate_workers

DEFAULT_CONFIG = Path(__file__).with_name("generation.json")


def load_config(path):
    config = GenerationConfig.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
    validate_workers(config.workers)
    for items in (
        [entry.source_id for entry in config.entries],
        [entry.target for entry in config.entries],
        [theme.key for theme in config.themes],
    ):
        if len(set(items)) != len(items):
            raise ValueError("generation entries and Theme keys must be unique")
    return config


def read_records(path):
    records = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                if record["id"] in records:
                    raise ValueError(f"duplicate checkpoint ID in {path}")
                records[record["id"]] = record
    return records


def append_record(path, records, record):
    # All callbacks run on the same event loop; no cross-thread file/client sharing.
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        file.flush()
    records[record["id"]] = record


def word_key(entry_index, theme=None):
    return f"word:{entry_index}:{theme or 'neutral'}"


async def generate_sources(lexicon, config, jobs, output, *, workers, progress):
    path = output / "sources.jsonl"
    saved = read_records(path)

    def save(_index, records):
        for record in records:
            append_record(path, saved, record)

    async def theme_source(spec):
        key = "theme:" + spec.key
        if key in saved:
            return []
        theme = await lexicon.get_theme(spec.key)
        usage = []
        if theme is None:
            theme, usage = await lexicon.create_theme(
                spec.key, spec.name, spec.concept, with_usage=True
            )
        return [{"id": key, "data": asdict(theme), "usage": [asdict(item) for item in usage]}]

    await map_workers(
        config.themes,
        theme_source,
        workers=workers,
        progress=progress,
        desc="Themes",
        on_result=save,
    )

    async def word_source(item):
        index, theme = item
        key = word_key(index, theme)
        if key in saved:
            return []
        entry = config.entries[index]
        word, usage = await lexicon.generate_word(
            entry.target,
            reference_id=encode_reference_id(entry.source_id),
            theme=theme,
            example_count=config.examples_per_sense,
            with_usage=True,
        )
        if not word.senses or any(
            sense.definition is None or len(sense.examples) != config.examples_per_sense
            for sense in word.senses
        ):
            raise ValueError(
                "source lacks complete definitions or requested examples; use a fresh DB"
            )
        return [{"id": key, "data": asdict(word), "usage": [asdict(item) for item in usage]}]

    await map_workers(
        [(index, None) for index in range(len(config.entries))],
        word_source,
        workers=workers,
        progress=progress,
        desc="Neutral Words",
        on_result=save,
    )
    needed = {index: [] for index in range(len(config.entries))}
    for job in jobs:
        if job["theme"] and job["theme"] not in needed[job["entry"]]:
            needed[job["entry"]].append(job["theme"])

    async def themed_group(index):
        records = []
        # Lexicon requires callers to serialize overlapping generation on one Word.
        # Different Words run concurrently; Themes of the SAME Word do not.
        for theme in needed[index]:
            records.extend(await word_source((index, theme)))
        return records

    await map_workers(
        [index for index, themes in needed.items() if themes],
        themed_group,
        workers=workers,
        progress=progress,
        desc="Themed Word groups",
        on_result=save,
    )


async def generate_answers(lexicon, config, jobs, output, *, workers, progress):
    sources = read_records(output / "sources.jsonl")
    questions = read_records(output / "questions.jsonl")
    answers = read_records(output / "answers.jsonl")
    labels = read_records(output / "labels.jsonl")
    required = {word_key(job["entry"], job["theme"]) for job in jobs} | {
        word_key(index) for index in range(len(config.entries))
    }
    if not required <= sources.keys():
        raise ValueError("source stage is incomplete; run --stage sources first")
    words = {}
    for key in sorted(required):
        record = sources[key]["data"]
        theme = key.split(":", 2)[2]
        word = await lexicon.get_word(record["id"], theme=None if theme == "neutral" else theme)
        if word is None or not word.senses or word.lemma != record["lemma"]:
            raise ValueError("source checkpoint does not match the generated database")
        words[key] = word

    # Use the synthesis model's exact configuration, never Decision or a different
    # fallback model. This is Lexi's existing grading transport, not a new client.
    model = DecisionModel(lexicon.decision_config, llm_config=lexicon.llm_config)

    async def one(job):
        if job["id"] in labels:
            return None
        word = words[word_key(job["entry"])]
        scoped = words[word_key(job["entry"], job["theme"])]
        # Rotate across the complete inventory, including less-common and rare Senses.
        sense_id = word.senses[job["sense_index"] % len(word.senses)].id
        sense = next(sense for sense in scoped.senses if sense.id == sense_id)
        if job["id"] in questions:
            question = await lexicon.get_question(questions[job["id"]]["data"]["id"])
            if question is None:
                raise ValueError("saved Question is missing from the generated database")
        else:
            generated, usage = await lexicon.generate_questions(
                sense.id,
                job["question_type"],
                1,
                theme=job["theme"],
                distractor_count=config.distractors_per_question,
                with_usage=True,
            )
            question = generated[0]
            # Save the Question BEFORE synthesis, so a failed synthesis does not append
            # another Question on resume.
            append_record(
                output / "questions.jsonl",
                questions,
                {
                    "id": job["id"],
                    "data": asdict(question),
                    "job": job,
                    "usage": [asdict(item) for item in usage],
                },
            )
        if job["id"] not in answers:
            kinds = answer_types(job, word, sense)
            context = synthesis_context(question, word, kinds)
            texts, usage = await synthesize(
                lexicon._llm(),
                context,
                question_type=question.question_type,
                structured_outputs=lexicon.llm_config.structured_outputs,
            )
            # Keep the generated strings even if their later grading fails.
            append_record(
                output / "answers.jsonl",
                answers,
                {
                    "id": job["id"],
                    "answers": texts,
                    "answer_types": kinds,
                    "usage": [asdict(item) for item in usage],
                },
            )
        cases, usage = await grade_answers(
            lexicon,
            model,
            job,
            question,
            word,
            sense,
            answers[job["id"]]["answers"],
        )
        return {
            "id": job["id"],
            "cases": cases,
            "label_source": "llm_grading",
            "usage": [asdict(item) for item in usage],
        }

    def save(_index, record):
        if record is not None:
            append_record(output / "labels.jsonl", labels, record)

    try:
        await map_workers(
            jobs,
            one,
            workers=workers,
            progress=progress,
            desc="Questions + answers + grading",
            on_result=save,
        )
    finally:
        await model.close()
    return labels


def export_cases(labels, output):
    directory = output / "cases"
    directory.mkdir(exist_ok=True)
    cases = [case for key in sorted(labels) for case in labels[key]["cases"]]
    for task in TASKS:
        # These are generator-owned derived files; checkpoints remain append-only.
        with (directory / (task + ".jsonl")).open("w", encoding="utf-8") as file:
            for case in cases:
                if case["task"] == task:
                    file.write(json.dumps(case, ensure_ascii=False, allow_nan=False) + "\n")
    return dict(Counter(case["task"] for case in cases))


async def run_generation(
    lexicon,
    config,
    *,
    output,
    stage="all",
    workers=None,
    progress=True,
    resume=False,
):
    workers = config.workers if workers is None else workers
    validate_workers(workers)
    if stage not in {"sources", "questions", "all"}:
        raise ValueError("unknown generation stage")
    jobs = question_plan(config)
    output = Path(output)
    snapshot = config.model_dump()
    fingerprint = hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()
    if resume:
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if manifest["config_sha256"] != fingerprint:
            raise ValueError("resume requires the original generation config")
        if manifest.get("pipeline") != "answers_then_llm_grading":
            raise ValueError("resume requires an answers-then-grading checkpoint")
    else:
        output.mkdir(parents=True, exist_ok=False)
        with (output / "manifest.json").open("x", encoding="utf-8") as file:
            json.dump(
                {
                    "config_sha256": fingerprint,
                    "config": snapshot,
                    "plan": jobs,
                    "pipeline": "answers_then_llm_grading",
                },
                file,
                ensure_ascii=False,
                indent=2,
            )
    if stage in {"sources", "all"}:
        await generate_sources(lexicon, config, jobs, output, workers=workers, progress=progress)
    summary = {"planned_questions": len(jobs), "planned_answers": len(jobs) * 3}
    if stage in {"questions", "all"}:
        labels = await generate_answers(
            lexicon, config, jobs, output, workers=workers, progress=progress
        )
        summary.update(
            questions=len(labels),
            answers=sum(
                len(item["answers"]) for item in read_records(output / "answers.jsonl").values()
            ),
            cases_by_task=export_cases(labels, output),
            label_source="llm_grading",
            label_status="proposed",
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--db-url", help="Already initialized generated DB; no automatic migrations"
    )
    parser.add_argument("--db-schema")
    parser.add_argument("--reference-path", default="")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--stage", choices=["sources", "questions", "all"], default="all")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--validate-only", action="store_true", help="No environment/provider/DB writes"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    workers = config.workers if args.workers is None else args.workers
    validate_workers(workers)

    async def run():
        # Read-only preflight checks every explicitly selected Cambridge entry.
        source = Cambridge(args.reference_path)
        for entry in config.entries:
            evidence = await source.fetch_by_id(entry.source_id)
            if evidence is None or not evidence.senses:
                raise ValueError(f"selected source missing for {entry.target}")
        if args.validate_only:
            jobs = question_plan(config)
            return {
                "entries": len(config.entries),
                "questions": len(jobs),
                "answers": len(jobs) * 3,
                "namespaces": dict(Counter(job["theme"] or "neutral" for job in jobs)),
            }
        if args.db_url is None or args.output is None:
            parser.error("generation requires --db-url and --output")
        provider = provider_options(load_provider_values(args.env_file), "LLM", require_key=True)
        if config.reasoning_effort is not None:
            provider["reasoning_effort"] = config.reasoning_effort
        llm_config = LLMConfig(
            **provider,
            timeout=config.timeout,
            max_completion_tokens=config.max_completion_tokens,
        )
        lexicon = Lexicon(
            args.db_url,
            str(source.path) if args.db_url.startswith("sqlite") else "",
            db_schema=args.db_schema,
            llm_config=llm_config,
        )
        try:
            if args.db_url.startswith("postgresql"):
                await lexicon.import_reference(source.path)
            await lexicon.start()
            return await run_generation(
                lexicon,
                config,
                output=args.output,
                stage=args.stage,
                workers=workers,
                progress=not args.no_progress,
                resume=args.resume,
            )
        finally:
            await lexicon.close()

    print(json.dumps(asyncio.run(run()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
