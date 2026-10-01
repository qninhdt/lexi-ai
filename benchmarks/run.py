"""One-model grading runs. Use `uv run python -m benchmarks.run --help`."""

import argparse
import asyncio
import hashlib
import json
import math
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from typesafe_sdk import AsyncTypeSafeClient, Noul, RetryPolicy

import lexi_ai
from examples._config import DEFAULT_ENV_FILE, load_provider_values, provider_options
from lexi_ai import DecisionConfig, DecisionMode, LLMConfig, TokenUsage
from lexi_ai.inference.decision import DecisionModel
from lexi_ai.inference.prompting import render_decision
from lexi_ai.vocab import POS_TAGS, QUESTION_FORMATS

from .scoring import summarize, validate_pricing

TASKS = (
    "grade_single_word_1",
    "grade_single_word_2",
    "grade_word_to_definition_1",
    "grade_word_to_definition_2",
    "grade_word_to_usage_1",
    "grade_word_to_usage_2",
)
MODES = (DecisionMode.DECISION_ONLY, DecisionMode.LLM_ONLY)
PROMPTS = "questions/prompts/decision/"
DEFAULT_CONFIG = Path(__file__).with_name("config.json")


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def prepare_case(case, *, allow_examples=False):
    """Validate labels against the actual rendered task before spending inference."""
    if not isinstance(case, dict) or not _text(case.get("id")):
        raise ValueError("case.id must be a nonempty string")
    if type(case.get("example", False)) is not bool:
        raise ValueError("case.example must be a boolean")
    if case.get("example") and not allow_examples:
        raise ValueError("template case: review the GT and remove example=true before running")
    task = case.get("task")
    if task not in TASKS:
        raise ValueError(f"unsupported grading task: {task!r}")
    context = case.get("input")
    expected = case.get("expected")
    if not isinstance(context, dict) or not isinstance(expected, dict):
        raise ValueError("case.input and case.expected must be objects")
    for field in ("answer", "question", "meaning"):
        if field == "answer" or field in context:
            if not _text(context.get(field)):
                raise ValueError(f"input.{field} must be a nonempty string")
    if task == "grade_single_word_1":
        if not _text(context.get("question")) or context.get("question_type") not in {
            kind for kind, formats in QUESTION_FORMATS.items() if "single_word" in formats
        }:
            raise ValueError("input requires question and a single-word question_type")
    elif task == "grade_single_word_2" and not _text(context.get("question")):
        raise ValueError("input.question is required")
    if task != "grade_single_word_1":
        field = "matched_word" if task == "grade_single_word_2" else "word"
        word = context.get(field)
        if not isinstance(word, dict) or not _text(word.get("lemma")):
            raise ValueError(f"input.{field}.lemma is required")
        if task in ("grade_single_word_2", "grade_word_to_definition_1"):
            senses = word.get("senses")
            if not isinstance(senses, list) or not senses:
                raise ValueError(f"input.{field}.senses requires the complete neutral inventory")
            ids = set()
            for sense in senses:
                if (
                    not isinstance(sense, dict)
                    or type(sense.get("id")) not in (str, int)
                    or not str(sense["id"]).strip()
                    or sense.get("pos") not in POS_TAGS
                    or not _text(sense.get("definition"))
                ):
                    raise ValueError("each Sense requires id, valid pos and definition")
                key = str(sense["id"])
                if key in ids:
                    raise ValueError("Sense IDs must be unique within the inventory")
                ids.add(key)
    if task == "grade_word_to_definition_2":
        sense = context.get("selectedSense")
        if not isinstance(sense, dict) or not _text(sense.get("definition")):
            raise ValueError("input.selectedSense.definition is required")
    if task == "grade_word_to_usage_2" and not _text(context.get("meaning")):
        raise ValueError("input.meaning is required")
    state, questions = render_decision(PROMPTS + task + ".json", **context)
    if set(expected) != set(questions):
        raise ValueError(f"expected must contain exactly {sorted(questions)}")
    for name, question in questions.items():
        value = expected[name]
        if isinstance(question, Noul):
            if type(value) is not bool:
                raise ValueError(f"expected.{name} must be a boolean, not a probability")
        elif not isinstance(value, str) or value not in question.criteria:
            raise ValueError(f"expected.{name} must be one of {list(question.criteria)}")
    return state, questions


def load_dataset(path, *, allow_examples=False):
    path = Path(path)
    paths = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    cases, ids = [], set()
    for file in paths:
        for line_number, line in enumerate(file.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                case = json.loads(line)
                prepare_case(case, allow_examples=allow_examples)
                if case["id"] in ids:
                    raise ValueError(f"duplicate case ID: {case['id']}")
            except (ValueError, TypeError) as error:
                raise ValueError(f"{file}:{line_number}: {error}") from error
            ids.add(case["id"])
            cases.append(case)
    if not cases:
        raise ValueError("dataset contains no cases")
    return cases


def load_settings(path, benchmark):
    settings = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(settings, dict) or not isinstance(settings.get("benchmarks"), dict):
        raise ValueError("benchmark config must contain a benchmarks object")
    profile = settings["benchmarks"].get(benchmark)
    if not isinstance(profile, dict):
        raise ValueError(f"unknown benchmark: {benchmark!r}; choose {list(settings['benchmarks'])}")
    config = dict(profile)
    if any(field in config for field in ("api_key", "base_url", "model")):
        raise ValueError("provider key/URL/model belong in the root .env, not benchmark config")
    mode = DecisionMode(config.get("mode"))
    if mode not in MODES:
        raise ValueError("benchmark mode must be decision_only or llm_only")
    if "structured_outputs" in config:
        if type(config["structured_outputs"]) is not bool:
            raise ValueError("benchmark structured_outputs must be a boolean")
        if mode != DecisionMode.LLM_ONLY:
            raise ValueError("structured_outputs applies only to llm_only benchmarks")
    if "temperature" in config:
        if mode != DecisionMode.LLM_ONLY:
            raise ValueError("temperature applies only to llm_only benchmarks")
        LLMConfig(temperature=config["temperature"])
    for name in ("threshold", "timeout"):
        value = config.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"config.{name} must be positive and finite")
    if config["threshold"] > 1 or config["timeout"] > 120:
        raise ValueError("threshold must be <= 1 and timeout must be <= 120 seconds")
    prices = settings.get("pricing", {})
    if not isinstance(prices, dict):
        raise ValueError("pricing must map model IDs to price snapshots")
    for snapshot in prices.values():
        validate_pricing(snapshot)
    return config, prices


def select_pricing(config, prices, model):
    key = config.get("pricing", model)
    if "pricing" in config and key not in prices:
        raise ValueError(f"unknown pricing entry: {key!r}")
    return prices.get(key)


def load_config(path, benchmark, *, env_file=DEFAULT_ENV_FILE):
    config, prices = load_settings(path, benchmark)
    prefix = "DECISION" if config["mode"] == DecisionMode.DECISION_ONLY else "LLM"
    provider = provider_options(load_provider_values(env_file), prefix, require_key=True)
    for field in ("structured_outputs", "temperature"):
        if field in config:
            provider.pop(field, None)
    config.update(provider)
    config["pricing"] = select_pricing(config, prices, config["model"])
    return config


@asynccontextmanager
async def configured_model(config):
    if config["mode"] == DecisionMode.DECISION_ONLY:
        decision = DecisionConfig(
            config["threshold"],
            api_key=config["api_key"],
            base_url=config["base_url"],
            model=config["model"],
        )
        async with AsyncTypeSafeClient(
            api_key=decision.api_key,
            base_url=decision.base_url,
            model=decision.model,
            timeout=config["timeout"],
            retry=RetryPolicy(max_retries=0),
        ) as primary:
            yield DecisionModel(decision, primary=primary)
    else:
        model = DecisionModel(
            DecisionConfig(config["threshold"]),
            llm_config=LLMConfig(
                api_key=config["api_key"],
                base_url=config["base_url"],
                model=config["model"],
                timeout=config["timeout"],
                structured_outputs=config["structured_outputs"],
                temperature=config["temperature"],
            ),
        )
        try:
            # Initialize outside per-case timing, just like the Decision client.
            model._fallback(mode=DecisionMode.LLM_ONLY)
            yield model
        finally:
            await model.close()


def _write_json(path, value):
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write("\n")


async def run_benchmark(cases, model, *, mode, timeout, output, pricing=None, metadata=None):
    mode = DecisionMode(mode)
    if mode not in MODES:
        raise ValueError("benchmark mode must be decision_only or llm_only")
    if not cases:
        raise ValueError("dataset contains no cases")
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive and finite")
    prepared = [prepare_case(case) for case in cases]
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("duplicate case ID")
    validate_pricing(pricing)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    root = Path(lexi_ai.__file__).parent
    tasks = sorted({case["task"] for case in cases})
    manifest = {
        **(metadata or {}),
        "mode": mode.value,
        "threshold": model.config.threshold,
        "timeout_seconds": timeout,
        "structured_outputs": (
            model.llm_config.structured_outputs if mode == DecisionMode.LLM_ONLY else None
        ),
        "temperature": model.llm_config.temperature if mode == DecisionMode.LLM_ONLY else None,
        "concurrency": 1,
        "repetitions": 1,
        "started_at": datetime.now(UTC).isoformat(),
        "dataset_sha256": hashlib.sha256(
            json.dumps(cases, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest(),
        "prompt_sha256": {
            task: hashlib.sha256((root / PROMPTS / (task + ".json")).read_bytes()).hexdigest()
            for task in tasks
        },
        "pricing": pricing,
    }
    _write_json(output / "manifest.json", manifest)
    records = []
    with (output / "results.jsonl").open("x", encoding="utf-8") as file:
        for case, (state, questions) in zip(cases, prepared, strict=True):
            record = {
                "id": case["id"],
                "task": case["task"],
                "expected": case["expected"],
                "prediction": None,
                "scores": {},
                "status": "ok",
                "error": None,
            }
            usage = []
            started = perf_counter()
            try:
                async with asyncio.timeout(timeout):
                    response, usage = await model.decide(
                        state, questions, mode=mode, with_usage=True
                    )
                record["prediction"] = {
                    name: model.config.accepts(response.nouls[name].noul)
                    if isinstance(question, Noul)
                    else response.choices[name].choice
                    for name, question in questions.items()
                }
                record["scores"] = {
                    name: response.nouls[name].noul
                    if isinstance(question, Noul)
                    else response.choices[name].confidence
                    for name, question in questions.items()
                }
            except Exception as error:
                record["status"] = "timeout" if isinstance(error, TimeoutError) else "error"
                # Do not persist provider exception bodies: they may contain
                # submitted text, headers or credentials.
                record["error"] = type(error).__name__
                usage = (
                    getattr(error, "usage", None)
                    or getattr(error.__cause__, "usage", None)
                    or usage
                )
                if not usage:
                    usage = [TokenUsage(None, None, None, None, None)]
            record["latency_ms"] = (perf_counter() - started) * 1_000
            record["usage"] = [asdict(item) for item in usage]
            record["correct"] = (
                record["status"] == "ok" and record["prediction"] == record["expected"]
            )
            file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            file.flush()
            records.append(record)
    summary = summarize(records, pricing)
    summary["mode"] = mode.value
    if metadata and "model" in metadata:
        summary["model"] = metadata["model"]
    _write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", type=Path, help="Grading JSONL file or directory")
    source.add_argument("--results", type=Path, help="Rescore saved results without provider calls")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Benchmark profiles and per-model prices (default: benchmarks/config.json)",
    )
    parser.add_argument("--benchmark", help="Profile name in benchmark config; required to run")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_FILE,
        help="Shared provider settings (default: root .env)",
    )
    parser.add_argument("--output", type=Path, help="New output directory (never overwrite a run)")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Check dataset format, including illustrative templates, without provider calls",
    )
    args = parser.parse_args()
    if args.validate_only and args.results:
        parser.error("--validate-only requires --dataset")
    if args.results:
        records = [
            json.loads(line) for line in args.results.read_text().splitlines() if line.strip()
        ]
        manifest = json.loads((args.results.parent / "manifest.json").read_text())
        if args.benchmark:
            config, prices = load_settings(args.config, args.benchmark)
            pricing = select_pricing(config, prices, manifest.get("model"))
        else:
            pricing = manifest.get("pricing")
        summary = summarize(records, pricing)
        if args.output:
            args.output.mkdir(parents=True, exist_ok=False)
            _write_json(args.output / "summary.json", summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
        return
    cases = load_dataset(args.dataset, allow_examples=args.validate_only)
    if args.validate_only:
        print(f"Valid: {len(cases)} cases, {len({case['task'] for case in cases})} grading tasks")
        return
    if args.benchmark is None:
        parser.error("--benchmark is required to run a provider")
    config = load_config(args.config, args.benchmark, env_file=args.env_file)
    output = args.output or Path("benchmark-results") / (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    )
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")

    async def run():
        async with configured_model(config) as model:
            return await run_benchmark(
                cases,
                model,
                mode=config["mode"],
                timeout=config["timeout"],
                output=output,
                pricing=config.get("pricing"),
                metadata={
                    "benchmark": args.benchmark,
                    "model": config["model"],
                    "base_url": config["base_url"],
                    "retries": 0,
                },
            )

    summary = asyncio.run(run())
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
