"""Shared root .env loading for examples/benchmarks, never for the installed library."""

import argparse
import os
from pathlib import Path

from dotenv import dotenv_values

from lexi_ai import DecisionConfig, LLMConfig

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = ROOT / ".env"

PROVIDER_VARIABLES = (
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "LLM_STRUCTURED_OUTPUTS",
    "LLM_TEMPERATURE",
    "LLM_REASONING_EFFORT",
    "LLM_MAX_RETRIES",
    "DECISION_API_KEY",
    "DECISION_BASE_URL",
    "DECISION_MODEL",
    "DECISION_FALLBACK_MODEL",
)


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db-url", required=True, help="Already initialized generated database")
    parser.add_argument(
        "--cambridge-path",
        default=str(ROOT / "data" / "cambridge.db"),
    )
    parser.add_argument("--db-schema", help="Optional existing PostgreSQL schema")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)


def load_provider_values(env_file: Path = DEFAULT_ENV_FILE):
    # No parent-directory discovery, process-env mutation or variable interpolation.
    env_file = Path(env_file)
    values = dotenv_values(env_file, interpolate=False) if env_file.is_file() else {}
    unexpected = set(values) - set(PROVIDER_VARIABLES)
    if unexpected:
        raise ValueError(f"Provider .env contains unsupported variables: {sorted(unexpected)}")
    for name in PROVIDER_VARIABLES:
        if name in os.environ:
            values[name] = os.environ[name]
    return values


def provider_options(values, prefix, *, require_key=False):
    for field in ("BASE_URL", "MODEL", *(("API_KEY",) if require_key else ())):
        value = values.get(f"{prefix}_{field}")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{prefix}_{field} is required in the root .env or process environment"
            )
    options = {
        "api_key": values.get(f"{prefix}_API_KEY") or None,
        "base_url": values[f"{prefix}_BASE_URL"],
        "model": values[f"{prefix}_MODEL"],
    }
    if prefix == "LLM":
        value = values.get("LLM_STRUCTURED_OUTPUTS", "true")
        if not isinstance(value, str) or value.strip().lower() not in ("true", "false"):
            raise ValueError("LLM_STRUCTURED_OUTPUTS must be true or false")
        options["structured_outputs"] = value.strip().lower() == "true"
        value = values.get("LLM_TEMPERATURE")
        try:
            temperature = float(value) if value is not None and value.strip() else None
            options["temperature"] = LLMConfig(temperature=temperature).temperature
        except ValueError as error:
            raise ValueError(
                "LLM_TEMPERATURE must be a finite number in [0, 2] or blank"
            ) from error
        value = values.get("LLM_REASONING_EFFORT")
        if value is not None and not isinstance(value, str):
            raise ValueError("LLM_REASONING_EFFORT must be a string or blank")
        options["reasoning_effort"] = value.strip() or None if value is not None else None
        value = values.get("LLM_MAX_RETRIES")
        if value is not None and not isinstance(value, str):
            raise ValueError("LLM_MAX_RETRIES must be a non-negative integer or blank")
        try:
            retries = int(value) if value is not None and value.strip() else 2
            options["max_retries"] = LLMConfig(max_retries=retries).max_retries
        except ValueError as error:
            raise ValueError("LLM_MAX_RETRIES must be a non-negative integer or blank") from error
    return options


def create_lexicon(args: argparse.Namespace):
    from lexi_ai import Lexicon

    values = load_provider_values(args.env_file)
    decision = {}
    if (values.get("DECISION_API_KEY") or "").strip():
        decision = provider_options(values, "DECISION", require_key=True)
    llm_config = LLMConfig(**provider_options(values, "LLM", require_key=True))

    return Lexicon(
        db_url=args.db_url,
        cambridge_path=args.cambridge_path,
        db_schema=args.db_schema,
        llm_config=llm_config,
        decision_config=DecisionConfig(
            threshold=args.threshold,
            **decision,
        ),
        decision_fallback_model=values.get("DECISION_FALLBACK_MODEL") or None,
    )
