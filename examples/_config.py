"""Example-only .env loading; the installed library receives explicit parameters."""

import argparse
import os
from pathlib import Path

from dotenv import dotenv_values

from lexi_ai import DecisionConfig, Lexicon, LLMConfig

PROVIDER_VARIABLES = (
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "DECISION_API_KEY",
    "DECISION_BASE_URL",
    "DECISION_MODEL",
    "DECISION_FALLBACK_MODEL",
)


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db-url", required=True, help="Already initialized generated database")
    parser.add_argument(
        "--cambridge-path",
        default=str(Path(__file__).resolve().parent.parent / "data" / "cambridge.db"),
    )
    parser.add_argument("--db-schema", help="Optional existing PostgreSQL schema")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--env-file", type=Path, default=Path(__file__).parent / ".env")


def create_lexicon(args: argparse.Namespace) -> Lexicon:
    # No parent-directory discovery, process-env mutation or variable interpolation.
    values = dotenv_values(args.env_file, interpolate=False) if args.env_file.is_file() else {}
    unexpected = set(values) - set(PROVIDER_VARIABLES)
    if unexpected:
        raise ValueError(f"Example .env contains unsupported variables: {sorted(unexpected)}")
    for name in PROVIDER_VARIABLES:
        if name in os.environ:
            values[name] = os.environ[name]

    return Lexicon(
        db_url=args.db_url,
        cambridge_path=args.cambridge_path,
        db_schema=args.db_schema,
        llm_config=LLMConfig(
            api_key=values.get("LLM_API_KEY") or None,
            base_url=values.get("LLM_BASE_URL") or "https://api.openai.com/v1",
            model=values.get("LLM_MODEL") or "gpt-4o",
        ),
        decision_config=DecisionConfig(
            threshold=args.threshold,
            api_key=values.get("DECISION_API_KEY") or None,
            base_url=values.get("DECISION_BASE_URL") or "https://api.typesafe.ai",
            model=values.get("DECISION_MODEL") or "jev-latest",
        ),
        decision_fallback_model=values.get("DECISION_FALLBACK_MODEL") or None,
    )
