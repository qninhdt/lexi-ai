# Lexi-AI

A lazy-generation English learner's dictionary library. It synthesizes
dictionary entries with an LLM on demand, anchored to Cambridge and WordNet for
hallucination control, and caches results in a local database so repeat lookups
cost zero tokens.

## Install

This project uses [uv](https://docs.astral.sh/uv/) for environment and
dependency management.

```bash
uv sync            # create .venv and install runtime + dev deps from the lockfile
```

Run the test suite:

```bash
uv run pytest
```

Lint and format:

```bash
uv run ruff check .
uv run ruff format .
```

## Status

Under active development — see `plans/` for the phased build.
