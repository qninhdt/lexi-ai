"""Runtime configuration (pydantic-settings).

Env-driven with sane dev defaults. Two DB locations (decision #14): the
generated dictionary (``db_url``, read/write) is separate from the read-only
Cambridge source (``cambridge_db_path``).
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LEXI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM (OpenAI-compatible endpoint, decision #10).
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1
    # Structured-output method for the OpenAI seam (see ``lexi_ai.llm``):
    #   "" / "json_schema" → strict native parse (best on real OpenAI);
    #   "function_calling" → forced single tool, for proxies that don't enforce
    #     strict json_schema (loose JSON: wrong enum casing / missing fields).
    llm_structured_method: str = ""
    # Reasoning effort for reasoning-capable models (minimal|low|medium|high).
    # Empty → omit the field entirely (plain chat models ignore it anyway).
    llm_reasoning_effort: str = ""
    # Hard ceiling on a single completion. Structured output is bounded by its
    # schema, so a response far past this is a model that has stopped emitting the
    # schema and started rambling — billed per token either way. 0 omits the field
    # for endpoints that reject it.
    llm_max_tokens: int = 4096
    # Wall-clock ceiling per request, in seconds. Without one, a provider that
    # accepts a connection and then stalls holds the caller open indefinitely:
    # generation is awaited inside a request, so the stall propagates.
    llm_timeout_seconds: float = 120.0

    # Optional per-task model override for translation. Empty → falls back to
    # ``llm_model`` (shares the same base_url/api_key/temperature).
    translate_model: str = ""

    # Generated dictionary DB (read/write). Async SQLite by default.
    db_url: str = "sqlite+aiosqlite:///./lexi.db"
    # PostgreSQL deployments isolate dictionary tables in this schema. SQLite
    # ignores it and continues to use its default namespace.
    db_schema: str = "lexi"

    # Read-only Cambridge source (plain filesystem path to the SQLite file).
    cambridge_db_path: str = "./data"

    # Local sense embeddings (transformers). Optional feature: needs the
    # ``[embeddings]`` extra (torch + transformers). All env-driven, never
    # hardcoded. Model is loaded lazily on first embed, so these are cheap to set
    # even when the extra isn't installed.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32
    embedding_max_length: int = 256

    # Where those vectors live. Embeddings are NOT in the primary database: they
    # are eventually consistent by design (written post-commit, best-effort,
    # reconciled by a backfill), so they get their own store.
    #
    # Semantic search is an OPT-IN feature and this is its switch:
    #   ``none``    — off (DEFAULT). No index; ``semantic_search`` and
    #                 ``backfill_embeddings`` raise ``SemanticSearchDisabled``, and
    #                 generation simply stores no vectors. Costs nothing: neither
    #                 optional dependency is needed.
    #   ``lancedb`` — embedded, on disk, durable; needs the ``[lancedb]`` extra.
    #                 The right choice in production once the feature is wanted.
    #   ``memory``  — exact-scan, non-durable (vectors die with the process);
    #                 for tests and local experiments, never for production.
    # Enabling also needs the ``[embeddings]`` extra for the encoder itself.
    # ``vector_metric`` must match the encoder's geometry; the Embedder
    # L2-normalizes, so cosine is correct.
    vector_backend: str = "none"
    vector_path: str = "./lexi-vectors"
    vector_metric: str = "cosine"

    # Content-addressed asset cache (translation text in DB, TTS clips on disk).
    # ``asset_cache_dir`` is where binary assets (TTS) are written, sharded by
    # content-hash prefix; DB rows store paths RELATIVE to it.
    asset_cache_dir: str = "./lexi-assets"

    # TTS provider (interface + stub this round — synthesis is NOT wired yet).
    # Defined so the seam is complete/documented; a real provider reads these.
    tts_base_url: str = ""
    tts_api_key: str = ""
    tts_model: str = ""
    tts_voice: str = "alloy"
    tts_format: str = "mp3"

    # Third-party question types are discovered via the ``lexi_ai.question_types``
    # entry-point group, but registered ONLY when their type_id appears here.
    # Empty (default) = built-in types only; untrusted discovery stays opt-in.
    question_type_allowlist: frozenset[str] = frozenset()


def get_settings() -> Settings:
    """Load settings from environment / .env."""
    return Settings()
