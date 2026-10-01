"""Runtime configuration (pydantic-settings).

Env-driven with sane dev defaults. The generated dictionary (``db_url``,
read/write) is a separate DB from the read-only Cambridge source.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict

from lexi_ai.vocab import DEFAULT_TTS_FORMAT, DEFAULT_TTS_VOICE


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LEXI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1
    # "" / "json_schema" → strict native parse; "function_calling" → forced single
    # tool, for proxies that don't enforce strict json_schema (loose JSON).
    llm_structured_method: str = ""
    # Reasoning effort (minimal|low|medium|high); empty → omit the field.
    llm_reasoning_effort: str = ""
    # Hard ceiling on a single completion; a response far past it is the model
    # rambling.
    llm_max_tokens: int = 4096
    # Wall-clock ceiling per request; without one a stalled provider holds the
    # caller open indefinitely.
    llm_timeout_seconds: float = 120.0

    # Optional per-task model override for translation; empty → ``llm_model``.
    translate_model: str = ""

    # Generated dictionary DB (read/write).
    db_url: str = "sqlite+aiosqlite:///./lexi.db"
    # PostgreSQL deployments isolate dictionary tables in this schema; SQLite ignores it.
    db_schema: str = "lexi"

    cambridge_db_path: str = "./data"

    # Content-addressed asset cache (translation text in DB, TTS clips on disk).
    # Binary assets are sharded here by content-hash prefix; DB rows store relative paths.
    asset_cache_dir: str = "./lexi-assets"

    # Leave the fields empty to use the raising stub.
    tts_base_url: str = ""
    tts_api_key: str = ""
    tts_model: str = ""
    # Defaulted from `lexi_ai.vocab` so this and the identity rules' fallback cannot drift.
    tts_voice: str = DEFAULT_TTS_VOICE
    tts_format: str = DEFAULT_TTS_FORMAT


def get_settings() -> Settings:
    return Settings()
