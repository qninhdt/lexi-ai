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

    # LLM (OpenAI-compatible via LangChain, decision #10).
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1

    # Generated dictionary DB (read/write). Async SQLite by default.
    db_url: str = "sqlite+aiosqlite:///./lexi.db"

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


def get_settings() -> Settings:
    """Load settings from environment / .env."""
    return Settings()
