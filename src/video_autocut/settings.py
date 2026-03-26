from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    openai_api_key: str = ""
    model_name: str = "openai:gpt-4o-mini"
    log_level: str = "INFO"
    ffmpeg_path: Path | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
