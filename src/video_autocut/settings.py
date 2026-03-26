from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SettingsValidationError(Exception):
    """Raised when one or more required settings are missing or invalid."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        bullet_list = "\n".join(f"  {i}. {e}" for i, e in enumerate(errors, 1))
        super().__init__(
            f"Settings validation failed with {len(errors)} error(s):\n{bullet_list}"
        )


class Settings(BaseSettings):
    """Application configuration loaded from environment variables and .env file.

    Required fields (``hunyuan_api_key``, ``hunyuan_base_url``) use empty-string
    defaults so the object can always be constructed.  Call ``validate_settings()``
    at startup to enforce that they are non-empty and to create working directories.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Hunyuan API (required) ------------------------------------------------
    hunyuan_api_key: str = Field(
        default="",
        description="Hunyuan API key for LLM access",
    )
    hunyuan_base_url: str = Field(
        default="",
        description="OpenAI-compatible base URL for the Hunyuan API",
    )
    model_name: str = Field(
        default="openai:hunyuan-turbos-latest",
        description="PydanticAI model identifier",
    )

    # --- Tool paths ------------------------------------------------------------
    ffmpeg_path: Path = Field(
        default=Path("ffmpeg"),
        description="Path to the ffmpeg executable",
    )
    ffprobe_path: Path = Field(
        default=Path("ffprobe"),
        description="Path to the ffprobe executable",
    )

    # --- Directories -----------------------------------------------------------
    temp_frames_dir: Path = Field(
        default=Path(".video_autocut/frames"),
        description="Temporary directory for extracted video frames",
    )
    output_dir: Path = Field(
        default=Path(".video_autocut/output"),
        description="Directory for final output files",
    )

    # --- Behaviour -------------------------------------------------------------
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retries for LLM API calls",
    )
    request_timeout_seconds: int = Field(
        default=60,
        ge=5,
        le=600,
        description="Timeout for a single LLM API call in seconds",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton ``Settings`` instance."""
    return Settings()


def validate_settings() -> Settings:
    """Load settings, check required values, and create working directories.

    Raises ``SettingsValidationError`` listing **all** problems found, so the
    developer can fix them in one pass rather than playing whack-a-mole.
    """
    settings = get_settings()
    errors: list[str] = []

    if not settings.hunyuan_api_key:
        errors.append(
            "HUNYUAN_API_KEY is required. "
            "Set it in your .env file or as an environment variable."
        )
    if not settings.hunyuan_base_url:
        errors.append(
            "HUNYUAN_BASE_URL is required. "
            "Set it in your .env file or as an environment variable."
        )

    if errors:
        raise SettingsValidationError(errors)

    # Create working directories only after validation passes.
    settings.temp_frames_dir.mkdir(parents=True, exist_ok=True)
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    return settings
