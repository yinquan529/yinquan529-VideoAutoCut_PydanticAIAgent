from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from video_autocut.settings import (
    Settings,
    SettingsValidationError,
    get_settings,
    validate_settings,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

REQUIRED_ENV = {
    "HUNYUAN_API_KEY": "test-key",
    "HUNYUAN_BASE_URL": "https://api.example.com/v1",
}


@pytest.fixture()
def _set_required_env(monkeypatch: pytest.MonkeyPatch):
    for key, val in REQUIRED_ENV.items():
        monkeypatch.setenv(key, val)


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------


class TestDefaults:
    @pytest.mark.usefixtures("_set_required_env")
    def test_optional_fields_have_expected_defaults(self):
        settings = Settings(
            hunyuan_api_key="k",
            hunyuan_base_url="http://x",
        )
        assert settings.model_name == "openai:hunyuan-turbos-latest"
        assert settings.ffmpeg_path == Path("ffmpeg")
        assert settings.ffprobe_path == Path("ffprobe")
        assert settings.temp_frames_dir == Path(".video_autocut/frames")
        assert settings.output_dir == Path(".video_autocut/output")
        assert settings.log_level == "INFO"
        assert settings.max_retries == 3
        assert settings.request_timeout_seconds == 60


# ---------------------------------------------------------------------------
# Environment overrides
# ---------------------------------------------------------------------------


class TestEnvOverrides:
    def test_all_fields_from_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("HUNYUAN_API_KEY", "my-key")
        monkeypatch.setenv("HUNYUAN_BASE_URL", "https://custom.api/v1")
        monkeypatch.setenv("MODEL_NAME", "openai:gpt-4o")
        monkeypatch.setenv("FFMPEG_PATH", "/usr/bin/ffmpeg")
        monkeypatch.setenv("FFPROBE_PATH", "/usr/bin/ffprobe")
        monkeypatch.setenv("TEMP_FRAMES_DIR", "/tmp/frames")
        monkeypatch.setenv("OUTPUT_DIR", "/tmp/output")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("MAX_RETRIES", "5")
        monkeypatch.setenv("REQUEST_TIMEOUT_SECONDS", "120")

        settings = get_settings()

        assert settings.hunyuan_api_key == "my-key"
        assert settings.hunyuan_base_url == "https://custom.api/v1"
        assert settings.model_name == "openai:gpt-4o"
        assert settings.ffmpeg_path == Path("/usr/bin/ffmpeg")
        assert settings.ffprobe_path == Path("/usr/bin/ffprobe")
        assert settings.temp_frames_dir == Path("/tmp/frames")
        assert settings.output_dir == Path("/tmp/output")
        assert settings.log_level == "DEBUG"
        assert settings.max_retries == 5
        assert settings.request_timeout_seconds == 120

    def test_path_fields_accept_strings(self):
        settings = Settings(
            ffmpeg_path="C:\\ffmpeg\\bin\\ffmpeg.exe",  # type: ignore[arg-type]
            ffprobe_path="C:\\ffmpeg\\bin\\ffprobe.exe",  # type: ignore[arg-type]
        )
        assert isinstance(settings.ffmpeg_path, Path)
        assert isinstance(settings.ffprobe_path, Path)

    def test_extra_env_vars_are_ignored(self, monkeypatch: pytest.MonkeyPatch):
        for key, val in REQUIRED_ENV.items():
            monkeypatch.setenv(key, val)
        monkeypatch.setenv("SOME_RANDOM_VAR", "should-not-break")
        settings = get_settings()
        assert settings.hunyuan_api_key == "test-key"


# ---------------------------------------------------------------------------
# Field validation
# ---------------------------------------------------------------------------


class TestFieldValidation:
    def test_invalid_log_level_rejected(self):
        with pytest.raises(ValidationError, match="Input should be"):
            Settings(log_level="TRACE")  # type: ignore[arg-type]

    def test_max_retries_too_low(self):
        with pytest.raises(ValidationError):
            Settings(max_retries=-1)

    def test_max_retries_too_high(self):
        with pytest.raises(ValidationError):
            Settings(max_retries=11)

    def test_request_timeout_too_low(self):
        with pytest.raises(ValidationError):
            Settings(request_timeout_seconds=4)

    def test_request_timeout_too_high(self):
        with pytest.raises(ValidationError):
            Settings(request_timeout_seconds=601)


# ---------------------------------------------------------------------------
# get_settings() caching
# ---------------------------------------------------------------------------


class TestGetSettings:
    @pytest.mark.usefixtures("_set_required_env")
    def test_returns_cached_instance(self):
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2


# ---------------------------------------------------------------------------
# validate_settings()
# ---------------------------------------------------------------------------


class TestValidateSettings:
    def test_creates_directories(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ):
        for key, val in REQUIRED_ENV.items():
            monkeypatch.setenv(key, val)
        frames = tmp_path / "frames"
        output = tmp_path / "output"
        monkeypatch.setenv("TEMP_FRAMES_DIR", str(frames))
        monkeypatch.setenv("OUTPUT_DIR", str(output))

        settings = validate_settings()

        assert frames.is_dir()
        assert output.is_dir()
        assert settings.temp_frames_dir == frames
        assert settings.output_dir == output

    def test_missing_api_key(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("HUNYUAN_BASE_URL", "https://api.example.com/v1")
        monkeypatch.delenv("HUNYUAN_API_KEY", raising=False)

        with pytest.raises(SettingsValidationError, match="HUNYUAN_API_KEY"):
            validate_settings()

    def test_missing_base_url(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("HUNYUAN_API_KEY", "test-key")
        monkeypatch.delenv("HUNYUAN_BASE_URL", raising=False)

        with pytest.raises(SettingsValidationError, match="HUNYUAN_BASE_URL"):
            validate_settings()

    def test_missing_both_reports_all_errors(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.delenv("HUNYUAN_API_KEY", raising=False)
        monkeypatch.delenv("HUNYUAN_BASE_URL", raising=False)

        with pytest.raises(SettingsValidationError) as exc_info:
            validate_settings()

        assert len(exc_info.value.errors) == 2
        joined = str(exc_info.value)
        assert "HUNYUAN_API_KEY" in joined
        assert "HUNYUAN_BASE_URL" in joined
