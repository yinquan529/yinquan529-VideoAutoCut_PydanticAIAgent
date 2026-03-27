"""Shared test helpers importable by any test module.

Usage::

    from tests.helpers import make_settings
    # or, since tests/ is on sys.path:
    from helpers import make_settings
"""

from __future__ import annotations

from pathlib import Path

from video_autocut.settings import Settings


def make_settings(**overrides: object) -> Settings:
    """Build a ``Settings`` instance with safe test defaults.

    Callers can pass keyword overrides for any ``Settings`` field::

        s = make_settings(max_retries=5, model_name="openai:custom")
    """
    defaults: dict[str, object] = dict(
        hunyuan_api_key="test-key",
        hunyuan_base_url="http://localhost",
        model_name="openai:test-model",
        ffmpeg_path=Path("ffmpeg"),
        ffprobe_path=Path("ffprobe"),
        temp_frames_dir=Path("/tmp/test_frames"),
        output_dir=Path("/tmp/test_output"),
        max_retries=0,
        request_timeout_seconds=10,
    )
    defaults.update(overrides)
    return Settings(**defaults)
