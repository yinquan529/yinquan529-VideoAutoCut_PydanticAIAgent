"""Shared pytest fixtures for the video-autocut test suite.

Every test module in ``tests/`` automatically inherits these fixtures
without an explicit import.

Windows Testing Notes
---------------------
- Run from the project root: ``py -m pytest tests/ -v``
- With coverage: ``py -m pytest tests/ -v --cov=video_autocut``
- ``tmp_path`` returns Windows-native paths automatically.
- ``ffmpeg_path`` / ``ffprobe_path`` use ``Path("ffmpeg")`` — no ``.exe``
  needed since subprocess is always mocked in tests.
- Environment variables: use ``set`` (not ``export``) or a ``.env`` file.
- Line endings: Git should handle CRLF via ``.gitattributes``; tests compare
  string content, not raw bytes.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from helpers import make_settings
from video_autocut.infrastructure.ffmpeg import FFmpegTools
from video_autocut.settings import Settings, get_settings

# ---------------------------------------------------------------------------
# Settings fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Clear the LRU-cached ``get_settings()`` before and after every test."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture()
def test_settings() -> Settings:
    """Convenience fixture returning ``make_settings()`` with no overrides."""
    return make_settings()


@pytest.fixture()
def mock_deps():
    """``VideoDeps`` with a ``MagicMock(spec=FFmpegTools)``."""
    from video_autocut.agent.orchestrator import VideoDeps

    return VideoDeps(settings=make_settings(), ffmpeg=MagicMock(spec=FFmpegTools))


# ---------------------------------------------------------------------------
# CI safety: block real HTTP requests
# ---------------------------------------------------------------------------

_BLOCKED_MSG = (
    "Real HTTP request intercepted during tests!  "
    "Use mocks or pydantic_ai.models.test.TestModel instead."
)


@pytest.fixture(autouse=True)
def _block_real_http_requests(monkeypatch: pytest.MonkeyPatch):
    """Prevent any real HTTP traffic from leaking during the test run.

    Both sync and async ``httpx`` send methods are replaced with a function
    that raises ``RuntimeError`` immediately.  Individual tests that need
    controlled HTTP (rare) can override at a narrower scope.
    """
    import httpx

    def _blocked(*args: object, **kwargs: object) -> None:  # noqa: ARG001
        raise RuntimeError(_BLOCKED_MSG)

    monkeypatch.setattr(httpx.AsyncClient, "send", _blocked)
    monkeypatch.setattr(httpx.Client, "send", _blocked)
