"""Tests for production reliability utilities and logging configuration.

Covers: correlation IDs, transient-error detection, retry_async,
with_timeout, safe_cleanup_frames, StepTimer, and the two log formatters.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from video_autocut.infrastructure.reliability import (
    StepTimer,
    _is_transient,
    get_run_id,
    new_run_id,
    retry_async,
    safe_cleanup_frames,
    with_timeout,
)
from video_autocut.logging_config import (
    _JsonFormatter,
    _TextFormatter,
    setup_logging,
)

# ---------------------------------------------------------------------------
# TestCorrelationId
# ---------------------------------------------------------------------------


class TestCorrelationId:
    def test_new_run_id_returns_8_hex(self):
        rid = new_run_id()
        assert len(rid) == 8
        assert re.fullmatch(r"[0-9a-f]{8}", rid)

    def test_get_run_id_returns_set_value(self):
        rid = new_run_id()
        assert get_run_id() == rid

    def test_get_run_id_default_empty_in_new_thread(self):
        """A fresh thread has no run ID set — should return ''."""
        result = [None]

        def _check():
            result[0] = get_run_id()

        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(_check).result()

        assert result[0] == ""


# ---------------------------------------------------------------------------
# TestIsTransient
# ---------------------------------------------------------------------------


class TestIsTransient:
    def test_timeout_error_is_transient(self):
        assert _is_transient(TimeoutError()) is True

    def test_connection_error_is_transient(self):
        assert _is_transient(ConnectionError()) is True

    def test_runtime_error_not_transient(self):
        assert _is_transient(RuntimeError("boom")) is False

    def test_value_error_not_transient(self):
        assert _is_transient(ValueError("bad")) is False


# ---------------------------------------------------------------------------
# TestRetryAsync
# ---------------------------------------------------------------------------


class TestRetryAsync:
    async def test_success_no_retry(self):
        calls = []

        async def factory():
            calls.append(1)
            return "ok"

        result = await retry_async(factory, max_attempts=3, base_delay=0.01)
        assert result == "ok"
        assert len(calls) == 1

    async def test_retries_on_transient_error(self):
        calls = []

        async def factory():
            calls.append(1)
            if len(calls) < 2:
                raise TimeoutError("transient")
            return "recovered"

        result = await retry_async(factory, max_attempts=3, base_delay=0.01, max_delay=0.05)
        assert result == "recovered"
        assert len(calls) == 2

    async def test_raises_non_transient_immediately(self):
        calls = []

        async def factory():
            calls.append(1)
            raise ValueError("permanent")

        with pytest.raises(ValueError, match="permanent"):
            await retry_async(factory, max_attempts=3, base_delay=0.01)

        assert len(calls) == 1

    async def test_exhausts_max_attempts(self):
        calls = []

        async def factory():
            calls.append(1)
            raise TimeoutError("still failing")

        with pytest.raises(TimeoutError, match="still failing"):
            await retry_async(factory, max_attempts=3, base_delay=0.01, max_delay=0.05)

        assert len(calls) == 3


# ---------------------------------------------------------------------------
# TestWithTimeout
# ---------------------------------------------------------------------------


class TestWithTimeout:
    async def test_completes_within_timeout(self):
        async def fast():
            return 42

        result = await with_timeout(fast(), seconds=5.0, label="fast_op")
        assert result == 42

    async def test_raises_on_timeout(self):
        async def slow():
            await asyncio.sleep(10)

        with pytest.raises(asyncio.TimeoutError, match="slow_op.*timed out"):
            await with_timeout(slow(), seconds=0.05, label="slow_op")


# ---------------------------------------------------------------------------
# TestSafeCleanupFrames
# ---------------------------------------------------------------------------


class TestSafeCleanupFrames:
    def test_removes_image_files(self, tmp_path: Path):
        (tmp_path / "a.jpg").write_bytes(b"\xff\xd8")
        (tmp_path / "b.png").write_bytes(b"\x89PNG")
        count = safe_cleanup_frames(tmp_path)
        assert count == 2
        assert not (tmp_path / "a.jpg").exists()
        assert not (tmp_path / "b.png").exists()

    def test_ignores_non_image_files(self, tmp_path: Path):
        (tmp_path / "notes.txt").write_text("keep me")
        (tmp_path / "img.jpg").write_bytes(b"\xff\xd8")
        count = safe_cleanup_frames(tmp_path)
        assert count == 1
        assert (tmp_path / "notes.txt").exists()

    def test_nonexistent_directory_returns_zero(self, tmp_path: Path):
        missing = tmp_path / "no_such_dir"
        assert safe_cleanup_frames(missing) == 0


# ---------------------------------------------------------------------------
# TestStepTimer
# ---------------------------------------------------------------------------


class TestStepTimer:
    def test_measures_elapsed_time(self):
        with StepTimer("test_step") as t:
            time.sleep(0.02)
        assert t.elapsed > 0.01

    def test_step_name_stored(self):
        with StepTimer("my_step") as t:
            pass
        assert t.step_name == "my_step"


# ---------------------------------------------------------------------------
# TestLoggingConfig
# ---------------------------------------------------------------------------


def _make_log_record(msg: str = "hello") -> logging.LogRecord:
    return logging.LogRecord(
        name="video_autocut.test",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )


class TestLoggingConfig:
    def test_text_formatter_includes_run_id(self):
        rid = new_run_id()
        fmt = _TextFormatter()
        output = fmt.format(_make_log_record("test message"))
        assert f"[{rid}]" in output
        assert "test message" in output
        assert "INFO" in output

    def test_json_formatter_output_is_valid_json(self):
        rid = new_run_id()
        fmt = _JsonFormatter()
        output = fmt.format(_make_log_record("json test"))
        parsed = json.loads(output)
        assert parsed["level"] == "INFO"
        assert parsed["msg"] == "json test"
        assert parsed["run_id"] == rid
        assert "ts" in parsed
        assert "logger" in parsed

    def test_setup_logging_idempotent(self):
        setup_logging("DEBUG")
        setup_logging("DEBUG")
        root = logging.getLogger()
        assert len(root.handlers) == 1
