"""Production reliability utilities: correlation IDs, retry helper, tool timeouts.

All helpers are lightweight, stdlib-only, and designed for local Windows
debugging — no external observability platform required.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
import uuid
from contextvars import ContextVar
from typing import TypeVar

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Correlation / run ID
# ---------------------------------------------------------------------------

_run_id: ContextVar[str] = ContextVar("run_id", default="")


def new_run_id() -> str:
    """Generate and store a short correlation ID for the current run.

    Format: 8 hex chars (first segment of a UUID-4).  Short enough to
    eyeball in log output, unique enough for a single machine.
    """
    rid = uuid.uuid4().hex[:8]
    _run_id.set(rid)
    return rid


def get_run_id() -> str:
    """Return the current correlation ID, or ``""`` if none was set."""
    return _run_id.get()


# ---------------------------------------------------------------------------
# Retry with exponential back-off
# ---------------------------------------------------------------------------

T = TypeVar("T")

_TRANSIENT_NAMES = frozenset({
    "TimeoutError",
    "ConnectionError",
    "httpx.ConnectError",
    "httpx.ReadTimeout",
    "httpx.ConnectTimeout",
    "openai.APIConnectionError",
    "openai.APITimeoutError",
    "openai.RateLimitError",
    "openai.InternalServerError",
})


def _is_transient(exc: BaseException) -> bool:
    """Decide whether *exc* is worth retrying.

    Matches by fully-qualified class name so we do not need hard imports
    of httpx or openai at module level.
    """
    fqn = f"{type(exc).__module__}.{type(exc).__qualname__}"
    return (
        type(exc).__name__ in _TRANSIENT_NAMES
        or fqn in _TRANSIENT_NAMES
    )


async def retry_async(
    coro_factory,
    *,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    label: str = "operation",
):
    """Retry an async callable with exponential back-off + jitter.

    Args:
        coro_factory: A zero-argument callable that returns an awaitable.
        max_attempts: Total attempts (including the first).
        base_delay: Initial delay in seconds before the first retry.
        max_delay: Cap on the delay between retries.
        label: Human label for log messages.

    Returns:
        The result of the awaitable on success.

    Raises:
        The last exception if all attempts fail.
    """
    last_exc: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await coro_factory()
        except Exception as exc:
            last_exc = exc
            if attempt == max_attempts or not _is_transient(exc):
                raise
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            delay *= 0.5 + random.random()  # jitter  # noqa: S311
            logger.warning(
                "[%s] %s failed (attempt %d/%d): %s — retrying in %.1fs",
                get_run_id(), label, attempt, max_attempts, exc, delay,
            )
            await asyncio.sleep(delay)

    raise last_exc  # type: ignore[misc]  # unreachable but keeps mypy happy


# ---------------------------------------------------------------------------
# Async timeout wrapper
# ---------------------------------------------------------------------------


async def with_timeout(coro, *, seconds: float, label: str = "operation"):
    """Run *coro* with a wall-clock timeout.

    Raises ``asyncio.TimeoutError`` with a descriptive message on expiry.
    """
    try:
        return await asyncio.wait_for(coro, timeout=seconds)
    except asyncio.TimeoutError:
        msg = f"{label} timed out after {seconds:.0f}s"
        logger.error("[%s] %s", get_run_id(), msg)
        raise asyncio.TimeoutError(msg) from None


# ---------------------------------------------------------------------------
# Safe cleanup helper
# ---------------------------------------------------------------------------


def safe_cleanup_frames(directory, *, label: str = "cleanup") -> int:
    """Remove frame images from *directory*, tolerating errors.

    Unlike ``FFmpegTools.cleanup_frames``, this catches per-file
    ``PermissionError`` / ``OSError`` and logs a warning instead of
    crashing.  Returns the count of files actually deleted.
    """
    from pathlib import Path

    _IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"})
    directory = Path(directory)

    if not directory.exists():
        logger.debug("[%s] %s: directory does not exist: %s", get_run_id(), label, directory)
        return 0

    count = 0
    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in _IMAGE_EXTS:
            try:
                file_path.unlink()
                count += 1
            except OSError as exc:
                logger.warning(
                    "[%s] %s: could not delete %s: %s",
                    get_run_id(), label, file_path.name, exc,
                )
    logger.debug("[%s] %s: removed %d file(s) from %s", get_run_id(), label, count, directory)
    return count


# ---------------------------------------------------------------------------
# Timing context manager
# ---------------------------------------------------------------------------


class StepTimer:
    """Lightweight timing helper for pipeline steps.

    Usage::

        with StepTimer("frame_extraction") as t:
            result = extract_frames(...)
        print(t.elapsed)  # seconds as float
    """

    def __init__(self, step_name: str) -> None:
        self.step_name = step_name
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> StepTimer:
        self._start = time.monotonic()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.elapsed = round(time.monotonic() - self._start, 3)
        logger.debug(
            "[%s] step=%s duration=%.3fs",
            get_run_id(), self.step_name, self.elapsed,
        )
