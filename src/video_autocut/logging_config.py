"""Structured logging configuration for VideoAutoCut.

Two modes are supported via the ``json`` parameter:

* **text** (default for interactive use):
  ``2025-03-27 14:30:01 | INFO     | video_autocut.tools | [a1b2c3d4] Message``

* **JSON** (for log aggregation / grep-friendly):
  ``{"ts":"2025-03-27T14:30:01","level":"INFO","logger":"video_autocut.tools","run_id":"a1b2c3d4","msg":"Message"}``

Sensitive fields (API keys, base URLs) are never logged — all log
calls use plain messages rather than dumping Settings objects.

Windows notes
-------------
* Timestamps use ISO-8601 with no timezone — ``datetime.now()`` is fine
  for local debugging.
* UTF-8 encoding is set on the stream handler so Rich and non-ASCII
  filenames render correctly in PowerShell / Windows Terminal.
"""

from __future__ import annotations

import json as _json
import logging
import sys
from datetime import datetime


class _JsonFormatter(logging.Formatter):
    """Emit one JSON object per log record — easy to ``grep`` or pipe to ``jq``."""

    def format(self, record: logging.LogRecord) -> str:
        from video_autocut.infrastructure.reliability import get_run_id

        entry = {
            "ts": datetime.fromtimestamp(record.created).strftime("%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "run_id": get_run_id(),
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1] is not None:
            entry["exc"] = self.formatException(record.exc_info)
        return _json.dumps(entry, ensure_ascii=False)


class _TextFormatter(logging.Formatter):
    """Human-readable text with an optional ``[run_id]`` prefix."""

    def format(self, record: logging.LogRecord) -> str:
        from video_autocut.infrastructure.reliability import get_run_id

        ts = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        rid = get_run_id()
        tag = f" [{rid}]" if rid else ""
        base = f"{ts} | {record.levelname:<8s} | {record.name}{tag} | {record.getMessage()}"
        if record.exc_info and record.exc_info[1] is not None:
            base += "\n" + self.formatException(record.exc_info)
        return base


def setup_logging(level: str = "INFO", *, json: bool = False) -> None:
    """Configure the root logger for VideoAutoCut.

    Args:
        level: Logging threshold (``DEBUG``, ``INFO``, …).
        json: When *True*, emit machine-readable JSON lines instead of
            human-readable text.

    This function is idempotent — safe to call more than once (handlers
    are cleared before adding new ones).
    """
    root = logging.getLogger()
    root.setLevel(level.upper())

    # Remove any existing handlers to avoid duplicates on re-call.
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_JsonFormatter() if json else _TextFormatter())
    root.addHandler(handler)
