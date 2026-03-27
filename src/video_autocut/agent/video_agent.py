"""Backward-compatible agent factory.

Delegates to :mod:`video_autocut.agent.hunyuan_client` so existing call
sites (``app/main.py``) keep working without changes.
"""

from __future__ import annotations

from pydantic_ai import Agent

from video_autocut.agent.hunyuan_client import create_agent as _create_agent
from video_autocut.settings import Settings


def create_agent(settings: Settings) -> Agent[None, str]:
    """Return a text-output agent backed by the Hunyuan model.

    Args:
        settings: Application settings (must include Hunyuan credentials).
    """
    return _create_agent(settings=settings)
