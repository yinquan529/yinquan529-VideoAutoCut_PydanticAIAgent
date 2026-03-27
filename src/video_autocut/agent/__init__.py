"""Agent construction and Hunyuan LLM integration.

The orchestrator (``VideoDeps``, ``create_orchestrator``, ``run_agent``)
lives in :mod:`video_autocut.agent.orchestrator` and must be imported
from there directly to avoid circular imports with the tools layer.
"""

from video_autocut.agent.hunyuan_client import (
    create_agent,
    create_model,
    create_structured_agent,
)

__all__ = [
    "create_agent",
    "create_model",
    "create_structured_agent",
]
