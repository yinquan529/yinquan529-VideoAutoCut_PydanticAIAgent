from __future__ import annotations

from pydantic_ai import Agent

SYSTEM_PROMPT = (
    "You are a video analysis assistant. You help users analyze video content, "
    "identify key segments, and suggest edits. When asked about a video, provide "
    "structured analysis including scene descriptions, timestamps, and editing "
    "recommendations."
)


def create_agent(model: str) -> Agent:
    return Agent(model=model, system_prompt=SYSTEM_PROMPT)
