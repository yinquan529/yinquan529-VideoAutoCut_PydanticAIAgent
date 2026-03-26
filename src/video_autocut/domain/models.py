from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class VideoFile(BaseModel):
    path: Path
    name: str
    duration_seconds: float | None = None


class AnalysisResult(BaseModel):
    video: VideoFile
    summary: str
    segments: list[dict[str, object]] = []
