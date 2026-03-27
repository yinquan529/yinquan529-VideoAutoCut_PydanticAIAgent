"""Pipeline-level result models.

These models aggregate outputs from multiple pipeline stages (extraction,
analysis, synthesis) into a single top-level result.  They are kept in a
separate file to avoid coupling the pure domain layer (``models.py``,
``script_models.py``) with pipeline orchestration concerns.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from video_autocut.domain.models import PipelineError, PipelineRunStats, TokenUsage
from video_autocut.domain.script_models import (
    FrameAnalysis,
    ShootingScript,
    VideoContentSummary,
)


class VideoAnalysisResult(BaseModel):
    """Complete output of the video analysis pipeline.

    Combines per-frame analyses, an optional aggregated content summary,
    runtime statistics, and any errors encountered during the run.
    """

    model_config = ConfigDict(frozen=True)

    video_name: str = Field(
        description="Name of the source video that was analyzed",
    )
    frame_analyses: list[FrameAnalysis] = Field(
        default_factory=list,
        description="Structured analysis of each successfully analyzed frame",
    )
    content_summary: VideoContentSummary | None = Field(
        default=None,
        description=(
            "Aggregated video content summary synthesized from frame analyses. "
            "None if synthesis was skipped or failed."
        ),
    )
    stats: PipelineRunStats = Field(
        description="Token usage and timing statistics for the pipeline run",
    )
    errors: list[PipelineError] = Field(
        default_factory=list,
        description="Errors encountered during extraction, analysis, or synthesis",
    )


class ScriptGenerationResult(BaseModel):
    """Output of the shooting-script generation step."""

    model_config = ConfigDict(frozen=True)

    script: ShootingScript | None = Field(
        default=None,
        description="The generated shooting script, or None on failure",
    )
    token_usage: TokenUsage | None = Field(
        default=None,
        description="Token consumption and timing for the LLM call",
    )
    error: PipelineError | None = Field(
        default=None,
        description="Error details if generation failed",
    )
