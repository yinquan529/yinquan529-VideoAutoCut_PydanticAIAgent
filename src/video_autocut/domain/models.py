from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from video_autocut.domain.enums import ErrorCategory, ExtractionStrategy

# ---------------------------------------------------------------------------
# Structured errors (defined first — referenced by FrameExtractionResult)
# ---------------------------------------------------------------------------


class PipelineError(BaseModel):
    """Structured error payload for any pipeline failure."""

    model_config = ConfigDict(frozen=True)

    category: ErrorCategory = Field(
        description="Which pipeline stage produced this error",
    )
    message: str = Field(
        description="Human-readable error description",
    )
    detail: str = Field(
        default="",
        description="Additional technical detail such as traceback excerpt or ffmpeg stderr",
    )
    timestamp_seconds: float | None = Field(
        default=None,
        description="Video timestamp associated with the error, if applicable",
    )
    frame_index: int | None = Field(
        default=None,
        description="Frame index associated with the error, if applicable",
    )


# ---------------------------------------------------------------------------
# Video metadata
# ---------------------------------------------------------------------------


class VideoMetadata(BaseModel):
    """Rich metadata about a source video file, populated by ffprobe."""

    path: Path = Field(description="Absolute filesystem path to the video file")
    name: str = Field(description="Human-readable filename without directory")
    duration_seconds: float | None = Field(
        default=None,
        description="Total duration of the video in seconds",
    )
    width: int | None = Field(
        default=None,
        description="Horizontal resolution in pixels",
    )
    height: int | None = Field(
        default=None,
        description="Vertical resolution in pixels",
    )
    fps: float | None = Field(
        default=None,
        description="Frames per second of the video stream",
    )
    codec: str | None = Field(
        default=None,
        description="Video codec name, e.g. h264 or vp9",
    )
    format_name: str | None = Field(
        default=None,
        description="Container format, e.g. mp4, mkv, or webm",
    )
    file_size_bytes: int | None = Field(
        default=None,
        description="File size in bytes",
    )


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------


class FrameExtractionRequest(BaseModel):
    """Parameters for requesting frame extraction from a video."""

    video: VideoMetadata = Field(
        description="Source video to extract frames from",
    )
    strategy: ExtractionStrategy = Field(
        default=ExtractionStrategy.UNIFORM,
        description="Algorithm for selecting which frames to extract",
    )
    max_frames: int = Field(
        default=10,
        ge=1,
        le=200,
        description="Maximum number of frames to extract",
    )
    output_dir: Path = Field(
        description="Directory where extracted frame images will be written",
    )


class ExtractedFrame(BaseModel):
    """A single frame extracted from a video, stored on disk."""

    model_config = ConfigDict(frozen=True)

    path: Path = Field(
        description="Filesystem path to the extracted frame image",
    )
    timestamp_seconds: float = Field(
        ge=0.0,
        description="Position in the source video where this frame was captured, in seconds",
    )
    frame_index: int = Field(
        ge=0,
        description="Zero-based ordinal index of this frame in the extraction sequence",
    )


class FrameExtractionResult(BaseModel):
    """Outcome of a frame extraction operation."""

    model_config = ConfigDict(frozen=True)

    video_name: str = Field(
        description="Name of the source video",
    )
    strategy: ExtractionStrategy = Field(
        description="Strategy that was used for extraction",
    )
    frames: list[ExtractedFrame] = Field(
        default_factory=list,
        description="Successfully extracted frames",
    )
    errors: list[PipelineError] = Field(
        default_factory=list,
        description="Errors encountered during extraction, if any",
    )


# ---------------------------------------------------------------------------
# Token usage and runtime stats
# ---------------------------------------------------------------------------


class TokenUsage(BaseModel):
    """Token consumption and timing for a single LLM call."""

    model_config = ConfigDict(frozen=True)

    model_name: str = Field(
        description="Identifier of the LLM model used",
    )
    prompt_tokens: int = Field(
        ge=0,
        description="Number of tokens in the prompt or input",
    )
    completion_tokens: int = Field(
        ge=0,
        description="Number of tokens in the completion or output",
    )
    total_tokens: int = Field(
        ge=0,
        description="Sum of prompt and completion tokens",
    )
    duration_seconds: float = Field(
        ge=0.0,
        description="Wall-clock time for the LLM call in seconds",
    )
    step_name: str = Field(
        default="",
        description=(
            "Pipeline step that triggered this call, "
            "e.g. frame_analysis or script_generation"
        ),
    )


class PipelineRunStats(BaseModel):
    """Aggregated statistics for a complete pipeline execution."""

    model_config = ConfigDict(frozen=True)

    total_duration_seconds: float = Field(
        ge=0.0,
        description="Total wall-clock time for the entire pipeline run",
    )
    llm_calls: list[TokenUsage] = Field(
        default_factory=list,
        description="Token usage records for each LLM call made during the run",
    )
    total_prompt_tokens: int = Field(
        default=0,
        ge=0,
        description="Sum of prompt tokens across all LLM calls",
    )
    total_completion_tokens: int = Field(
        default=0,
        ge=0,
        description="Sum of completion tokens across all LLM calls",
    )
    frames_extracted: int = Field(
        default=0,
        ge=0,
        description="Number of frames successfully extracted",
    )
    frames_analyzed: int = Field(
        default=0,
        ge=0,
        description="Number of frames successfully analyzed by the vision LLM",
    )
