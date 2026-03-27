"""Core video analysis tool — the orchestration layer.

Ties together frame extraction, multimodal LLM analysis, and content
synthesis into a single async pipeline.  Designed to be called directly
by PydanticAI agent tools or higher-level application code.

Pipeline stages
---------------
1. **Frame extraction** — delegates to :func:`extract_frames` (uniform,
   scene-change, or keyframe strategy).
2. **Per-frame analysis** — sends each frame image to a multimodal LLM
   via ``BinaryContent``, producing a :class:`FrameAnalysis` per frame.
3. **Content synthesis** — aggregates individual frame analyses into a
   single :class:`VideoContentSummary`.

Design choices
--------------
* **Agents are created once per pipeline run**, not per frame.  Each
  ``agent.run()`` is independent (no shared message history) so frames
  are analyzed in isolation — deterministic for the same image+prompt.

* **Partial failure is tolerated.**  If 8 of 10 frames analyse
  successfully, the pipeline returns those 8 plus 2 ``PipelineError``
  entries.  The synthesis step still runs on whatever frames succeeded.

* **Cleanup is guaranteed** via a ``finally`` block that removes frame
  files even if the pipeline raises mid-way.

* **Token usage is tracked per LLM call** and aggregated into
  ``PipelineRunStats`` so callers can monitor cost and latency.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Sequence

from pydantic_ai.messages import BinaryContent, UserContent

from video_autocut.agent.hunyuan_client import create_structured_agent
from video_autocut.domain.enums import ErrorCategory
from video_autocut.domain.models import (
    ExtractedFrame,
    PipelineError,
    PipelineRunStats,
    TokenUsage,
)
from video_autocut.domain.results import VideoAnalysisResult
from video_autocut.domain.script_models import FrameAnalysis, VideoContentSummary
from video_autocut.infrastructure.reliability import get_run_id, safe_cleanup_frames
from video_autocut.settings import Settings, get_settings
from video_autocut.tools.frame_extraction import extract_frames

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

FRAME_ANALYSIS_PROMPT = """\
You are a professional video analyst.  Analyze the provided video frame \
image and return structured data.

For each frame, identify:
- A concise description of what is visible (1-2 sentences)
- Notable objects, people, or elements in the scene
- Whether the scene is interior, exterior, or mixed
- The visual mood or emotional atmosphere
- Up to three dominant colors

Be precise and objective.  Describe what you see, not what you infer."""

_LANG_INSTRUCTIONS: dict[str, str] = {
    "en": "You MUST write all output text in English.",
    "zh": "你必须用中文撰写所有输出内容。",
}

SYNTHESIS_PROMPT = """\
You are a professional video analyst.  Synthesize the per-frame analyses \
below into a coherent video content summary.

Consider:
- The overall narrative arc and what the video is about
- Recurring themes and topics
- The visual style and cinematography approach
- Pacing and rhythm (based on scene changes and content flow)
- Chronologically notable moments with their timestamps
- The overall emotional tone

Produce a structured summary suitable for downstream shooting script \
generation."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _localized_prompt(prompt: str, lang: str) -> str:
    """Append a language instruction to *prompt* if *lang* is not English."""
    instruction = _LANG_INSTRUCTIONS.get(lang, "")
    if instruction:
        return f"{prompt}\n\n{instruction}"
    return prompt


async def analyze_video(
    video_path: str | Path,
    *,
    strategy: str = "uniform",
    max_frames: int = 10,
    user_prompt: str = "",
    lang: str = "en",
    settings: Settings | None = None,
) -> VideoAnalysisResult:
    """Analyze a video end-to-end: extract → analyze → synthesize.

    This is the primary entry point for the video analysis pipeline.
    It accepts simple primitives so agent tools can call it without
    constructing domain objects.

    Args:
        video_path: Path to the video file on disk.
        strategy: Frame extraction strategy
            (``"uniform"``, ``"scene_change"``, ``"keyframe"``).
        max_frames: Maximum number of frames to extract and analyze.
        user_prompt: Optional context from the user that is included in
            every LLM prompt (e.g. "focus on the outdoor scenes").
        lang: Output language code (``"en"`` or ``"zh"``).
        settings: Explicit settings instance.  When *None* the cached
            singleton from :func:`get_settings` is used.

    Returns:
        A :class:`VideoAnalysisResult` with frame analyses, an optional
        content summary, runtime stats, and any errors.
    """
    if settings is None:
        settings = get_settings()

    pipeline_start = time.monotonic()
    video_path = Path(video_path)
    video_name = video_path.name

    all_errors: list[PipelineError] = []
    all_token_usage: list[TokenUsage] = []
    frame_analyses: list[FrameAnalysis] = []
    content_summary: VideoContentSummary | None = None
    output_dir: Path | None = None

    # Strip "openai:" prefix for display / token-usage tracking.
    model_name = settings.model_name
    if ":" in model_name:
        model_name = model_name.split(":", 1)[1]

    try:
        # Phase 0: Extract frames -------------------------------------------
        extraction_result = extract_frames(
            video_path,
            strategy=strategy,
            max_frames=max_frames,
            deduplicate=True,
        )
        all_errors.extend(extraction_result.errors)

        if extraction_result.frames:
            output_dir = extraction_result.frames[0].path.parent

        if not extraction_result.frames:
            all_errors.append(PipelineError(
                category=ErrorCategory.FRAME_EXTRACTION,
                message=f"No frames extracted from {video_name}",
            ))
        else:
            # Phase 1: Per-frame analysis ------------------------------------
            frame_analyses, analysis_errors, analysis_usage = await _analyze_frames(
                extraction_result.frames,
                video_name=video_name,
                model_name=model_name,
                user_prompt=user_prompt,
                lang=lang,
                settings=settings,
            )
            all_errors.extend(analysis_errors)
            all_token_usage.extend(analysis_usage)

            # Phase 2: Synthesis ---------------------------------------------
            if frame_analyses:
                summary, synth_usage, synth_error = await _synthesize_summary(
                    frame_analyses,
                    video_name=video_name,
                    model_name=model_name,
                    user_prompt=user_prompt,
                    lang=lang,
                    settings=settings,
                )
                content_summary = summary
                if synth_usage:
                    all_token_usage.append(synth_usage)
                if synth_error:
                    all_errors.append(synth_error)

    finally:
        # Cleanup frame files regardless of success or failure.
        if output_dir is not None:
            safe_cleanup_frames(output_dir, label="pipeline_cleanup")

    # Build stats
    total_duration = time.monotonic() - pipeline_start
    stats = PipelineRunStats(
        total_duration_seconds=round(total_duration, 3),
        llm_calls=all_token_usage,
        total_prompt_tokens=sum(u.prompt_tokens for u in all_token_usage),
        total_completion_tokens=sum(u.completion_tokens for u in all_token_usage),
        frames_extracted=len(extraction_result.frames),
        frames_analyzed=len(frame_analyses),
    )

    return VideoAnalysisResult(
        video_name=video_name,
        frame_analyses=frame_analyses,
        content_summary=content_summary,
        stats=stats,
        errors=all_errors,
    )


# ---------------------------------------------------------------------------
# Phase 1: per-frame analysis
# ---------------------------------------------------------------------------


async def _analyze_frames(
    frames: list[ExtractedFrame],
    *,
    video_name: str,
    model_name: str,
    user_prompt: str,
    lang: str,
    settings: Settings,
) -> tuple[list[FrameAnalysis], list[PipelineError], list[TokenUsage]]:
    """Analyze extracted frames one-by-one with the vision LLM.

    Returns:
        Tuple of (successful analyses, errors, token usage records).
    """
    analyses: list[FrameAnalysis] = []
    errors: list[PipelineError] = []
    usage_records: list[TokenUsage] = []

    # Create the structured agent once and reuse for all frames.
    agent = create_structured_agent(
        FrameAnalysis,
        settings=settings,
        system_prompt=_localized_prompt(FRAME_ANALYSIS_PROMPT, lang),
    )

    total = len(frames)
    for frame in frames:
        try:
            analysis, token_rec = await _analyze_single_frame(
                frame,
                agent=agent,
                video_name=video_name,
                model_name=model_name,
                total_frames=total,
                user_prompt=user_prompt,
            )
            analyses.append(analysis)
            usage_records.append(token_rec)
        except Exception as exc:
            logger.warning(
                "Frame %d analysis failed: %s", frame.frame_index, exc,
            )
            errors.append(PipelineError(
                category=ErrorCategory.FRAME_ANALYSIS,
                message=f"Frame {frame.frame_index} analysis failed: {exc}",
                detail=str(type(exc).__name__),
                frame_index=frame.frame_index,
                timestamp_seconds=frame.timestamp_seconds,
            ))

    return analyses, errors, usage_records


async def _analyze_single_frame(
    frame: ExtractedFrame,
    *,
    agent,
    video_name: str,
    model_name: str,
    total_frames: int,
    user_prompt: str,
) -> tuple[FrameAnalysis, TokenUsage]:
    """Analyze a single frame image and return structured output plus usage."""
    frame_bytes = frame.path.read_bytes()
    image_content = BinaryContent(data=frame_bytes, media_type="image/jpeg")

    # Build the per-frame user prompt as a sequence of text + image.
    user_context = f"\nAdditional context: {user_prompt}" if user_prompt else ""
    prompt_parts: Sequence[UserContent] = [
        (
            f"Analyze this video frame.\n"
            f"Video: {video_name}\n"
            f"Frame {frame.frame_index} of {total_frames}, "
            f"captured at {frame.timestamp_seconds:.1f}s."
            f"{user_context}"
        ),
        image_content,
    ]

    start = time.monotonic()
    result = await agent.run(prompt_parts)
    duration = time.monotonic() - start

    usage = result.usage()
    token_rec = TokenUsage(
        model_name=model_name,
        prompt_tokens=usage.input_tokens,
        completion_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        duration_seconds=round(duration, 3),
        step_name="frame_analysis",
    )

    logger.debug(
        "[%s] Frame %d analyzed in %.1fs (%d tokens)",
        get_run_id(), frame.frame_index, duration, usage.total_tokens,
    )
    return result.output, token_rec


# ---------------------------------------------------------------------------
# Phase 2: content synthesis
# ---------------------------------------------------------------------------


async def _synthesize_summary(
    frame_analyses: list[FrameAnalysis],
    *,
    video_name: str,
    model_name: str,
    user_prompt: str,
    lang: str,
    settings: Settings,
) -> tuple[VideoContentSummary | None, TokenUsage | None, PipelineError | None]:
    """Aggregate per-frame analyses into a single content summary.

    Returns:
        Tuple of (summary_or_none, token_usage_or_none, error_or_none).
    """
    agent = create_structured_agent(
        VideoContentSummary,
        settings=settings,
        system_prompt=_localized_prompt(SYNTHESIS_PROMPT, lang),
    )

    # Serialize frame analyses to JSON for the text prompt.
    analyses_json = "\n".join(
        fa.model_dump_json(indent=2) for fa in frame_analyses
    )
    user_context = f"\nAdditional context: {user_prompt}" if user_prompt else ""
    prompt = (
        f"Summarize this video based on {len(frame_analyses)} analyzed frames "
        f"from \"{video_name}\":\n\n"
        f"{analyses_json}"
        f"{user_context}"
    )

    try:
        start = time.monotonic()
        result = await agent.run(prompt)
        duration = time.monotonic() - start

        usage = result.usage()
        token_rec = TokenUsage(
            model_name=model_name,
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            duration_seconds=round(duration, 3),
            step_name="content_synthesis",
        )
        logger.info(
            "[%s] Synthesis completed in %.1fs (%d tokens)",
            get_run_id(), duration, usage.total_tokens,
        )
        return result.output, token_rec, None

    except Exception as exc:
        logger.warning("Content synthesis failed: %s", exc)
        error = PipelineError(
            category=ErrorCategory.FRAME_ANALYSIS,
            message=f"Content synthesis failed: {exc}",
            detail=str(type(exc).__name__),
        )
        return None, None, error
