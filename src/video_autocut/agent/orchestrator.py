"""Main PydanticAI orchestrator agent for VideoAutoCut.

Registers all pipeline tools (video info, frame extraction, content
analysis, script generation, image batch analysis) on a single
``Agent[VideoDeps, str]`` so the LLM can decide which to call.

Designed to be reusable from both the CLI entry-point and any future
API layer — callers only need to construct :class:`VideoDeps` and
call ``agent.run()``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import BinaryContent, UserContent

from video_autocut.agent.hunyuan_client import create_model, create_structured_agent
from video_autocut.domain.script_models import FrameAnalysis
from video_autocut.infrastructure.exceptions import FFmpegError
from video_autocut.infrastructure.ffmpeg import FFmpegTools
from video_autocut.infrastructure.reliability import with_timeout
from video_autocut.settings import Settings, get_settings
from video_autocut.tools.frame_extraction import extract_frames
from video_autocut.tools.script_generator import generate_script
from video_autocut.tools.script_renderer import render_script
from video_autocut.tools.video_analysis import analyze_video

# Default timeout for long-running tool calls (seconds).
_TOOL_TIMEOUT = 300

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed dependencies
# ---------------------------------------------------------------------------


@dataclass
class VideoDeps:
    """Runtime dependencies injected into every tool call."""

    settings: Settings
    ffmpeg: FFmpegTools


# ---------------------------------------------------------------------------
# System instructions
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTIONS = """\
You are VideoAutoCut, a professional video analysis assistant running \
on the user's local Windows machine.

You have the following tools — use them to fulfil user requests:

1. **get_video_info** — retrieve technical metadata (duration, \
resolution, FPS, codec) for a local video file.  Call this first \
when the user mentions a video.
2. **extract_video_frames** — extract still frames from a video for \
visual inspection.  Choose the appropriate strategy (uniform, \
scene_change, keyframe).
3. **analyze_video_content** — the primary analysis tool.  Uses AI \
vision to describe every extracted frame and synthesize a content \
summary.  **You MUST call this before making any claims about what \
a video contains.**  NEVER describe video content, scenes, objects, \
or moods without first running this tool.
4. **generate_video_script** — produce a full shooting script (scenes, \
shots, narration, music cues) from a video.  Supports styles: \
documentary, promotional, tutorial, social_media, narrative.
5. **analyze_images** — analyze a batch of standalone images \
(JPEG/PNG).  Useful for comparing frames or working with stills \
outside a video.

Guidelines:
- Be concise and structured in your replies.
- Always ground your statements in tool output — do not hallucinate \
video facts.
- When unsure about a video, call the analysis tool rather than \
guessing.
- Present numeric data (duration, resolution, token counts) precisely.
- If a tool returns an error, report it clearly and suggest next \
steps."""


# ---------------------------------------------------------------------------
# Tool functions (plain async functions with RunContext[VideoDeps])
# ---------------------------------------------------------------------------


async def get_video_info(
    ctx: RunContext[VideoDeps],
    video_path: str,
) -> str:
    """Get technical metadata about a video file.

    Returns duration, resolution, FPS, codec, and file size.
    Call this first to understand a video before running analysis.

    Args:
        ctx: Injected run context with dependencies.
        video_path: Absolute or relative path to the video file.
    """
    try:
        meta = ctx.deps.ffmpeg.probe_video(Path(video_path))
    except FFmpegError as exc:
        return f"Error probing video: {exc}"

    return meta.model_dump_json(indent=2)


async def extract_video_frames(
    ctx: RunContext[VideoDeps],
    video_path: str,
    strategy: str = "uniform",
    max_frames: int = 10,
) -> str:
    """Extract still frames from a video file.

    Strategies: 'uniform' (evenly spaced), 'scene_change' (cut
    detection), 'keyframe' (I-frames only).

    Args:
        ctx: Injected run context with dependencies.
        video_path: Path to the video file.
        strategy: Extraction strategy name.
        max_frames: Maximum number of frames to extract (1-200).
    """
    try:
        result = extract_frames(
            video_path,
            strategy=strategy,
            max_frames=max_frames,
        )
    except Exception as exc:
        return f"Frame extraction failed: {exc}"

    parts = [f"Extracted {len(result.frames)} frames from {result.video_name}"]
    parts.append(f"Strategy: {result.strategy.value}")
    if result.frames:
        paths = [str(f.path) for f in result.frames]
        parts.append(f"Paths: {', '.join(paths)}")
    if result.errors:
        parts.append(
            f"Errors ({len(result.errors)}): "
            + "; ".join(e.message for e in result.errors)
        )
    return "\n".join(parts)


async def analyze_video_content(
    ctx: RunContext[VideoDeps],
    video_path: str,
    strategy: str = "uniform",
    max_frames: int = 10,
    focus: str = "",
) -> str:
    """Analyze video content using AI vision.

    This is the primary analysis tool.  ALWAYS call this before making
    any claims about what a video contains.  Returns a per-frame
    analysis and a high-level content summary.

    Args:
        ctx: Injected run context with dependencies.
        video_path: Path to the video file.
        strategy: Frame extraction strategy.
        max_frames: Maximum frames to analyze.
        focus: Optional focus hint (e.g. 'concentrate on outdoor scenes').
    """
    try:
        result = await with_timeout(
            analyze_video(
                video_path,
                strategy=strategy,
                max_frames=max_frames,
                user_prompt=focus,
                settings=ctx.deps.settings,
            ),
            seconds=_TOOL_TIMEOUT,
            label="analyze_video_content",
        )
    except Exception as exc:
        return f"Video analysis failed: {exc}"

    parts = [f"Analysis of {result.video_name}"]
    parts.append(f"Frames analyzed: {result.stats.frames_analyzed}")

    if result.content_summary:
        cs = result.content_summary
        parts.append(f"Summary: {cs.overall_summary}")
        parts.append(f"Themes: {', '.join(cs.themes)}")
        parts.append(f"Visual style: {cs.visual_style}")
        parts.append(f"Pacing: {cs.pacing}")
        parts.append(f"Tone: {cs.estimated_tone}")
        if cs.key_moments:
            moments = "; ".join(
                f"{m.timestamp_seconds:.1f}s — {m.description}"
                for m in cs.key_moments
            )
            parts.append(f"Key moments: {moments}")

    if result.errors:
        parts.append(
            f"Errors ({len(result.errors)}): "
            + "; ".join(e.message for e in result.errors)
        )

    parts.append(
        f"Token usage: {result.stats.total_prompt_tokens} prompt "
        f"+ {result.stats.total_completion_tokens} completion"
    )
    return "\n".join(parts)


async def generate_video_script(
    ctx: RunContext[VideoDeps],
    video_path: str,
    script_type: str = "documentary",
    target_duration: float = 0.0,
    target_audience: str = "",
    style: str = "",
    emphasis: str = "",
) -> str:
    """Generate a professional shooting script from a video.

    Runs full analysis then produces a script with scenes, shots,
    narration cues, and music cues.  Script types: documentary,
    promotional, tutorial, social_media, narrative.

    Args:
        ctx: Injected run context with dependencies.
        video_path: Path to the video file.
        script_type: Genre of the shooting script.
        target_duration: Desired duration in seconds (0 = infer).
        target_audience: Who the video is for.
        style: Visual/editorial style guidance.
        emphasis: Special focus instructions.
    """
    # Phase 1: analyse
    try:
        analysis = await with_timeout(
            analyze_video(video_path, settings=ctx.deps.settings),
            seconds=_TOOL_TIMEOUT,
            label="generate_script.analyze",
        )
    except Exception as exc:
        return f"Video analysis failed: {exc}"

    if not analysis.frame_analyses:
        msg = "No frames could be analyzed."
        if analysis.errors:
            msg += " " + "; ".join(e.message for e in analysis.errors)
        return msg

    # Phase 2: generate script
    try:
        gen_result = await with_timeout(
            generate_script(
                analysis,
                script_type=script_type,
                target_duration=target_duration,
                target_audience=target_audience,
                style=style,
                emphasis=emphasis,
                settings=ctx.deps.settings,
            ),
            seconds=_TOOL_TIMEOUT,
            label="generate_script.generate",
        )
    except Exception as exc:
        return f"Script generation failed: {exc}"

    if gen_result.error:
        return f"Script generation error: {gen_result.error.message}"

    if gen_result.script is None:
        return "Script generation returned no output."

    return render_script(gen_result.script)


async def analyze_images(
    ctx: RunContext[VideoDeps],
    image_paths: list[str],
    prompt: str = "Describe this image in detail.",
) -> str:
    """Analyze a batch of images using AI vision.

    Useful for comparing frames or analyzing standalone images outside
    of a video pipeline.  Each image is analyzed independently.

    Args:
        ctx: Injected run context with dependencies.
        image_paths: List of file paths to JPEG or PNG images.
        prompt: The analysis prompt sent with each image.
    """
    if not image_paths:
        return "No image paths provided."

    agent = create_structured_agent(
        FrameAnalysis,
        settings=ctx.deps.settings,
    )

    results: list[dict[str, str]] = []
    for path_str in image_paths:
        p = Path(path_str)
        if not p.exists():
            results.append({"path": path_str, "error": "File not found"})
            continue

        try:
            image_bytes = p.read_bytes()
            content: Sequence[UserContent] = [
                prompt,
                BinaryContent(data=image_bytes, media_type="image/jpeg"),
            ]
            result = await agent.run(content)
            results.append({
                "path": path_str,
                "description": result.output.description,
                "scene_type": result.output.scene_type.value,
                "mood": result.output.visual_mood,
            })
        except Exception as exc:
            results.append({"path": path_str, "error": str(exc)})

    return json.dumps(results, indent=2)


# Collect all tool functions for registration.
TOOLS = [
    get_video_info,
    extract_video_frames,
    analyze_video_content,
    generate_video_script,
    analyze_images,
]


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


def create_orchestrator(
    settings: Settings | None = None,
) -> Agent[VideoDeps, str]:
    """Build the main tool-equipped orchestrator agent.

    Args:
        settings: Explicit settings.  Defaults to :func:`get_settings`.

    Returns:
        A ``Agent[VideoDeps, str]`` with all video tools registered.
    """
    if settings is None:
        settings = get_settings()

    model = create_model(settings)

    return Agent(
        model,
        deps_type=VideoDeps,
        system_prompt=SYSTEM_INSTRUCTIONS,
        retries=settings.max_retries,
        tools=TOOLS,
    )


# ---------------------------------------------------------------------------
# One-shot convenience runner
# ---------------------------------------------------------------------------


async def run_agent(
    user_prompt: str,
    *,
    settings: Settings | None = None,
) -> str:
    """Create the orchestrator, build deps, run a single prompt.

    This is the simplest entry-point for callers who just want a text
    answer.  For multi-turn conversations, use :func:`create_orchestrator`
    directly.

    Args:
        user_prompt: The user's question or instruction.
        settings: Explicit settings.  Defaults to :func:`get_settings`.

    Returns:
        The agent's text response.
    """
    if settings is None:
        settings = get_settings()

    agent = create_orchestrator(settings)
    deps = VideoDeps(
        settings=settings,
        ffmpeg=FFmpegTools(settings.ffmpeg_path, settings.ffprobe_path),
    )
    result = await agent.run(user_prompt, deps=deps)
    return result.output
