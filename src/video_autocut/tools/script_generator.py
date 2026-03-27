"""Shooting-script generator — produces a typed ``ShootingScript`` from
video analysis results via the Hunyuan LLM.

The generator accepts a :class:`VideoAnalysisResult` (from
:func:`analyze_video`) plus creative parameters (script type, audience,
style) and returns a fully structured :class:`ShootingScript` with scenes,
shots, narration cues, and music cues.

Design choices
--------------
* **Single LLM call.**  The entire script is generated in one structured-
  output request.  This keeps the script internally consistent — the LLM
  can balance scene timings, narration pacing, and music cues holistically
  rather than assembling independent fragments.

* **Mode-specific system prompts.**  Each ``ScriptType`` (documentary,
  promotional, tutorial, social_media, narrative) gets a tailored system
  prompt that steers tone, pacing, and structure conventions specific to
  that genre.  A shared base prompt covers the universal requirements.

* **Same token-tracking pattern** as ``video_analysis.py`` — returns a
  ``ScriptGenerationResult`` containing the script, token usage, and
  any errors, so the caller can aggregate stats across pipeline stages.
"""

from __future__ import annotations

import logging
import time

from video_autocut.agent.hunyuan_client import create_structured_agent
from video_autocut.domain.enums import ErrorCategory, ScriptType
from video_autocut.domain.models import PipelineError, TokenUsage
from video_autocut.domain.results import ScriptGenerationResult, VideoAnalysisResult
from video_autocut.domain.script_models import ShootingScript
from video_autocut.settings import Settings, get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_BASE_SYSTEM_PROMPT = """\
You are an expert video director and scriptwriter.  Given the video \
analysis data below, produce a professional shooting script.

Requirements:
- Every scene must have at least one shot with concrete camera direction.
- Shot timecodes must be contiguous and fit within the target duration.
- Narration cues must not overlap and should complement — not repeat — \
the visual content.
- Music cues should reinforce the emotional arc without overwhelming \
dialogue or narration.
- Production notes should be actionable, not vague platitudes."""

_MODE_PROMPTS: dict[ScriptType, str] = {
    ScriptType.DOCUMENTARY: """\
Style: observational documentary.
- Favor wide establishing shots and slow reveals.
- Narration should provide context and insight, not describe the obvious.
- Music should be understated — ambient or sparse orchestral.
- Let the subjects and visuals tell the story; narration fills gaps.""",

    ScriptType.PROMOTIONAL: """\
Style: promotional / commercial.
- Open with a hook shot in the first 3 seconds.
- Use dynamic cuts — medium and close-up shots dominate.
- Narration should be concise, benefit-focused, and end with a clear CTA.
- Music should be upbeat and energetic, matching the brand tone.
- Keep total shot count high relative to duration for visual energy.""",

    ScriptType.TUTORIAL: """\
Style: instructional tutorial.
- Structure scenes around discrete learning steps.
- Favor close-up and insert shots that show detail clearly.
- Narration is the primary information channel — be precise and clear.
- Music should be minimal and unobtrusive (light background only).
- Include brief intro and recap scenes to frame the lesson.""",

    ScriptType.SOCIAL_MEDIA: """\
Style: short-form social media content.
- Assume vertical or square framing; tight shots dominate.
- Front-load the most compelling visual in the first 2 seconds.
- Narration should be punchy, conversational, and < 20 words per cue.
- Music should be trending and high-energy; silence is acceptable for \
impact.
- Keep total scene count low (2-4) with fast internal cuts.""",

    ScriptType.NARRATIVE: """\
Style: narrative / cinematic storytelling.
- Build a three-act structure: setup, confrontation, resolution.
- Vary shot types deliberately — wide for context, close for emotion.
- Narration (if any) should be character-driven or poetic, not expository.
- Music should follow the emotional arc with distinct movements.
- Include scene direction notes for acting beats and transitions.""",
}


def _build_system_prompt(script_type: ScriptType) -> str:
    """Combine the base prompt with the mode-specific addendum."""
    mode = _MODE_PROMPTS.get(script_type, "")
    return f"{_BASE_SYSTEM_PROMPT}\n\n{mode}".strip()


def _build_user_prompt(
    analysis: VideoAnalysisResult,
    script_type: ScriptType,
    target_duration: float,
    target_audience: str,
    style: str,
    emphasis: str,
) -> str:
    """Build the user-facing prompt that carries all creative context."""
    # Serialize the analysis content compactly.
    summary_block = ""
    if analysis.content_summary:
        cs = analysis.content_summary
        summary_block = (
            f"Video summary: {cs.overall_summary}\n"
            f"Themes: {', '.join(cs.themes)}\n"
            f"Visual style: {cs.visual_style}\n"
            f"Pacing: {cs.pacing}\n"
            f"Tone: {cs.estimated_tone}\n"
        )
        if cs.key_moments:
            moments = "; ".join(
                f"{m.timestamp_seconds:.1f}s — {m.description}"
                for m in cs.key_moments
            )
            summary_block += f"Key moments: {moments}\n"

    frames_block = ""
    if analysis.frame_analyses:
        lines = []
        for fa in analysis.frame_analyses:
            lines.append(
                f"  [{fa.timestamp_seconds:.1f}s] {fa.description} "
                f"(mood: {fa.visual_mood}, scene: {fa.scene_type.value})"
            )
        frames_block = "Frame analyses:\n" + "\n".join(lines) + "\n"

    audience_line = f"Target audience: {target_audience}\n" if target_audience else ""
    style_line = f"Style guidance: {style}\n" if style else ""
    emphasis_line = f"Special emphasis: {emphasis}\n" if emphasis else ""

    duration_str = (
        f"Target duration: {target_duration:.0f} seconds.\n"
        if target_duration > 0
        else "Target duration: infer from source material.\n"
    )

    return (
        f"Generate a {script_type.value} shooting script "
        f"for \"{analysis.video_name}\".\n"
        f"{duration_str}"
        f"{audience_line}"
        f"{style_line}"
        f"{emphasis_line}"
        f"\n{summary_block}"
        f"\n{frames_block}"
    ).strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


_LANG_INSTRUCTIONS: dict[str, str] = {
    "en": "You MUST write all output text in English.",
    "zh": "你必须用中文撰写所有输出内容。",
}


def _localized_prompt(prompt: str, lang: str) -> str:
    """Append a language instruction to *prompt* if applicable."""
    instruction = _LANG_INSTRUCTIONS.get(lang, "")
    if instruction:
        return f"{prompt}\n\n{instruction}"
    return prompt


async def generate_script(
    analysis: VideoAnalysisResult,
    *,
    script_type: str | ScriptType = ScriptType.DOCUMENTARY,
    target_duration: float = 0.0,
    target_audience: str = "",
    style: str = "",
    emphasis: str = "",
    lang: str = "en",
    settings: Settings | None = None,
) -> ScriptGenerationResult:
    """Generate a shooting script from a video analysis result.

    Args:
        analysis: Output of :func:`analyze_video`.
        script_type: One of ``"documentary"``, ``"promotional"``,
            ``"tutorial"``, ``"social_media"``, ``"narrative"``
            or a :class:`ScriptType` enum value.
        target_duration: Desired final video duration in seconds.
            When ``0.0``, the LLM infers duration from source material.
        target_audience: Who the video is for (e.g. ``"developers"``).
        style: Visual/editorial style guidance (e.g. ``"minimalist"``).
        emphasis: Optional user note to steer the LLM
            (e.g. ``"focus on the cooking scenes"``).
        lang: Output language code (``"en"`` or ``"zh"``).
        settings: Explicit settings.  Defaults to :func:`get_settings`.

    Returns:
        A :class:`ScriptGenerationResult` with the script, usage, and
        any errors.
    """
    if settings is None:
        settings = get_settings()

    if isinstance(script_type, str):
        script_type = ScriptType(script_type)

    model_name = settings.model_name
    if ":" in model_name:
        model_name = model_name.split(":", 1)[1]

    system_prompt = _localized_prompt(_build_system_prompt(script_type), lang)
    user_prompt = _build_user_prompt(
        analysis, script_type, target_duration,
        target_audience, style, emphasis,
    )

    agent = create_structured_agent(
        ShootingScript,
        settings=settings,
        system_prompt=system_prompt,
    )

    try:
        start = time.monotonic()
        result = await agent.run(user_prompt)
        duration = time.monotonic() - start

        usage = result.usage()
        token_rec = TokenUsage(
            model_name=model_name,
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            duration_seconds=round(duration, 3),
            step_name="script_generation",
        )
        logger.info(
            "Script generated in %.1fs (%d tokens)",
            duration, usage.total_tokens,
        )
        return ScriptGenerationResult(
            script=result.output,
            token_usage=token_rec,
        )

    except Exception as exc:
        logger.warning("Script generation failed: %s", exc)
        error = PipelineError(
            category=ErrorCategory.SCRIPT_GENERATION,
            message=f"Script generation failed: {exc}",
            detail=type(exc).__name__,
        )
        return ScriptGenerationResult(error=error)
