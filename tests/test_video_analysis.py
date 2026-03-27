"""Tests for the video analysis pipeline.

Mocking strategy:
- ``extract_frames`` is patched to return pre-built ``FrameExtractionResult``
  with real frame files on disk (created via ``tmp_path``).
- ``create_structured_agent`` is patched to return a fake agent whose
  ``run()`` method returns pre-built domain objects with token usage.
- No real ffmpeg or LLM calls are made.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from helpers import make_settings
from video_autocut.domain.enums import (
    ErrorCategory,
    ExtractionStrategy,
    SceneType,
)
from video_autocut.domain.models import (
    ExtractedFrame,
    FrameExtractionResult,
    PipelineError,
)
from video_autocut.domain.results import VideoAnalysisResult
from video_autocut.domain.script_models import (
    FrameAnalysis,
    KeyMoment,
    VideoContentSummary,
)
from video_autocut.tools.video_analysis import (
    FRAME_ANALYSIS_PROMPT,
    SYNTHESIS_PROMPT,
    analyze_video,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_frame_files(tmp_path: Path, count: int) -> list[ExtractedFrame]:
    """Create real frame files and return ExtractedFrame objects."""
    frames = []
    out_dir = tmp_path / "analysis_frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        path = out_dir / f"frame_{i:04d}.jpg"
        # Distinct file content per frame (avoids dedup issues in other tests)
        path.write_bytes(b"\xff\xd8" + b"\x00" * (2000 + i * 100))
        frames.append(ExtractedFrame(
            path=path,
            timestamp_seconds=float(i * 5),
            frame_index=i,
        ))
    return frames


SAMPLE_FRAME_ANALYSIS = FrameAnalysis(
    frame_path="frame_0000.jpg",
    timestamp_seconds=0.0,
    description="A wide shot of a city skyline at sunset.",
    detected_objects=["buildings", "sky", "sun"],
    scene_type=SceneType.EXTERIOR,
    visual_mood="warm and contemplative",
    dominant_colors=["orange", "deep blue", "gold"],
)

SAMPLE_SUMMARY = VideoContentSummary(
    overall_summary="A cinematic tour of an urban skyline during golden hour.",
    themes=["urban landscape", "golden hour"],
    visual_style="Wide cinematic shots with warm color grading.",
    pacing="Slow and contemplative with gradual transitions.",
    key_moments=[
        KeyMoment(timestamp_seconds=0.0, description="Opening skyline shot"),
    ],
    estimated_tone="serene and reflective",
)


@dataclass
class FakeUsage:
    """Mimics pydantic_ai RunUsage for testing."""

    input_tokens: int = 100
    output_tokens: int = 50
    total_tokens: int = 150


class FakeRunResult:
    """Mimics pydantic_ai AgentRunResult for testing."""

    def __init__(self, output, usage=None):
        self.output = output
        self._usage = usage or FakeUsage()

    def usage(self):
        return self._usage


class FakeAgent:
    """A fake Agent whose run() returns predetermined results."""

    def __init__(self, output):
        self._output = output
        self._call_count = 0

    async def run(self, user_prompt, **kwargs):
        self._call_count += 1
        if isinstance(self._output, Exception):
            raise self._output
        if isinstance(self._output, list):
            # Return different outputs per call
            idx = min(self._call_count - 1, len(self._output) - 1)
            return FakeRunResult(self._output[idx])
        return FakeRunResult(self._output)


# ---------------------------------------------------------------------------
# TestAnalyzeVideo
# ---------------------------------------------------------------------------


class TestAnalyzeVideo:
    def _run(self, coro):
        """Run an async coroutine synchronously."""
        return asyncio.get_event_loop().run_until_complete(coro)

    def _patch_extract(self, frames, errors=None):
        """Return a patch for extract_frames that yields pre-built frames."""
        result = FrameExtractionResult(
            video_name="test.mp4",
            strategy=ExtractionStrategy.UNIFORM,
            frames=frames,
            errors=errors or [],
        )
        return patch(
            "video_autocut.tools.video_analysis.extract_frames",
            return_value=result,
        )

    def _patch_cleanup(self):
        return patch(
            "video_autocut.tools.video_analysis.safe_cleanup_frames",
            return_value=0,
        )

    def _patch_agent(self, frame_output, summary_output):
        """Patch create_structured_agent to return fake agents.

        First call returns a frame-analysis agent.
        Second call returns a synthesis agent.
        """
        agents = [
            FakeAgent(frame_output),
            FakeAgent(summary_output),
        ]
        call_count = {"n": 0}

        def fake_create(output_type, **kwargs):
            idx = min(call_count["n"], len(agents) - 1)
            call_count["n"] += 1
            return agents[idx]

        return patch(
            "video_autocut.tools.video_analysis.create_structured_agent",
            side_effect=fake_create,
        )

    async def test_basic_happy_path(self, tmp_path: Path):
        frames = _make_frame_files(tmp_path, 2)
        settings = make_settings()

        with (
            self._patch_extract(frames),
            self._patch_cleanup(),
            self._patch_agent(SAMPLE_FRAME_ANALYSIS, SAMPLE_SUMMARY),
        ):
            result = await analyze_video(
                tmp_path / "test.mp4",
                settings=settings,
            )

        assert isinstance(result, VideoAnalysisResult)
        assert result.video_name == "test.mp4"
        assert len(result.frame_analyses) == 2
        assert result.content_summary is not None
        assert result.content_summary.overall_summary == SAMPLE_SUMMARY.overall_summary
        assert len(result.errors) == 0
        assert result.stats.frames_extracted == 2
        assert result.stats.frames_analyzed == 2
        # 2 frame analyses + 1 synthesis = 3 LLM calls
        assert len(result.stats.llm_calls) == 3

    async def test_token_usage_tracked(self, tmp_path: Path):
        frames = _make_frame_files(tmp_path, 1)
        settings = make_settings()

        with (
            self._patch_extract(frames),
            self._patch_cleanup(),
            self._patch_agent(SAMPLE_FRAME_ANALYSIS, SAMPLE_SUMMARY),
        ):
            result = await analyze_video(
                tmp_path / "test.mp4",
                settings=settings,
            )

        assert result.stats.total_prompt_tokens == 200  # 100 + 100
        assert result.stats.total_completion_tokens == 100  # 50 + 50
        assert result.stats.llm_calls[0].step_name == "frame_analysis"
        assert result.stats.llm_calls[1].step_name == "content_synthesis"
        assert result.stats.total_duration_seconds >= 0

    async def test_frame_analysis_error_collected(self, tmp_path: Path):
        """One frame fails, the other succeeds — partial results returned."""
        frames = _make_frame_files(tmp_path, 2)
        settings = make_settings()

        # Frame agent: first call succeeds, second raises
        agents_created = []

        class PartialFailAgent:
            def __init__(self):
                self._call_count = 0

            async def run(self, user_prompt, **kwargs):
                self._call_count += 1
                if self._call_count == 2:
                    raise RuntimeError("LLM timeout")
                return FakeRunResult(SAMPLE_FRAME_ANALYSIS)

        def fake_create(output_type, **kwargs):
            if output_type is FrameAnalysis:
                a = PartialFailAgent()
                agents_created.append(a)
                return a
            return FakeAgent(SAMPLE_SUMMARY)

        with (
            self._patch_extract(frames),
            self._patch_cleanup(),
            patch(
                "video_autocut.tools.video_analysis.create_structured_agent",
                side_effect=fake_create,
            ),
        ):
            result = await analyze_video(
                tmp_path / "test.mp4",
                settings=settings,
            )

        assert len(result.frame_analyses) == 1
        assert len(result.errors) == 1
        assert result.errors[0].category == ErrorCategory.FRAME_ANALYSIS
        assert "Frame 1" in result.errors[0].message
        # Synthesis still runs with 1 successful frame
        assert result.content_summary is not None

    async def test_synthesis_error_collected(self, tmp_path: Path):
        """Synthesis fails — frame analyses still returned, summary is None."""
        frames = _make_frame_files(tmp_path, 1)
        settings = make_settings()

        def fake_create(output_type, **kwargs):
            if output_type is FrameAnalysis:
                return FakeAgent(SAMPLE_FRAME_ANALYSIS)
            return FakeAgent(RuntimeError("synthesis crashed"))

        with (
            self._patch_extract(frames),
            self._patch_cleanup(),
            patch(
                "video_autocut.tools.video_analysis.create_structured_agent",
                side_effect=fake_create,
            ),
        ):
            result = await analyze_video(
                tmp_path / "test.mp4",
                settings=settings,
            )

        assert len(result.frame_analyses) == 1
        assert result.content_summary is None
        assert any("synthesis failed" in e.message.lower() for e in result.errors)

    async def test_no_frames_extracted(self, tmp_path: Path):
        """No frames extracted — appropriate error, no LLM calls."""
        settings = make_settings()

        with (
            self._patch_extract(frames=[]),
            self._patch_cleanup(),
            self._patch_agent(SAMPLE_FRAME_ANALYSIS, SAMPLE_SUMMARY),
        ):
            result = await analyze_video(
                tmp_path / "test.mp4",
                settings=settings,
            )

        assert len(result.frame_analyses) == 0
        assert result.content_summary is None
        assert len(result.errors) == 1
        assert "No frames extracted" in result.errors[0].message
        assert result.stats.frames_extracted == 0
        assert len(result.stats.llm_calls) == 0

    async def test_cleanup_runs_on_success(self, tmp_path: Path):
        frames = _make_frame_files(tmp_path, 1)
        settings = make_settings()
        cleanup_mock = MagicMock(return_value=1)

        with (
            self._patch_extract(frames),
            patch(
                "video_autocut.tools.video_analysis.safe_cleanup_frames",
                cleanup_mock,
            ),
            self._patch_agent(SAMPLE_FRAME_ANALYSIS, SAMPLE_SUMMARY),
        ):
            await analyze_video(tmp_path / "test.mp4", settings=settings)

        cleanup_mock.assert_called_once()

    async def test_cleanup_runs_on_error(self, tmp_path: Path):
        """Cleanup still runs even if frame analysis raises unexpectedly."""
        frames = _make_frame_files(tmp_path, 1)
        settings = make_settings()
        cleanup_mock = MagicMock(return_value=1)

        def fake_create(output_type, **kwargs):
            # Both agents raise
            return FakeAgent(RuntimeError("total failure"))

        with (
            self._patch_extract(frames),
            patch(
                "video_autocut.tools.video_analysis.safe_cleanup_frames",
                cleanup_mock,
            ),
            patch(
                "video_autocut.tools.video_analysis.create_structured_agent",
                side_effect=fake_create,
            ),
        ):
            result = await analyze_video(
                tmp_path / "test.mp4",
                settings=settings,
            )

        # All frames failed + synthesis skipped, but cleanup still ran
        cleanup_mock.assert_called_once()
        assert len(result.frame_analyses) == 0

    async def test_strategy_passed_through(self, tmp_path: Path):
        frames = _make_frame_files(tmp_path, 1)
        settings = make_settings()
        extract_mock = MagicMock(return_value=FrameExtractionResult(
            video_name="test.mp4",
            strategy=ExtractionStrategy.SCENE_CHANGE,
            frames=frames,
            errors=[],
        ))

        with (
            patch(
                "video_autocut.tools.video_analysis.extract_frames",
                extract_mock,
            ),
            self._patch_cleanup(),
            self._patch_agent(SAMPLE_FRAME_ANALYSIS, SAMPLE_SUMMARY),
        ):
            await analyze_video(
                tmp_path / "test.mp4",
                strategy="scene_change",
                settings=settings,
            )

        extract_mock.assert_called_once()
        call_kwargs = extract_mock.call_args
        assert call_kwargs.kwargs.get("strategy") == "scene_change"

    async def test_custom_user_prompt(self, tmp_path: Path):
        """User prompt is passed through to frame analysis."""
        frames = _make_frame_files(tmp_path, 1)
        settings = make_settings()

        prompts_received = []

        class CapturingAgent:
            async def run(self, user_prompt, **kwargs):
                prompts_received.append(user_prompt)
                return FakeRunResult(SAMPLE_FRAME_ANALYSIS)

        class SummaryCapturingAgent:
            async def run(self, user_prompt, **kwargs):
                prompts_received.append(user_prompt)
                return FakeRunResult(SAMPLE_SUMMARY)

        call_n = {"n": 0}

        def fake_create(output_type, **kwargs):
            call_n["n"] += 1
            if output_type is FrameAnalysis:
                return CapturingAgent()
            return SummaryCapturingAgent()

        with (
            self._patch_extract(frames),
            self._patch_cleanup(),
            patch(
                "video_autocut.tools.video_analysis.create_structured_agent",
                side_effect=fake_create,
            ),
        ):
            await analyze_video(
                tmp_path / "test.mp4",
                user_prompt="focus on the outdoor scenes",
                settings=settings,
            )

        # Frame prompt is a list [str, BinaryContent]
        frame_prompt = prompts_received[0]
        assert isinstance(frame_prompt, list)
        assert "focus on the outdoor scenes" in frame_prompt[0]

        # Synthesis prompt is a plain string
        synth_prompt = prompts_received[1]
        assert "focus on the outdoor scenes" in synth_prompt

    async def test_settings_default(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """When settings=None, uses get_settings()."""
        settings = make_settings()
        monkeypatch.setattr(
            "video_autocut.tools.video_analysis.get_settings",
            lambda: settings,
        )

        with (
            self._patch_extract(frames=[]),
            self._patch_cleanup(),
        ):
            result = await analyze_video(tmp_path / "test.mp4")

        # Should succeed (no crash) and use default settings
        assert isinstance(result, VideoAnalysisResult)

    async def test_extraction_errors_propagated(self, tmp_path: Path):
        """Extraction errors from extract_frames are included in final result."""
        frames = _make_frame_files(tmp_path, 1)
        settings = make_settings()
        extraction_error = PipelineError(
            category=ErrorCategory.FRAME_EXTRACTION,
            message="Frame 3 failed to extract",
            frame_index=3,
        )

        with (
            self._patch_extract(frames, errors=[extraction_error]),
            self._patch_cleanup(),
            self._patch_agent(SAMPLE_FRAME_ANALYSIS, SAMPLE_SUMMARY),
        ):
            result = await analyze_video(
                tmp_path / "test.mp4",
                settings=settings,
            )

        # The extraction error should appear in the final errors list
        assert any(
            "Frame 3 failed to extract" in e.message for e in result.errors
        )


# ---------------------------------------------------------------------------
# TestPromptTemplates
# ---------------------------------------------------------------------------


class TestPromptTemplates:
    def test_frame_prompt_is_nonempty(self):
        assert len(FRAME_ANALYSIS_PROMPT) > 50
        assert "frame" in FRAME_ANALYSIS_PROMPT.lower()

    def test_synthesis_prompt_is_nonempty(self):
        assert len(SYNTHESIS_PROMPT) > 50
        assert "synthesize" in SYNTHESIS_PROMPT.lower()


# ---------------------------------------------------------------------------
# TestVideoAnalysisResult schema
# ---------------------------------------------------------------------------


class TestVideoAnalysisResultSchema:
    def test_json_schema(self):
        schema = VideoAnalysisResult.model_json_schema()
        assert "video_name" in schema["properties"]
        assert "frame_analyses" in schema["properties"]
        assert "content_summary" in schema["properties"]
        assert "stats" in schema["properties"]
        assert "errors" in schema["properties"]

    def test_frozen(self):
        from video_autocut.domain.models import PipelineRunStats

        result = VideoAnalysisResult(
            video_name="test.mp4",
            stats=PipelineRunStats(total_duration_seconds=1.0),
        )
        with pytest.raises(Exception):
            result.video_name = "changed"
