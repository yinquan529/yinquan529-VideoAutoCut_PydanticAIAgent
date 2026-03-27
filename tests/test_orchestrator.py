"""Tests for the main orchestrator agent.

Mocking strategy:
- Tool functions are tested by calling them directly with a mock
  ``RunContext[VideoDeps]``.
- Underlying pipeline functions (``analyze_video``, ``generate_script``,
  ``extract_frames``) are patched to return pre-built results.
- ``create_model`` is patched so no real LLM connection is needed.
- No real ffmpeg, LLM, or network calls are made.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from video_autocut.domain.enums import (
    ErrorCategory,
    ExtractionStrategy,
    SceneType,
    ScriptType,
)
from video_autocut.domain.models import (
    ExtractedFrame,
    FrameExtractionResult,
    PipelineError,
    PipelineRunStats,
    TokenUsage,
    VideoMetadata,
)
from video_autocut.domain.results import ScriptGenerationResult, VideoAnalysisResult
from video_autocut.domain.script_models import (
    FrameAnalysis,
    SceneDefinition,
    ShootingScript,
    ShotDefinition,
    ShotType,
    VideoContentSummary,
)
from video_autocut.infrastructure.exceptions import VideoProbeError
from video_autocut.infrastructure.ffmpeg import FFmpegTools
from video_autocut.settings import Settings, get_settings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(**overrides) -> Settings:
    defaults = dict(
        hunyuan_api_key="test-key",
        hunyuan_base_url="http://localhost",
        model_name="openai:test-model",
        ffmpeg_path=Path("ffmpeg"),
        ffprobe_path=Path("ffprobe"),
        temp_frames_dir=Path("/tmp/test_frames"),
        output_dir=Path("/tmp/test_output"),
        max_retries=0,
        request_timeout_seconds=10,
    )
    defaults.update(overrides)
    return Settings(**defaults)


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _make_deps(settings: Settings | None = None):
    from video_autocut.agent.orchestrator import VideoDeps

    s = settings or _make_settings()
    return VideoDeps(settings=s, ffmpeg=MagicMock(spec=FFmpegTools))


def _make_ctx(deps=None):
    """Build a minimal RunContext-like object for direct tool calls."""
    ctx = MagicMock()
    ctx.deps = deps or _make_deps()
    return ctx


# ---------------------------------------------------------------------------
# Sample domain objects
# ---------------------------------------------------------------------------

SAMPLE_METADATA = VideoMetadata(
    path=Path("/videos/sample.mp4"),
    name="sample.mp4",
    duration_seconds=120.0,
    width=1920,
    height=1080,
    fps=30.0,
    codec="h264",
    format_name="mov,mp4,m4a,3gp,3g2,mj2",
    file_size_bytes=50_000_000,
)

SAMPLE_FRAME_ANALYSIS = FrameAnalysis(
    frame_path="frame_0000.jpg",
    timestamp_seconds=0.0,
    description="A wide shot of a mountain landscape.",
    detected_objects=["mountain", "sky", "trees"],
    scene_type=SceneType.EXTERIOR,
    visual_mood="serene",
    dominant_colors=["blue", "green", "white"],
)

SAMPLE_SUMMARY = VideoContentSummary(
    overall_summary="A nature documentary about mountains.",
    themes=["nature", "landscape"],
    visual_style="cinematic",
    pacing="slow",
    key_moments=[],
    estimated_tone="contemplative",
)

SAMPLE_STATS = PipelineRunStats(
    total_duration_seconds=5.0,
    llm_calls=[],
    total_prompt_tokens=100,
    total_completion_tokens=200,
    frames_extracted=3,
    frames_analyzed=3,
)

SAMPLE_ANALYSIS = VideoAnalysisResult(
    video_name="sample.mp4",
    frame_analyses=[SAMPLE_FRAME_ANALYSIS],
    content_summary=SAMPLE_SUMMARY,
    stats=SAMPLE_STATS,
    errors=[],
)

SAMPLE_SHOT = ShotDefinition(
    shot_number=1,
    shot_type=ShotType.WIDE,
    start_seconds=0.0,
    end_seconds=10.0,
    description="Mountain panorama",
)

SAMPLE_SCENE = SceneDefinition(
    scene_number=1,
    title="Opening",
    location="Mountains",
    scene_type=SceneType.EXTERIOR,
    mood="serene",
    start_seconds=0.0,
    end_seconds=30.0,
    shots=[SAMPLE_SHOT],
)

SAMPLE_SCRIPT = ShootingScript(
    title="Mountain Documentary",
    script_type=ScriptType.DOCUMENTARY,
    target_duration_seconds=120.0,
    synopsis="A visual journey through mountain landscapes.",
    scenes=[SAMPLE_SCENE],
)


# ---------------------------------------------------------------------------
# Fake agent for analyze_images
# ---------------------------------------------------------------------------


@dataclass
class FakeUsage:
    input_tokens: int = 100
    output_tokens: int = 50
    total_tokens: int = 150


class FakeRunResult:
    def __init__(self, output):
        self.output = output

    def usage(self):
        return FakeUsage()


class FakeAgent:
    def __init__(self, output):
        self._output = output

    async def run(self, user_prompt, **kwargs):
        if isinstance(self._output, Exception):
            raise self._output
        return FakeRunResult(self._output)


# ===================================================================
# Test: VideoDeps
# ===================================================================


class TestVideoDeps:
    def test_construction(self):
        from video_autocut.agent.orchestrator import VideoDeps

        s = _make_settings()
        ffmpeg = FFmpegTools()
        deps = VideoDeps(settings=s, ffmpeg=ffmpeg)
        assert deps.settings is s
        assert deps.ffmpeg is ffmpeg


# ===================================================================
# Test: create_orchestrator
# ===================================================================


class TestCreateOrchestrator:
    def test_returns_agent(self):
        from pydantic_ai import Agent

        from video_autocut.agent.orchestrator import create_orchestrator

        settings = _make_settings()
        agent = create_orchestrator(settings)
        assert isinstance(agent, Agent)

    def test_default_settings(self, monkeypatch: pytest.MonkeyPatch):
        from video_autocut.agent.orchestrator import create_orchestrator

        settings = _make_settings()
        monkeypatch.setattr(
            "video_autocut.agent.orchestrator.get_settings",
            lambda: settings,
        )
        agent = create_orchestrator()
        # Agent was created with settings.max_retries
        assert agent._max_result_retries == settings.max_retries

    def test_has_five_tools(self):
        from video_autocut.agent.orchestrator import create_orchestrator

        settings = _make_settings()
        agent = create_orchestrator(settings)
        toolset = agent._function_toolset
        assert len(toolset.tools) == 5

    def test_system_prompt_contains_guardrail(self):
        from video_autocut.agent.orchestrator import SYSTEM_INSTRUCTIONS

        assert "MUST call this before" in SYSTEM_INSTRUCTIONS
        assert "NEVER describe video content" in SYSTEM_INSTRUCTIONS


# ===================================================================
# Test: get_video_info
# ===================================================================


class TestGetVideoInfo:
    async def test_happy_path(self):
        from video_autocut.agent.orchestrator import get_video_info

        ctx = _make_ctx()
        ctx.deps.ffmpeg.probe_video.return_value = SAMPLE_METADATA

        result = await get_video_info(ctx, "/videos/sample.mp4")

        ctx.deps.ffmpeg.probe_video.assert_called_once_with(
            Path("/videos/sample.mp4"),
        )
        parsed = json.loads(result)
        assert parsed["name"] == "sample.mp4"
        assert parsed["width"] == 1920

    async def test_probe_error(self):
        from video_autocut.agent.orchestrator import get_video_info

        ctx = _make_ctx()
        ctx.deps.ffmpeg.probe_video.side_effect = VideoProbeError(
            "ffprobe failed"
        )

        result = await get_video_info(ctx, "/bad/video.mp4")
        assert "Error probing video" in result


# ===================================================================
# Test: extract_video_frames
# ===================================================================


class TestExtractVideoFrames:
    def _patch_extract(self, frames=None, errors=None):
        result = FrameExtractionResult(
            video_name="test.mp4",
            strategy=ExtractionStrategy.UNIFORM,
            frames=frames or [],
            errors=errors or [],
        )
        return patch(
            "video_autocut.agent.orchestrator.extract_frames",
            return_value=result,
        )

    async def test_happy_path(self, tmp_path: Path):
        from video_autocut.agent.orchestrator import extract_video_frames

        frame = ExtractedFrame(
            path=tmp_path / "frame_0000.jpg",
            timestamp_seconds=0.0,
            frame_index=0,
        )
        with self._patch_extract(frames=[frame]):
            result = await extract_video_frames(
                _make_ctx(), "test.mp4",
            )

        assert "Extracted 1 frames" in result
        assert "uniform" in result

    async def test_with_strategy(self):
        from video_autocut.agent.orchestrator import extract_video_frames

        with self._patch_extract() as mock_ef:
            await extract_video_frames(
                _make_ctx(), "test.mp4",
                strategy="scene_change", max_frames=5,
            )
            mock_ef.assert_called_once_with(
                "test.mp4", strategy="scene_change", max_frames=5,
            )

    async def test_with_errors(self):
        from video_autocut.agent.orchestrator import extract_video_frames

        error = PipelineError(
            category=ErrorCategory.FRAME_EXTRACTION,
            message="Frame 3 failed",
        )
        with self._patch_extract(errors=[error]):
            result = await extract_video_frames(
                _make_ctx(), "test.mp4",
            )
        assert "Errors (1)" in result
        assert "Frame 3 failed" in result

    async def test_extraction_error(self):
        from video_autocut.agent.orchestrator import extract_video_frames

        with patch(
            "video_autocut.agent.orchestrator.extract_frames",
            side_effect=RuntimeError("ffmpeg crashed"),
        ):
            result = await extract_video_frames(
                _make_ctx(), "test.mp4",
            )
        assert "Frame extraction failed" in result


# ===================================================================
# Test: analyze_video_content
# ===================================================================


class TestAnalyzeVideoContent:
    def _patch_analyze(self, result=None):
        return patch(
            "video_autocut.agent.orchestrator.analyze_video",
            return_value=result or SAMPLE_ANALYSIS,
        )

    async def test_happy_path(self):
        from video_autocut.agent.orchestrator import analyze_video_content

        with self._patch_analyze():
            result = await analyze_video_content(
                _make_ctx(), "sample.mp4",
            )

        assert "Analysis of sample.mp4" in result
        assert "nature documentary" in result.lower()
        assert "Token usage:" in result

    async def test_with_key_moments(self):
        from video_autocut.agent.orchestrator import analyze_video_content
        from video_autocut.domain.script_models import KeyMoment

        summary_with_moments = VideoContentSummary(
            overall_summary="A test video.",
            themes=["test"],
            visual_style="flat",
            pacing="fast",
            key_moments=[
                KeyMoment(timestamp_seconds=5.0, description="Opening"),
                KeyMoment(timestamp_seconds=30.0, description="Climax"),
            ],
            estimated_tone="neutral",
        )
        analysis = VideoAnalysisResult(
            video_name="moments.mp4",
            frame_analyses=[SAMPLE_FRAME_ANALYSIS],
            content_summary=summary_with_moments,
            stats=SAMPLE_STATS,
            errors=[],
        )
        with self._patch_analyze(analysis):
            result = await analyze_video_content(
                _make_ctx(), "moments.mp4",
            )
        assert "Key moments:" in result
        assert "5.0s" in result
        assert "Climax" in result

    async def test_with_focus(self):
        from video_autocut.agent.orchestrator import analyze_video_content

        with self._patch_analyze() as mock_av:
            await analyze_video_content(
                _make_ctx(), "sample.mp4", focus="outdoor scenes",
            )
            mock_av.assert_called_once()
            _, kwargs = mock_av.call_args
            assert kwargs["user_prompt"] == "outdoor scenes"

    async def test_error_returns_message(self):
        from video_autocut.agent.orchestrator import analyze_video_content

        with patch(
            "video_autocut.agent.orchestrator.analyze_video",
            side_effect=RuntimeError("LLM down"),
        ):
            result = await analyze_video_content(
                _make_ctx(), "sample.mp4",
            )
        assert "Video analysis failed" in result

    async def test_analysis_with_errors(self):
        from video_autocut.agent.orchestrator import analyze_video_content

        analysis = VideoAnalysisResult(
            video_name="err.mp4",
            frame_analyses=[SAMPLE_FRAME_ANALYSIS],
            content_summary=SAMPLE_SUMMARY,
            stats=SAMPLE_STATS,
            errors=[
                PipelineError(
                    category=ErrorCategory.FRAME_ANALYSIS,
                    message="Frame 2 failed",
                ),
            ],
        )
        with self._patch_analyze(analysis):
            result = await analyze_video_content(
                _make_ctx(), "err.mp4",
            )
        assert "Errors (1)" in result
        assert "Frame 2 failed" in result

    async def test_no_summary(self):
        from video_autocut.agent.orchestrator import analyze_video_content

        analysis = VideoAnalysisResult(
            video_name="nosummary.mp4",
            frame_analyses=[],
            content_summary=None,
            stats=SAMPLE_STATS,
            errors=[],
        )
        with self._patch_analyze(analysis):
            result = await analyze_video_content(
                _make_ctx(), "nosummary.mp4",
            )
        assert "Summary:" not in result
        assert "Frames analyzed: " in result


# ===================================================================
# Test: generate_video_script
# ===================================================================


class TestGenerateVideoScript:
    def _patches(
        self,
        analysis=None,
        gen_result=None,
        analysis_exc=None,
        gen_exc=None,
    ):
        """Return a context manager that patches both analyze_video and generate_script."""
        from contextlib import ExitStack

        stack = ExitStack()

        if analysis_exc:
            stack.enter_context(patch(
                "video_autocut.agent.orchestrator.analyze_video",
                side_effect=analysis_exc,
            ))
        else:
            stack.enter_context(patch(
                "video_autocut.agent.orchestrator.analyze_video",
                return_value=analysis or SAMPLE_ANALYSIS,
            ))

        if gen_exc:
            stack.enter_context(patch(
                "video_autocut.agent.orchestrator.generate_script",
                side_effect=gen_exc,
            ))
        else:
            stack.enter_context(patch(
                "video_autocut.agent.orchestrator.generate_script",
                return_value=gen_result or ScriptGenerationResult(
                    script=SAMPLE_SCRIPT,
                    token_usage=TokenUsage(
                        model_name="test",
                        prompt_tokens=100,
                        completion_tokens=200,
                        total_tokens=300,
                        duration_seconds=1.0,
                        step_name="script_generation",
                    ),
                ),
            ))

        return stack

    async def test_happy_path(self):
        from video_autocut.agent.orchestrator import generate_video_script

        with self._patches():
            result = await generate_video_script(
                _make_ctx(), "sample.mp4",
            )

        assert "Mountain Documentary" in result
        assert "Opening" in result

    async def test_script_type_forwarded(self):
        from video_autocut.agent.orchestrator import generate_video_script

        with self._patches():
            await generate_video_script(
                _make_ctx(), "sample.mp4",
                script_type="promotional",
                target_audience="developers",
            )

    async def test_analysis_failure(self):
        from video_autocut.agent.orchestrator import generate_video_script

        with self._patches(analysis_exc=RuntimeError("boom")):
            result = await generate_video_script(
                _make_ctx(), "sample.mp4",
            )
        assert "Video analysis failed" in result

    async def test_no_frames_analyzed(self):
        from video_autocut.agent.orchestrator import generate_video_script

        empty_analysis = VideoAnalysisResult(
            video_name="empty.mp4",
            frame_analyses=[],
            content_summary=None,
            stats=SAMPLE_STATS,
            errors=[
                PipelineError(
                    category=ErrorCategory.FRAME_EXTRACTION,
                    message="No frames extracted",
                ),
            ],
        )
        with self._patches(analysis=empty_analysis):
            result = await generate_video_script(
                _make_ctx(), "empty.mp4",
            )
        assert "No frames could be analyzed" in result

    async def test_generation_failure(self):
        from video_autocut.agent.orchestrator import generate_video_script

        with self._patches(gen_exc=RuntimeError("LLM timeout")):
            result = await generate_video_script(
                _make_ctx(), "sample.mp4",
            )
        assert "Script generation failed" in result

    async def test_generation_error_in_result(self):
        from video_autocut.agent.orchestrator import generate_video_script

        err_result = ScriptGenerationResult(
            error=PipelineError(
                category=ErrorCategory.SCRIPT_GENERATION,
                message="Invalid schema",
            ),
        )
        with self._patches(gen_result=err_result):
            result = await generate_video_script(
                _make_ctx(), "sample.mp4",
            )
        assert "Script generation error" in result
        assert "Invalid schema" in result

    async def test_no_script_in_result(self):
        from video_autocut.agent.orchestrator import generate_video_script

        empty_result = ScriptGenerationResult(script=None)
        with self._patches(gen_result=empty_result):
            result = await generate_video_script(
                _make_ctx(), "sample.mp4",
            )
        assert "no output" in result.lower()


# ===================================================================
# Test: analyze_images
# ===================================================================


class TestAnalyzeImages:
    async def test_happy_path(self, tmp_path: Path):
        from video_autocut.agent.orchestrator import analyze_images

        img = tmp_path / "photo.jpg"
        img.write_bytes(b"\xff\xd8" + b"\x00" * 100)

        with patch(
            "video_autocut.agent.orchestrator.create_structured_agent",
            return_value=FakeAgent(SAMPLE_FRAME_ANALYSIS),
        ):
            result = await analyze_images(
                _make_ctx(), [str(img)],
            )

        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["description"] == SAMPLE_FRAME_ANALYSIS.description
        assert parsed[0]["scene_type"] == "exterior"

    async def test_missing_file(self):
        from video_autocut.agent.orchestrator import analyze_images

        result = await analyze_images(
            _make_ctx(), ["/nonexistent/photo.jpg"],
        )

        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["error"] == "File not found"

    async def test_empty_list(self):
        from video_autocut.agent.orchestrator import analyze_images

        result = await analyze_images(_make_ctx(), [])
        assert "No image paths" in result

    async def test_partial_failure(self, tmp_path: Path):
        from video_autocut.agent.orchestrator import analyze_images

        img = tmp_path / "ok.jpg"
        img.write_bytes(b"\xff\xd8" + b"\x00" * 100)

        call_count = {"n": 0}

        class PartialAgent:
            async def run(self, user_prompt, **kwargs):
                call_count["n"] += 1
                if call_count["n"] == 1:
                    return FakeRunResult(SAMPLE_FRAME_ANALYSIS)
                raise RuntimeError("LLM failed")

        with patch(
            "video_autocut.agent.orchestrator.create_structured_agent",
            return_value=PartialAgent(),
        ):
            img2 = tmp_path / "bad.jpg"
            img2.write_bytes(b"\xff\xd8" + b"\x00" * 100)
            result = await analyze_images(
                _make_ctx(), [str(img), str(img2)],
            )

        parsed = json.loads(result)
        assert len(parsed) == 2
        assert "description" in parsed[0]
        assert "error" in parsed[1]

    async def test_custom_prompt(self, tmp_path: Path):
        from video_autocut.agent.orchestrator import analyze_images

        img = tmp_path / "photo.jpg"
        img.write_bytes(b"\xff\xd8" + b"\x00" * 100)

        captured_prompt = {}

        class CapturingAgent:
            async def run(self, user_prompt, **kwargs):
                captured_prompt["value"] = user_prompt[0]
                return FakeRunResult(SAMPLE_FRAME_ANALYSIS)

        with patch(
            "video_autocut.agent.orchestrator.create_structured_agent",
            return_value=CapturingAgent(),
        ):
            await analyze_images(
                _make_ctx(), [str(img)],
                prompt="Focus on the background.",
            )

        assert captured_prompt["value"] == "Focus on the background."


# ===================================================================
# Test: run_agent
# ===================================================================


class TestRunAgent:
    async def test_convenience_runner(self, monkeypatch: pytest.MonkeyPatch):
        from video_autocut.agent.orchestrator import run_agent

        settings = _make_settings()
        monkeypatch.setattr(
            "video_autocut.agent.orchestrator.get_settings",
            lambda: settings,
        )

        fake_result = MagicMock()
        fake_result.output = "Hello! I can help with video analysis."

        fake_agent = MagicMock()
        fake_agent.run = MagicMock(return_value=fake_result)

        async def fake_run(*args, **kwargs):
            return fake_result

        fake_agent.run = fake_run

        with patch(
            "video_autocut.agent.orchestrator.create_orchestrator",
            return_value=fake_agent,
        ):
            result = await run_agent("Hello!")

        assert result == "Hello! I can help with video analysis."


# ===================================================================
# Test: module exports
# ===================================================================


class TestExports:
    def test_orchestrator_module_exports(self):
        from video_autocut.agent.orchestrator import (
            VideoDeps,  # noqa: F401
            create_orchestrator,
            run_agent,
        )

        assert callable(create_orchestrator)
        assert callable(run_agent)

    def test_tools_list(self):
        from video_autocut.agent.orchestrator import TOOLS

        assert len(TOOLS) == 5
        names = [t.__name__ for t in TOOLS]
        assert "get_video_info" in names
        assert "extract_video_frames" in names
        assert "analyze_video_content" in names
        assert "generate_video_script" in names
        assert "analyze_images" in names
