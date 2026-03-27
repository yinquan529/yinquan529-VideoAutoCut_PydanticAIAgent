"""Integration tests using PydanticAI TestModel and end-to-end pipeline tests.

Group A: Tests the orchestrator agent with ``pydantic_ai.models.test.TestModel``
         to verify tool registration and structured output without real LLM calls.

Group B: Structured-output validation — ensures Pydantic models reject invalid
         data and honour frozen constraints.

Group C: End-to-end happy-path tests that exercise the full CLI pipeline
         with all external dependencies mocked.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError
from pydantic_ai.models.test import TestModel
from typer.testing import CliRunner

from helpers import make_settings
from video_autocut.agent.hunyuan_client import create_structured_agent
from video_autocut.agent.orchestrator import (
    TOOLS,
    VideoDeps,
    create_orchestrator,
)
from video_autocut.app.cli import app
from video_autocut.domain.enums import (
    ErrorCategory,
    SceneType,
    ScriptType,
    ShotType,
)
from video_autocut.domain.models import (
    PipelineError,
    PipelineRunStats,
    TokenUsage,
)
from video_autocut.domain.results import ScriptGenerationResult, VideoAnalysisResult
from video_autocut.domain.script_models import (
    FrameAnalysis,
    SceneDefinition,
    ShootingScript,
    ShotDefinition,
    VideoContentSummary,
)
from video_autocut.infrastructure.ffmpeg import FFmpegTools

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_deps():
    return VideoDeps(
        settings=make_settings(),
        ffmpeg=MagicMock(spec=FFmpegTools),
    )


# ---------------------------------------------------------------------------
# Group A: TestModel-based orchestrator tests
# ---------------------------------------------------------------------------


class TestOrchestratorWithTestModel:
    """Verify the orchestrator agent works end-to-end with PydanticAI's TestModel."""

    def _agent(self, **model_kwargs):
        settings = make_settings()
        agent = create_orchestrator(settings)
        agent.model = TestModel(**model_kwargs)
        return agent

    async def test_custom_output_text(self):
        """TestModel(custom_output_text=...) should return the fixed text."""
        agent = self._agent(custom_output_text="Hello from TestModel")
        result = await agent.run("Hi", deps=_make_deps())
        assert result.output == "Hello from TestModel"

    async def test_selective_tools(self):
        """TestModel(call_tools=[...]) should only call the specified tool."""
        from video_autocut.infrastructure.exceptions import FFmpegError

        agent = self._agent(
            call_tools=["get_video_info"],
            custom_output_text="done",
        )
        deps = _make_deps()
        # Raise FFmpegError (which the tool catches and converts to a message)
        deps.ffmpeg.probe_video.side_effect = FFmpegError("no such file")
        result = await agent.run("Check video.mp4", deps=deps)
        # Tool was called (probe_video was invoked), and agent returned text.
        assert result.output == "done"
        deps.ffmpeg.probe_video.assert_called_once()

    async def test_no_tools_mode(self):
        """TestModel(call_tools=[]) should skip tools entirely."""
        agent = self._agent(call_tools=[], custom_output_text="no tools")
        result = await agent.run("Hi", deps=_make_deps())
        assert result.output == "no tools"

    def test_tool_docstrings_present(self):
        """Every registered tool function must have a non-empty docstring."""
        for tool_fn in TOOLS:
            assert tool_fn.__doc__, f"{tool_fn.__name__} has no docstring"

    def test_retries_match_settings(self):
        settings = make_settings(max_retries=3)
        agent = create_orchestrator(settings)
        assert agent._max_result_retries == 3

    def test_tool_count(self):
        settings = make_settings()
        agent = create_orchestrator(settings)
        tool_names = list(agent._function_toolset.tools.keys())
        assert len(tool_names) == 5
        expected = {
            "get_video_info",
            "extract_video_frames",
            "analyze_video_content",
            "generate_video_script",
            "analyze_images",
        }
        assert set(tool_names) == expected


class TestStructuredAgentWithTestModel:
    """Verify structured-output agents work with TestModel via PromptedOutput (text-based).

    Because ``create_structured_agent`` uses ``PromptedOutput`` (to avoid
    ``tool_choice`` which Hunyuan rejects), the model returns JSON as plain
    text.  We use ``custom_output_text`` with JSON strings rather than
    ``custom_output_args``.
    """

    async def test_frame_analysis_output(self):
        text = json.dumps({
            "frame_path": "frame_0000.jpg",
            "timestamp_seconds": 1.5,
            "description": "A sunset over the ocean.",
            "detected_objects": ["sun", "ocean"],
            "scene_type": "exterior",
            "visual_mood": "warm",
            "dominant_colors": ["orange", "blue"],
        })
        settings = make_settings()
        agent = create_structured_agent(
            FrameAnalysis,
            settings,
            system_prompt="Analyze this frame.",
        )
        agent.model = TestModel(custom_output_text=text)
        result = await agent.run("Describe the image.")
        assert isinstance(result.output, FrameAnalysis)
        assert result.output.description == "A sunset over the ocean."
        assert result.output.scene_type == SceneType.EXTERIOR

    async def test_shooting_script_output(self):
        text = json.dumps({
            "title": "Test Script",
            "script_type": "documentary",
            "target_duration_seconds": 60.0,
            "synopsis": "A brief test.",
            "scenes": [],
            "narration_cues": [],
            "music_cues": [],
            "production_notes": "none",
        })
        settings = make_settings()
        agent = create_structured_agent(
            ShootingScript,
            settings,
            system_prompt="Generate a script.",
        )
        agent.model = TestModel(custom_output_text=text)
        result = await agent.run("Generate.")
        assert isinstance(result.output, ShootingScript)
        assert result.output.title == "Test Script"
        assert result.output.script_type == ScriptType.DOCUMENTARY
        assert result.output.scenes == []

    async def test_video_content_summary_output(self):
        text = json.dumps({
            "overall_summary": "Cityscape footage.",
            "themes": ["urban"],
            "visual_style": "handheld",
            "pacing": "fast",
            "key_moments": [],
            "estimated_tone": "energetic",
        })
        settings = make_settings()
        agent = create_structured_agent(
            VideoContentSummary,
            settings,
            system_prompt="Summarize.",
        )
        agent.model = TestModel(custom_output_text=text)
        result = await agent.run("Summarize.")
        assert isinstance(result.output, VideoContentSummary)
        assert result.output.overall_summary == "Cityscape footage."


# ---------------------------------------------------------------------------
# Group B: Structured output validation
# ---------------------------------------------------------------------------


class TestStructuredOutputValidation:
    """Ensure Pydantic models correctly reject invalid data."""

    def test_frame_analysis_missing_required_field(self):
        with pytest.raises(ValidationError):
            FrameAnalysis(
                frame_path="test.jpg",
                timestamp_seconds=0.0,
                # missing: description, scene_type, visual_mood
            )

    def test_shooting_script_empty_scenes_allowed(self):
        script = ShootingScript(
            title="Empty",
            script_type=ScriptType.DOCUMENTARY,
            target_duration_seconds=10.0,
            synopsis="Nothing.",
            scenes=[],
        )
        assert script.scenes == []

    def test_shooting_script_invalid_shot_type(self):
        with pytest.raises(ValidationError):
            ShotDefinition(
                shot_number=1,
                shot_type="NONEXISTENT",
                start_seconds=0.0,
                end_seconds=5.0,
                description="test",
            )

    def test_video_content_summary_frozen(self):
        summary = VideoContentSummary(
            overall_summary="Test.",
            themes=["test"],
            visual_style="flat",
            pacing="slow",
            key_moments=[],
            estimated_tone="neutral",
        )
        with pytest.raises(ValidationError):
            summary.overall_summary = "Modified."

    def test_pipeline_error_defaults(self):
        err = PipelineError(
            category=ErrorCategory.FRAME_EXTRACTION,
            message="something failed",
        )
        assert err.detail == ""
        assert err.timestamp_seconds is None
        assert err.frame_index is None


# ---------------------------------------------------------------------------
# Group C: End-to-end integration tests
# ---------------------------------------------------------------------------


SAMPLE_FRAME = FrameAnalysis(
    frame_path="frame_0000.jpg",
    timestamp_seconds=0.0,
    description="A mountain landscape.",
    detected_objects=["mountain"],
    scene_type=SceneType.EXTERIOR,
    visual_mood="serene",
    dominant_colors=["blue"],
)

SAMPLE_SUMMARY = VideoContentSummary(
    overall_summary="Mountains and nature.",
    themes=["nature"],
    visual_style="cinematic",
    pacing="slow",
    key_moments=[],
    estimated_tone="contemplative",
)

SAMPLE_STATS = PipelineRunStats(
    total_duration_seconds=2.0,
    llm_calls=[],
    total_prompt_tokens=100,
    total_completion_tokens=200,
    frames_extracted=3,
    frames_analyzed=3,
)

SAMPLE_ANALYSIS = VideoAnalysisResult(
    video_name="integration_test.mp4",
    frame_analyses=[SAMPLE_FRAME],
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
    title="Integration Test Script",
    script_type=ScriptType.DOCUMENTARY,
    target_duration_seconds=120.0,
    synopsis="A visual journey.",
    scenes=[SAMPLE_SCENE],
)

SAMPLE_TOKEN = TokenUsage(
    model_name="test",
    prompt_tokens=50,
    completion_tokens=100,
    total_tokens=150,
    duration_seconds=1.0,
    step_name="script_generation",
)

SAMPLE_GEN_RESULT = ScriptGenerationResult(
    script=SAMPLE_SCRIPT,
    token_usage=SAMPLE_TOKEN,
)


class TestE2EGenerate:
    """End-to-end tests for the CLI ``generate`` command."""

    def _make_video(self, tmp_path: Path) -> Path:
        video = tmp_path / "integration_test.mp4"
        video.write_bytes(b"\x00" * 64)
        return video

    def test_happy_path_markdown(self, tmp_path: Path):
        video = self._make_video(tmp_path)
        with (
            patch("video_autocut.app.cli.validate_settings", return_value=make_settings()),
            patch("video_autocut.app.cli.analyze_video", return_value=SAMPLE_ANALYSIS),
            patch("video_autocut.app.cli.generate_script", return_value=SAMPLE_GEN_RESULT),
        ):
            result = runner.invoke(app, [
                "generate", str(video),
                "--output-dir", str(tmp_path),
                "--format", "md",
            ])
        assert result.exit_code == 0, result.output
        # An output file should have been created
        md_files = list(tmp_path.glob("integration_test_documentary_*.md"))
        assert len(md_files) == 1
        content = md_files[0].read_text()
        assert "Integration Test Script" in content

    def test_happy_path_json(self, tmp_path: Path):
        video = self._make_video(tmp_path)
        with (
            patch("video_autocut.app.cli.validate_settings", return_value=make_settings()),
            patch("video_autocut.app.cli.analyze_video", return_value=SAMPLE_ANALYSIS),
            patch("video_autocut.app.cli.generate_script", return_value=SAMPLE_GEN_RESULT),
        ):
            result = runner.invoke(app, [
                "generate", str(video),
                "--output-dir", str(tmp_path),
                "--format", "json",
            ])
        assert result.exit_code == 0, result.output
        json_files = list(tmp_path.glob("integration_test_documentary_*.json"))
        assert len(json_files) == 1
        data = json.loads(json_files[0].read_text())
        assert data["title"] == "Integration Test Script"

    def test_happy_path_txt(self, tmp_path: Path):
        video = self._make_video(tmp_path)
        with (
            patch("video_autocut.app.cli.validate_settings", return_value=make_settings()),
            patch("video_autocut.app.cli.analyze_video", return_value=SAMPLE_ANALYSIS),
            patch("video_autocut.app.cli.generate_script", return_value=SAMPLE_GEN_RESULT),
        ):
            result = runner.invoke(app, [
                "generate", str(video),
                "--output-dir", str(tmp_path),
                "--format", "txt",
            ])
        assert result.exit_code == 0, result.output
        txt_files = list(tmp_path.glob("integration_test_documentary_*.txt"))
        assert len(txt_files) == 1


class TestE2EChat:
    """End-to-end test for the CLI ``chat`` command."""

    def test_chat_happy_path(self):
        """CLI chat command invokes the orchestrator and prints the response."""
        from unittest.mock import AsyncMock

        mock_result = MagicMock()
        mock_result.output = "I can help you with video analysis."

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)

        with (
            patch("video_autocut.app.cli.validate_settings", return_value=make_settings()),
            patch("video_autocut.app.cli.create_orchestrator", return_value=mock_agent),
        ):
            result = runner.invoke(app, ["chat", "Hello"])

        assert result.exit_code == 0, result.output
        assert "video analysis" in result.output
