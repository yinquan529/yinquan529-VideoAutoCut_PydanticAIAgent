"""Tests for the CLI module.

Mocking strategy:
- ``analyze_video`` and ``generate_script`` are patched so no real LLM
  or ffmpeg calls are made.
- ``validate_settings`` is patched to return test settings without
  requiring environment variables.
- Typer's ``CliRunner`` captures output and exit codes.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

from video_autocut.app.cli import (
    OutputFormat,
    _build_output_path,
    _validate_video_path,
    app,
)
from video_autocut.domain.enums import ErrorCategory, SceneType, ScriptType, ShotType
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
from video_autocut.settings import Settings, get_settings

runner = CliRunner()


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


# ---------------------------------------------------------------------------
# Sample domain objects
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
    overall_summary="Mountains.",
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
    video_name="test.mp4",
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
    title="Mountain Documentary",
    script_type=ScriptType.DOCUMENTARY,
    target_duration_seconds=120.0,
    synopsis="A visual journey.",
    scenes=[SAMPLE_SCENE],
)

SAMPLE_TOKEN_USAGE = TokenUsage(
    model_name="test",
    prompt_tokens=50,
    completion_tokens=100,
    total_tokens=150,
    duration_seconds=1.0,
    step_name="script_generation",
)

SAMPLE_GEN_RESULT = ScriptGenerationResult(
    script=SAMPLE_SCRIPT,
    token_usage=SAMPLE_TOKEN_USAGE,
)


def _patch_settings():
    return patch(
        "video_autocut.app.cli.validate_settings",
        return_value=_make_settings(),
    )


def _patch_pipeline(
    analysis=None,
    gen_result=None,
    analysis_exc=None,
    gen_exc=None,
):
    """Return a context manager patching both analyze_video and generate_script."""
    from contextlib import ExitStack

    stack = ExitStack()

    stack.enter_context(_patch_settings())

    if analysis_exc:
        stack.enter_context(patch(
            "video_autocut.app.cli.analyze_video",
            side_effect=analysis_exc,
        ))
    else:
        stack.enter_context(patch(
            "video_autocut.app.cli.analyze_video",
            return_value=analysis or SAMPLE_ANALYSIS,
        ))

    if gen_exc:
        stack.enter_context(patch(
            "video_autocut.app.cli.generate_script",
            side_effect=gen_exc,
        ))
    else:
        stack.enter_context(patch(
            "video_autocut.app.cli.generate_script",
            return_value=gen_result or SAMPLE_GEN_RESULT,
        ))

    stack.enter_context(patch(
        "video_autocut.app.cli.render_script",
        return_value="# Mountain Documentary\n\nRendered markdown.",
    ))

    return stack


# ===================================================================
# Test: help output
# ===================================================================


class TestHelp:
    def test_main_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "generate" in result.output
        assert "chat" in result.output

    def test_generate_help(self):
        result = runner.invoke(app, ["generate", "--help"])
        assert result.exit_code == 0
        assert "--type" in result.output
        assert "--duration" in result.output
        assert "--audience" in result.output
        assert "--format" in result.output
        assert "--max-frames" in result.output

    def test_chat_help(self):
        result = runner.invoke(app, ["chat", "--help"])
        assert result.exit_code == 0
        assert "prompt" in result.output.lower()


# ===================================================================
# Test: _validate_video_path
# ===================================================================


class TestValidateVideoPath:
    def test_nonexistent_file(self):
        with pytest.raises(typer.Exit):
            _validate_video_path(Path("/nonexistent/video.mp4"))

    def test_unsupported_format(self, tmp_path: Path):
        bad = tmp_path / "video.xyz"
        bad.write_bytes(b"fake")
        with pytest.raises(typer.Exit):
            _validate_video_path(bad)

    def test_valid_file(self, tmp_path: Path):
        mp4 = tmp_path / "test.mp4"
        mp4.write_bytes(b"fake")
        result = _validate_video_path(mp4)
        assert result == mp4


# ===================================================================
# Test: _build_output_path
# ===================================================================


class TestBuildOutputPath:
    def test_pattern(self):
        path = _build_output_path(
            Path("video.mp4"), "documentary", Path("/out"), OutputFormat.md,
        )
        assert path.parent == Path("/out")
        assert path.name.startswith("video_documentary_")
        assert path.suffix == ".md"

    def test_json_extension(self):
        path = _build_output_path(
            Path("clip.mov"), "tutorial", Path("/out"), OutputFormat.json,
        )
        assert path.suffix == ".json"
        assert "clip_tutorial_" in path.name

    def test_txt_extension(self):
        path = _build_output_path(
            Path("a.avi"), "narrative", Path("/x"), OutputFormat.txt,
        )
        assert path.suffix == ".txt"


# ===================================================================
# Test: OutputFormat enum
# ===================================================================


class TestOutputFormat:
    def test_values(self):
        assert OutputFormat.md.value == "md"
        assert OutputFormat.json.value == "json"
        assert OutputFormat.txt.value == "txt"


# ===================================================================
# Test: generate command
# ===================================================================


class TestGenerateCommand:
    def test_missing_video_file(self):
        with _patch_settings():
            result = runner.invoke(app, ["generate", "/nonexistent/video.mp4"])
        assert result.exit_code == 1
        assert "not found" in result.output.lower() or result.exit_code == 1

    def test_unsupported_format(self, tmp_path: Path):
        bad = tmp_path / "video.xyz"
        bad.write_bytes(b"fake")
        with _patch_settings():
            result = runner.invoke(app, ["generate", str(bad)])
        assert result.exit_code == 1

    def test_happy_path(self, tmp_path: Path):
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake video")
        out_dir = tmp_path / "output"

        with _patch_pipeline():
            result = runner.invoke(app, [
                "generate", str(video),
                "--output-dir", str(out_dir),
                "--format", "md",
            ])

        assert result.exit_code == 0
        assert "Done!" in result.output
        assert "Mountain Documentary" in result.output

        # Verify output file was written
        files = list(out_dir.glob("*.md"))
        assert len(files) == 1
        assert "Rendered markdown" in files[0].read_text()

    def test_json_format(self, tmp_path: Path):
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake")
        out_dir = tmp_path / "output"

        # For JSON format, render_script is NOT called — model_dump_json is
        with _patch_pipeline():
            # Need to also patch the JSON path
            result = runner.invoke(app, [
                "generate", str(video),
                "--output-dir", str(out_dir),
                "--format", "json",
            ])

        assert result.exit_code == 0
        files = list(out_dir.glob("*.json"))
        assert len(files) == 1

    def test_with_options(self, tmp_path: Path):
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake")
        out_dir = tmp_path / "output"

        with _patch_pipeline():
            result = runner.invoke(app, [
                "generate", str(video),
                "--type", "promotional",
                "--duration", "60",
                "--audience", "teens",
                "--style", "energetic",
                "--prompt", "focus on action",
                "--output-dir", str(out_dir),
                "--max-frames", "5",
                "--strategy", "scene_change",
            ])

        assert result.exit_code == 0
        assert "Done!" in result.output

    def test_analysis_failure(self, tmp_path: Path):
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake")

        with _patch_pipeline(analysis_exc=RuntimeError("LLM down")):
            result = runner.invoke(app, ["generate", str(video)])

        assert result.exit_code == 1
        assert "Analysis failed" in result.output

    def test_no_frames_analyzed(self, tmp_path: Path):
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake")

        empty_analysis = VideoAnalysisResult(
            video_name="test.mp4",
            frame_analyses=[],
            content_summary=None,
            stats=SAMPLE_STATS,
            errors=[],
        )
        with _patch_pipeline(analysis=empty_analysis):
            result = runner.invoke(app, ["generate", str(video)])

        assert result.exit_code == 1
        assert "No frames" in result.output

    def test_generation_error(self, tmp_path: Path):
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake")

        err_result = ScriptGenerationResult(
            error=PipelineError(
                category=ErrorCategory.SCRIPT_GENERATION,
                message="Schema mismatch",
            ),
        )
        with _patch_pipeline(gen_result=err_result):
            result = runner.invoke(app, ["generate", str(video)])

        assert result.exit_code == 1
        assert "Schema mismatch" in result.output

    def test_generation_exception(self, tmp_path: Path):
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake")

        with _patch_pipeline(gen_exc=RuntimeError("timeout")):
            result = runner.invoke(app, ["generate", str(video)])

        assert result.exit_code == 1
        assert "Script generation failed" in result.output

    def test_generation_no_script(self, tmp_path: Path):
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake")

        empty_gen = ScriptGenerationResult(script=None)
        with _patch_pipeline(gen_result=empty_gen):
            result = runner.invoke(app, ["generate", str(video)])

        assert result.exit_code == 1
        assert "no output" in result.output.lower()

    def test_analysis_warnings_shown(self, tmp_path: Path):
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake")
        out_dir = tmp_path / "output"

        analysis_with_warnings = VideoAnalysisResult(
            video_name="test.mp4",
            frame_analyses=[SAMPLE_FRAME],
            content_summary=SAMPLE_SUMMARY,
            stats=SAMPLE_STATS,
            errors=[
                PipelineError(
                    category=ErrorCategory.FRAME_ANALYSIS,
                    message="Frame 2 was blurry",
                ),
            ],
        )
        with _patch_pipeline(analysis=analysis_with_warnings):
            result = runner.invoke(app, [
                "generate", str(video),
                "--output-dir", str(out_dir),
            ])

        assert result.exit_code == 0
        assert "Frame 2 was blurry" in result.output

    def test_duration_shown(self, tmp_path: Path):
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake")
        out_dir = tmp_path / "output"

        with _patch_pipeline():
            result = runner.invoke(app, [
                "generate", str(video),
                "--duration", "90",
                "--output-dir", str(out_dir),
            ])

        assert result.exit_code == 0
        assert "90s" in result.output


# ===================================================================
# Test: chat command
# ===================================================================


class TestChatCommand:
    def test_happy_path(self):
        with (
            _patch_settings(),
            patch(
                "video_autocut.app.cli.create_orchestrator",
            ) as mock_orch,
        ):
            fake_result = MagicMock()
            fake_result.output = "I can help with video analysis!"

            fake_agent = MagicMock()

            async def fake_run(*args, **kwargs):
                return fake_result

            fake_agent.run = fake_run
            mock_orch.return_value = fake_agent

            result = runner.invoke(app, ["chat", "What can you do?"])

        assert result.exit_code == 0
        assert "I can help with video analysis!" in result.output

    def test_missing_prompt(self):
        result = runner.invoke(app, ["chat"])
        assert result.exit_code != 0


# ===================================================================
# Test: settings validation error
# ===================================================================


class TestSettingsError:
    def test_missing_api_key(self, tmp_path: Path):
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake")

        from video_autocut.settings import SettingsValidationError

        with patch(
            "video_autocut.app.cli.validate_settings",
            side_effect=SettingsValidationError(["Missing HUNYUAN_API_KEY"]),
        ):
            result = runner.invoke(app, ["generate", str(video)])

        assert result.exit_code == 1
        assert "Configuration error" in result.output


# ===================================================================
# Test: main entry point
# ===================================================================


class TestMainEntryPoint:
    def test_main_callable(self):
        from video_autocut.app.main import main

        assert callable(main)

    def test_cli_main_callable(self):
        from video_autocut.app.cli import main

        assert callable(main)
