from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from video_autocut.domain import (
    ErrorCategory,
    ExtractedFrame,
    ExtractionStrategy,
    FrameAnalysis,
    FrameExtractionRequest,
    PipelineError,
    SceneDefinition,
    SceneType,
    ScriptType,
    ShootingScript,
    ShotDefinition,
    ShotType,
    TokenUsage,
    VideoContentSummary,
    VideoMetadata,
)

# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestEnums:
    def test_extraction_strategy_values(self):
        assert ExtractionStrategy.UNIFORM == "uniform"
        assert ExtractionStrategy.SCENE_CHANGE == "scene_change"
        assert ExtractionStrategy.KEYFRAME == "keyframe"

    def test_script_type_values(self):
        assert ScriptType.DOCUMENTARY == "documentary"
        assert ScriptType.SOCIAL_MEDIA == "social_media"

    def test_shot_type_membership(self):
        assert len(ShotType) == 10
        assert ShotType("wide") is ShotType.WIDE

    def test_scene_type_values(self):
        assert set(SceneType) == {
            SceneType.INTERIOR,
            SceneType.EXTERIOR,
            SceneType.MIXED,
        }

    def test_error_category_values(self):
        assert ErrorCategory.FRAME_EXTRACTION == "frame_extraction"
        assert ErrorCategory.VIDEO_PROBE == "video_probe"


# ---------------------------------------------------------------------------
# Internal model tests
# ---------------------------------------------------------------------------


class TestVideoMetadata:
    def test_minimal_construction(self):
        meta = VideoMetadata(path=Path("/videos/test.mp4"), name="test.mp4")
        assert meta.name == "test.mp4"
        assert meta.duration_seconds is None
        assert meta.width is None

    def test_full_construction(self):
        meta = VideoMetadata(
            path=Path("/videos/test.mp4"),
            name="test.mp4",
            duration_seconds=120.5,
            width=1920,
            height=1080,
            fps=29.97,
            codec="h264",
            format_name="mp4",
            file_size_bytes=50_000_000,
        )
        assert meta.fps == 29.97
        assert meta.file_size_bytes == 50_000_000


class TestFrameExtractionRequest:
    def test_default_values(self):
        meta = VideoMetadata(path=Path("/v.mp4"), name="v.mp4")
        req = FrameExtractionRequest(
            video=meta,
            output_dir=Path("/tmp/frames"),
        )
        assert req.strategy == ExtractionStrategy.UNIFORM
        assert req.max_frames == 10

    def test_max_frames_bounds(self):
        meta = VideoMetadata(path=Path("/v.mp4"), name="v.mp4")
        with pytest.raises(ValidationError):
            FrameExtractionRequest(
                video=meta,
                max_frames=0,
                output_dir=Path("/tmp/frames"),
            )
        with pytest.raises(ValidationError):
            FrameExtractionRequest(
                video=meta,
                max_frames=201,
                output_dir=Path("/tmp/frames"),
            )


class TestExtractedFrame:
    def test_frozen(self):
        frame = ExtractedFrame(
            path=Path("/tmp/frame_0.jpg"),
            timestamp_seconds=1.5,
            frame_index=0,
        )
        with pytest.raises(ValidationError):
            frame.frame_index = 99


class TestTokenUsage:
    def test_construction(self):
        usage = TokenUsage(
            model_name="openai:gpt-4o",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            duration_seconds=1.2,
            step_name="frame_analysis",
        )
        assert usage.total_tokens == 150


class TestPipelineError:
    def test_construction(self):
        err = PipelineError(
            category=ErrorCategory.FRAME_EXTRACTION,
            message="FFmpeg failed",
            detail="exit code 1",
            frame_index=3,
        )
        assert err.category == ErrorCategory.FRAME_EXTRACTION
        assert err.timestamp_seconds is None


# ---------------------------------------------------------------------------
# LLM-output model tests
# ---------------------------------------------------------------------------


class TestFrameAnalysis:
    def test_construction(self):
        analysis = FrameAnalysis(
            frame_path="/tmp/frame_0.jpg",
            timestamp_seconds=0.0,
            description="A wide shot of a city skyline at sunset.",
            scene_type=SceneType.EXTERIOR,
            visual_mood="warm and tranquil",
        )
        assert analysis.scene_type == SceneType.EXTERIOR
        assert analysis.detected_objects == []
        assert analysis.dominant_colors == []


class TestVideoContentSummary:
    def test_construction(self):
        summary = VideoContentSummary(
            overall_summary="A short documentary about urban life.",
            themes=["urban", "community"],
            visual_style="handheld with natural lighting",
            pacing="moderate with contemplative pauses",
            estimated_tone="reflective",
        )
        assert len(summary.themes) == 2
        assert summary.key_moments == []


class TestShootingScript:
    def _build_script(self) -> ShootingScript:
        shot = ShotDefinition(
            shot_number=1,
            shot_type=ShotType.WIDE,
            start_seconds=0.0,
            end_seconds=5.0,
            description="Establishing shot of the building exterior.",
        )
        scene = SceneDefinition(
            scene_number=1,
            title="Opening",
            location="EXT. OFFICE BUILDING - DAY",
            scene_type=SceneType.EXTERIOR,
            mood="professional",
            start_seconds=0.0,
            end_seconds=5.0,
            shots=[shot],
        )
        return ShootingScript(
            title="Product Launch Video",
            script_type=ScriptType.PROMOTIONAL,
            target_duration_seconds=60.0,
            synopsis="A promotional video showcasing the new product launch.",
            scenes=[scene],
        )

    def test_construction(self):
        script = self._build_script()
        assert script.script_type == ScriptType.PROMOTIONAL
        assert len(script.scenes) == 1
        assert script.scenes[0].shots[0].shot_type == ShotType.WIDE

    def test_frozen(self):
        script = self._build_script()
        with pytest.raises(ValidationError):
            script.title = "Changed"

    def test_json_round_trip(self):
        script = self._build_script()
        json_str = script.model_dump_json()
        parsed = json.loads(json_str)
        restored = ShootingScript.model_validate(parsed)
        assert restored == script

    def test_json_schema_is_valid(self):
        schema = ShootingScript.model_json_schema()
        assert schema["title"] == "ShootingScript"
        assert "properties" in schema
        assert "scenes" in schema["properties"]
