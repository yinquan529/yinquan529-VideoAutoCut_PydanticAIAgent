"""Tests for the shooting-script generator and Markdown renderer.

Mocking strategy: ``create_structured_agent`` is patched to return a fake
agent whose ``run()`` returns a pre-built ``ShootingScript``.
No real LLM calls are made.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest

from video_autocut.domain.enums import (
    SceneType,
    ScriptType,
    ShotType,
)
from video_autocut.domain.models import PipelineRunStats
from video_autocut.domain.results import (
    ScriptGenerationResult,
    VideoAnalysisResult,
)
from video_autocut.domain.script_models import (
    FrameAnalysis,
    KeyMoment,
    MusicCue,
    NarrationCue,
    SceneDefinition,
    ShootingScript,
    ShotDefinition,
    VideoContentSummary,
)
from video_autocut.settings import Settings, get_settings
from video_autocut.tools.script_generator import (
    _build_system_prompt,
    _build_user_prompt,
    generate_script,
)
from video_autocut.tools.script_renderer import (
    _fmt_range,
    _fmt_time,
    render_music,
    render_narration,
    render_scene,
    render_script,
    render_shot,
)

# ---------------------------------------------------------------------------
# Shared fixtures
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


SAMPLE_FRAME_ANALYSIS = FrameAnalysis(
    frame_path="frame_0000.jpg",
    timestamp_seconds=0.0,
    description="A wide shot of a city skyline at sunset.",
    detected_objects=["buildings", "sky"],
    scene_type=SceneType.EXTERIOR,
    visual_mood="warm",
    dominant_colors=["orange", "blue"],
)

SAMPLE_SUMMARY = VideoContentSummary(
    overall_summary="A cinematic tour of an urban skyline.",
    themes=["urban", "golden hour"],
    visual_style="Wide cinematic shots.",
    pacing="Slow and contemplative.",
    key_moments=[
        KeyMoment(timestamp_seconds=0.0, description="Opening skyline"),
        KeyMoment(timestamp_seconds=30.0, description="Sunset peak"),
    ],
    estimated_tone="serene",
)

SAMPLE_ANALYSIS = VideoAnalysisResult(
    video_name="city_sunset.mp4",
    frame_analyses=[SAMPLE_FRAME_ANALYSIS],
    content_summary=SAMPLE_SUMMARY,
    stats=PipelineRunStats(total_duration_seconds=5.0),
)

SAMPLE_SHOT = ShotDefinition(
    shot_number=1,
    shot_type=ShotType.WIDE,
    start_seconds=0.0,
    end_seconds=10.0,
    description="Establishing shot of city skyline",
    framing_notes="Rule of thirds; skyline in lower third",
    camera_movement="slow pan right",
)

SAMPLE_SCENE = SceneDefinition(
    scene_number=1,
    title="Opening Skyline",
    location="EXT. ROOFTOP - SUNSET",
    scene_type=SceneType.EXTERIOR,
    mood="serene",
    start_seconds=0.0,
    end_seconds=15.0,
    shots=[SAMPLE_SHOT],
    scene_direction="Let the sunset breathe; hold the wide shot.",
)

SAMPLE_NARRATION = NarrationCue(
    start_seconds=0.0,
    end_seconds=8.0,
    text="As the sun dips below the skyline, the city transforms.",
    speaker="narrator",
    tone="warm",
)

SAMPLE_MUSIC = MusicCue(
    start_seconds=0.0,
    end_seconds=30.0,
    mood="serene",
    genre="ambient electronic",
    tempo="slow",
    notes="Fade in over 3 seconds",
)

SAMPLE_SCRIPT = ShootingScript(
    title="Golden Hour",
    script_type=ScriptType.DOCUMENTARY,
    target_duration_seconds=60.0,
    synopsis="A meditative short capturing the city at golden hour.",
    scenes=[SAMPLE_SCENE],
    narration_cues=[SAMPLE_NARRATION],
    music_cues=[SAMPLE_MUSIC],
    production_notes="Shoot during actual golden hour for authentic color.",
)


# ---------------------------------------------------------------------------
# Fake agent
# ---------------------------------------------------------------------------


@dataclass
class FakeUsage:
    input_tokens: int = 200
    output_tokens: int = 300
    total_tokens: int = 500


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


# ---------------------------------------------------------------------------
# TestGenerateScript
# ---------------------------------------------------------------------------


class TestGenerateScript:
    def _patch_agent(self, output):
        return patch(
            "video_autocut.tools.script_generator.create_structured_agent",
            return_value=FakeAgent(output),
        )

    async def test_happy_path(self):
        settings = _make_settings()
        with self._patch_agent(SAMPLE_SCRIPT):
            result = await generate_script(
                SAMPLE_ANALYSIS,
                script_type="documentary",
                target_duration=60.0,
                settings=settings,
            )
        assert isinstance(result, ScriptGenerationResult)
        assert result.script is not None
        assert result.script.title == "Golden Hour"
        assert result.error is None
        assert result.token_usage is not None
        assert result.token_usage.step_name == "script_generation"

    async def test_token_usage_captured(self):
        settings = _make_settings()
        with self._patch_agent(SAMPLE_SCRIPT):
            result = await generate_script(
                SAMPLE_ANALYSIS, settings=settings,
            )
        assert result.token_usage.prompt_tokens == 200
        assert result.token_usage.completion_tokens == 300
        assert result.token_usage.total_tokens == 500
        assert result.token_usage.model_name == "test-model"

    async def test_error_collected(self):
        settings = _make_settings()
        with self._patch_agent(RuntimeError("LLM timeout")):
            result = await generate_script(
                SAMPLE_ANALYSIS, settings=settings,
            )
        assert result.script is None
        assert result.token_usage is None
        assert result.error is not None
        assert "Script generation failed" in result.error.message

    async def test_string_script_type(self):
        settings = _make_settings()
        with self._patch_agent(SAMPLE_SCRIPT):
            result = await generate_script(
                SAMPLE_ANALYSIS,
                script_type="promotional",
                settings=settings,
            )
        assert result.script is not None

    async def test_enum_script_type(self):
        settings = _make_settings()
        with self._patch_agent(SAMPLE_SCRIPT):
            result = await generate_script(
                SAMPLE_ANALYSIS,
                script_type=ScriptType.TUTORIAL,
                settings=settings,
            )
        assert result.script is not None

    async def test_invalid_script_type_raises(self):
        settings = _make_settings()
        with pytest.raises(ValueError):
            await generate_script(
                SAMPLE_ANALYSIS,
                script_type="nonexistent",
                settings=settings,
            )

    async def test_default_settings(self, monkeypatch: pytest.MonkeyPatch):
        settings = _make_settings()
        monkeypatch.setattr(
            "video_autocut.tools.script_generator.get_settings",
            lambda: settings,
        )
        with self._patch_agent(SAMPLE_SCRIPT):
            result = await generate_script(SAMPLE_ANALYSIS)
        assert result.script is not None

    async def test_all_creative_params(self):
        """Verify audience, style, emphasis are passed to the prompt."""
        settings = _make_settings()
        prompts_received = []

        class CapturingAgent:
            async def run(self, user_prompt, **kwargs):
                prompts_received.append(user_prompt)
                return FakeRunResult(SAMPLE_SCRIPT)

        with patch(
            "video_autocut.tools.script_generator.create_structured_agent",
            return_value=CapturingAgent(),
        ):
            await generate_script(
                SAMPLE_ANALYSIS,
                script_type="narrative",
                target_duration=90.0,
                target_audience="young adults",
                style="cinematic noir",
                emphasis="focus on the rooftop scenes",
                settings=settings,
            )

        prompt = prompts_received[0]
        assert "young adults" in prompt
        assert "cinematic noir" in prompt
        assert "focus on the rooftop scenes" in prompt
        assert "90 seconds" in prompt
        assert "narrative" in prompt


# ---------------------------------------------------------------------------
# TestBuildPrompts
# ---------------------------------------------------------------------------


class TestBuildPrompts:
    def test_system_prompt_documentary(self):
        prompt = _build_system_prompt(ScriptType.DOCUMENTARY)
        assert "documentary" in prompt.lower()
        assert "expert video director" in prompt.lower()

    def test_system_prompt_promotional(self):
        prompt = _build_system_prompt(ScriptType.PROMOTIONAL)
        assert "hook" in prompt.lower()
        assert "CTA" in prompt

    def test_system_prompt_tutorial(self):
        prompt = _build_system_prompt(ScriptType.TUTORIAL)
        assert "tutorial" in prompt.lower()

    def test_system_prompt_social_media(self):
        prompt = _build_system_prompt(ScriptType.SOCIAL_MEDIA)
        assert "social media" in prompt.lower()

    def test_system_prompt_narrative(self):
        prompt = _build_system_prompt(ScriptType.NARRATIVE)
        assert "three-act" in prompt.lower()

    def test_user_prompt_includes_video_name(self):
        prompt = _build_user_prompt(
            SAMPLE_ANALYSIS, ScriptType.DOCUMENTARY, 60.0, "", "", "",
        )
        assert "city_sunset.mp4" in prompt

    def test_user_prompt_includes_summary(self):
        prompt = _build_user_prompt(
            SAMPLE_ANALYSIS, ScriptType.DOCUMENTARY, 60.0, "", "", "",
        )
        assert "cinematic tour" in prompt.lower()
        assert "golden hour" in prompt.lower()

    def test_user_prompt_no_summary(self):
        analysis = VideoAnalysisResult(
            video_name="bare.mp4",
            stats=PipelineRunStats(total_duration_seconds=1.0),
        )
        prompt = _build_user_prompt(
            analysis, ScriptType.DOCUMENTARY, 30.0, "", "", "",
        )
        assert "bare.mp4" in prompt

    def test_user_prompt_zero_duration(self):
        prompt = _build_user_prompt(
            SAMPLE_ANALYSIS, ScriptType.DOCUMENTARY, 0.0, "", "", "",
        )
        assert "infer from source" in prompt.lower()

    def test_user_prompt_key_moments(self):
        prompt = _build_user_prompt(
            SAMPLE_ANALYSIS, ScriptType.DOCUMENTARY, 60.0, "", "", "",
        )
        assert "Opening skyline" in prompt
        assert "30.0s" in prompt


# ---------------------------------------------------------------------------
# TestRenderScript (Markdown renderer)
# ---------------------------------------------------------------------------


class TestRenderScript:
    def test_full_render(self):
        md = render_script(SAMPLE_SCRIPT)
        assert "# Golden Hour" in md
        assert "documentary" in md
        assert "01:00" in md  # 60 seconds
        assert "Synopsis" in md
        assert "Scenes" in md
        assert "Opening Skyline" in md
        assert "Narration" in md
        assert "Music" in md
        assert "Production Notes" in md
        assert "golden hour" in md.lower()

    def test_no_narration(self):
        script = ShootingScript(
            title="Minimal",
            script_type=ScriptType.TUTORIAL,
            target_duration_seconds=30.0,
            synopsis="A quick tutorial.",
            scenes=[SAMPLE_SCENE],
        )
        md = render_script(script)
        assert "Narration" not in md

    def test_no_music(self):
        script = ShootingScript(
            title="Minimal",
            script_type=ScriptType.TUTORIAL,
            target_duration_seconds=30.0,
            synopsis="A quick tutorial.",
            scenes=[SAMPLE_SCENE],
        )
        md = render_script(script)
        assert "Music" not in md

    def test_no_production_notes(self):
        script = ShootingScript(
            title="Minimal",
            script_type=ScriptType.TUTORIAL,
            target_duration_seconds=30.0,
            synopsis="A quick tutorial.",
            scenes=[SAMPLE_SCENE],
        )
        md = render_script(script)
        assert "Production Notes" not in md

    def test_empty_scenes(self):
        script = ShootingScript(
            title="Empty",
            script_type=ScriptType.NARRATIVE,
            target_duration_seconds=10.0,
            synopsis="Nothing to see.",
            scenes=[],
        )
        md = render_script(script)
        assert "Scenes" not in md


class TestRenderShot:
    def test_basic(self):
        md = render_shot(SAMPLE_SHOT)
        assert "Shot 1" in md
        assert "wide" in md
        assert "Establishing shot" in md
        assert "slow pan right" in md
        assert "Rule of thirds" in md

    def test_static_no_camera_line(self):
        shot = ShotDefinition(
            shot_number=1,
            shot_type=ShotType.CLOSE_UP,
            start_seconds=0.0,
            end_seconds=5.0,
            description="Close-up of hands typing",
            camera_movement="static",
        )
        md = render_shot(shot)
        assert "Camera:" not in md


class TestRenderScene:
    def test_includes_direction(self):
        md = render_scene(SAMPLE_SCENE)
        assert "Let the sunset breathe" in md
        assert "EXT. ROOFTOP" in md

    def test_no_direction(self):
        scene = SceneDefinition(
            scene_number=1,
            title="Test",
            location="INT. STUDIO",
            scene_type=SceneType.INTERIOR,
            mood="neutral",
            start_seconds=0.0,
            end_seconds=10.0,
            shots=[SAMPLE_SHOT],
        )
        md = render_scene(scene)
        assert ">" not in md  # No blockquote


class TestRenderNarration:
    def test_table_rows(self):
        md = render_narration([SAMPLE_NARRATION])
        assert "Narration" in md
        assert "narrator" in md
        assert "sun dips" in md

    def test_empty(self):
        assert render_narration([]) == ""

    def test_no_tone_shows_dash(self):
        cue = NarrationCue(
            start_seconds=0.0, end_seconds=5.0,
            text="Hello world",
        )
        md = render_narration([cue])
        assert "—" in md


class TestRenderMusic:
    def test_table_rows(self):
        md = render_music([SAMPLE_MUSIC])
        assert "Music" in md
        assert "ambient electronic" in md
        assert "Fade in" in md

    def test_empty(self):
        assert render_music([]) == ""

    def test_no_notes_shows_dash(self):
        cue = MusicCue(
            start_seconds=0.0, end_seconds=10.0,
            mood="happy", genre="pop",
        )
        md = render_music([cue])
        assert "—" in md


class TestFormatHelpers:
    def test_fmt_time(self):
        assert _fmt_time(0) == "00:00"
        assert _fmt_time(65) == "01:05"
        assert _fmt_time(3661) == "61:01"

    def test_fmt_range(self):
        assert _fmt_range(0, 10) == "[00:00 – 00:10]"


# ---------------------------------------------------------------------------
# TestScriptGenerationResult schema
# ---------------------------------------------------------------------------


class TestScriptGenerationResultSchema:
    def test_json_schema(self):
        schema = ScriptGenerationResult.model_json_schema()
        assert "script" in schema["properties"]
        assert "token_usage" in schema["properties"]
        assert "error" in schema["properties"]

    def test_frozen(self):
        result = ScriptGenerationResult()
        with pytest.raises(Exception):
            result.script = None
