"""LLM-output models for the video script generation pipeline.

Every model in this module is designed to be safe as a PydanticAI ``output_type``:
no ``Path``, no ``datetime`` — only ``str``, ``int``, ``float``, ``bool``,
``list``, nested ``BaseModel``, and ``str`` enums.  All models are frozen.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from video_autocut.domain.enums import SceneType, ScriptType, ShotType

# ---------------------------------------------------------------------------
# Frame / image analysis
# ---------------------------------------------------------------------------


class FrameAnalysis(BaseModel):
    """Vision LLM analysis of a single extracted video frame."""

    model_config = ConfigDict(frozen=True)

    frame_path: str = Field(
        description="File path of the analyzed frame image",
    )
    timestamp_seconds: float = Field(
        ge=0.0,
        description="Position in the source video where this frame occurs, in seconds",
    )
    description: str = Field(
        description=(
            "One or two sentence natural-language description "
            "of what is visible in the frame"
        ),
    )
    detected_objects: list[str] = Field(
        default_factory=list,
        description="Notable objects, people, or elements visible in the frame",
    )
    scene_type: SceneType = Field(
        description="Whether the scene is interior, exterior, or mixed",
    )
    visual_mood: str = Field(
        description="The emotional tone or atmosphere conveyed by the frame, "
        "e.g. warm and inviting, tense, serene, or energetic",
    )
    dominant_colors: list[str] = Field(
        default_factory=list,
        description="Up to three dominant colors in the frame, e.g. deep blue or warm orange",
    )


# ---------------------------------------------------------------------------
# Video content summary
# ---------------------------------------------------------------------------


class KeyMoment(BaseModel):
    """A notable moment identified during video analysis."""

    model_config = ConfigDict(frozen=True)

    timestamp_seconds: float = Field(
        ge=0.0,
        description="When this moment occurs in the video, in seconds",
    )
    description: str = Field(
        description="What happens at this moment and why it is significant",
    )


class VideoContentSummary(BaseModel):
    """Aggregated analysis of an entire video, synthesized from individual frame analyses."""

    model_config = ConfigDict(frozen=True)

    overall_summary: str = Field(
        description=(
            "A coherent two to four sentence summary "
            "of the video content and narrative arc"
        ),
    )
    themes: list[str] = Field(
        description="Major themes or topics present in the video, "
        "e.g. urban exploration or product showcase",
    )
    visual_style: str = Field(
        description="Description of the overall visual style and cinematography, "
        "e.g. handheld documentary style with natural lighting",
    )
    pacing: str = Field(
        description="Assessment of the video pacing and rhythm, "
        "e.g. slow and contemplative or fast-paced with quick cuts",
    )
    key_moments: list[KeyMoment] = Field(
        default_factory=list,
        description="Chronologically ordered notable moments in the video",
    )
    estimated_tone: str = Field(
        description="The overall emotional tone of the video, "
        "e.g. inspirational, informative, or dramatic",
    )


# ---------------------------------------------------------------------------
# Shot definition
# ---------------------------------------------------------------------------


class ShotDefinition(BaseModel):
    """A single camera shot within a scene of a shooting script."""

    model_config = ConfigDict(frozen=True)

    shot_number: int = Field(
        ge=1,
        description="Sequential shot number within the parent scene, starting at 1",
    )
    shot_type: ShotType = Field(
        description="Camera shot classification such as wide, close_up, or medium",
    )
    start_seconds: float = Field(
        ge=0.0,
        description="When this shot begins in the final timeline, in seconds",
    )
    end_seconds: float = Field(
        ge=0.0,
        description="When this shot ends in the final timeline, in seconds",
    )
    description: str = Field(
        description="What the camera shows during this shot: subject, action, and visual intent",
    )
    framing_notes: str = Field(
        default="",
        description="Additional direction for camera operator, "
        "e.g. rack focus from foreground to subject or slow pan left",
    )
    camera_movement: str = Field(
        default="static",
        description=(
            "Type of camera movement such as static, "
            "pan left, dolly in, or handheld tracking"
        ),
    )


# ---------------------------------------------------------------------------
# Scene definition
# ---------------------------------------------------------------------------


class SceneDefinition(BaseModel):
    """A group of shots forming a narrative unit in the shooting script."""

    model_config = ConfigDict(frozen=True)

    scene_number: int = Field(
        ge=1,
        description="Sequential scene number in the script, starting at 1",
    )
    title: str = Field(
        description=(
            "Short descriptive title for this scene, "
            "e.g. Opening Montage or Interview Segment"
        ),
    )
    location: str = Field(
        description="Where the scene takes place, e.g. EXT. ROOFTOP - DAY or INT. OFFICE",
    )
    scene_type: SceneType = Field(
        description="Whether the scene is interior, exterior, or mixed",
    )
    mood: str = Field(
        description="The intended emotional atmosphere of the scene, "
        "e.g. mysterious, upbeat, or reflective",
    )
    start_seconds: float = Field(
        ge=0.0,
        description="When this scene begins in the final timeline, in seconds",
    )
    end_seconds: float = Field(
        ge=0.0,
        description="When this scene ends in the final timeline, in seconds",
    )
    shots: list[ShotDefinition] = Field(
        description="Ordered list of camera shots that compose this scene",
    )
    scene_direction: str = Field(
        default="",
        description="Overall directorial notes for the scene",
    )


# ---------------------------------------------------------------------------
# Narration cue
# ---------------------------------------------------------------------------


class NarrationCue(BaseModel):
    """Voiceover or narration text tied to a timecode range."""

    model_config = ConfigDict(frozen=True)

    start_seconds: float = Field(
        ge=0.0,
        description="When the narration begins, in seconds",
    )
    end_seconds: float = Field(
        ge=0.0,
        description="When the narration ends, in seconds",
    )
    text: str = Field(
        description="The exact narration or voiceover text to be spoken",
    )
    speaker: str = Field(
        default="narrator",
        description="Who delivers this line, e.g. narrator, host, or interview_subject",
    )
    tone: str = Field(
        default="",
        description="How the line should be delivered, e.g. warm, authoritative, or excited",
    )


# ---------------------------------------------------------------------------
# Music cue
# ---------------------------------------------------------------------------


class MusicCue(BaseModel):
    """Background music suggestion tied to a timecode range."""

    model_config = ConfigDict(frozen=True)

    start_seconds: float = Field(
        ge=0.0,
        description="When the music begins, in seconds",
    )
    end_seconds: float = Field(
        ge=0.0,
        description="When the music ends or fades out, in seconds",
    )
    mood: str = Field(
        description="The emotional quality of the music, e.g. uplifting, tense, or nostalgic",
    )
    genre: str = Field(
        description="Musical genre or style, e.g. ambient electronic, acoustic folk, or orchestral",
    )
    tempo: str = Field(
        default="medium",
        description="General tempo indication such as slow, medium, fast, or building",
    )
    notes: str = Field(
        default="",
        description="Additional music direction, e.g. fade in over 3 seconds or underscore only",
    )


# ---------------------------------------------------------------------------
# Shooting script — the primary pipeline output
# ---------------------------------------------------------------------------


class ShootingScript(BaseModel):
    """Complete shooting script: the primary output of the video-to-script pipeline."""

    model_config = ConfigDict(frozen=True)

    title: str = Field(
        description="Title of the shooting script",
    )
    script_type: ScriptType = Field(
        description=(
            "Genre or format of this script: documentary, "
            "promotional, tutorial, social_media, or narrative"
        ),
    )
    target_duration_seconds: float = Field(
        ge=0.0,
        description="Intended total duration of the final video in seconds",
    )
    synopsis: str = Field(
        description="A two to four sentence high-level summary of the script narrative and purpose",
    )
    scenes: list[SceneDefinition] = Field(
        description="Ordered list of scenes that compose the script",
    )
    narration_cues: list[NarrationCue] = Field(
        default_factory=list,
        description="Voiceover and narration cues across the entire script timeline",
    )
    music_cues: list[MusicCue] = Field(
        default_factory=list,
        description="Background music suggestions across the entire script timeline",
    )
    production_notes: str = Field(
        default="",
        description="General production notes, style guidance, or special requirements",
    )
