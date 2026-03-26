from __future__ import annotations

from enum import Enum


class ExtractionStrategy(str, Enum):
    """Strategy for selecting which frames to extract from a video."""

    UNIFORM = "uniform"
    SCENE_CHANGE = "scene_change"
    KEYFRAME = "keyframe"


class ScriptType(str, Enum):
    """Genre or format of the shooting script to generate."""

    DOCUMENTARY = "documentary"
    PROMOTIONAL = "promotional"
    TUTORIAL = "tutorial"
    SOCIAL_MEDIA = "social_media"
    NARRATIVE = "narrative"


class ShotType(str, Enum):
    """Standard camera shot classifications."""

    WIDE = "wide"
    MEDIUM = "medium"
    CLOSE_UP = "close_up"
    EXTREME_CLOSE_UP = "extreme_close_up"
    OVER_THE_SHOULDER = "over_the_shoulder"
    POV = "pov"
    AERIAL = "aerial"
    INSERT = "insert"
    TWO_SHOT = "two_shot"
    ESTABLISHING = "establishing"


class SceneType(str, Enum):
    """Classification of visual scene environment."""

    INTERIOR = "interior"
    EXTERIOR = "exterior"
    MIXED = "mixed"


class ErrorCategory(str, Enum):
    """Category of pipeline error for structured error reporting."""

    FRAME_EXTRACTION = "frame_extraction"
    FRAME_ANALYSIS = "frame_analysis"
    SCRIPT_GENERATION = "script_generation"
    VIDEO_PROBE = "video_probe"
    VALIDATION = "validation"
