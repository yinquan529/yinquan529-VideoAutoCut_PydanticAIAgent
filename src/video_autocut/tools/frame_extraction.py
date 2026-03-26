"""Agent-tool-friendly frame extraction facade.

Hides FFmpegTools construction, settings lookup, duplicate detection,
and cleanup behind simple function calls that accept plain strings
and primitives.

Example usage::

    # Uniform extraction (simplest)
    result = extract_frames("input.mp4")

    # Scene-change extraction with more frames
    result = extract_frames("input.mp4", strategy="scene_change", max_frames=20)

    # Keyframe extraction with automatic cleanup
    with extraction_context("input.mp4", strategy="keyframe") as result:
        for frame in result.frames:
            print(frame.path, frame.timestamp_seconds)
    # frame files are deleted here
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from video_autocut.domain.enums import ExtractionStrategy
from video_autocut.domain.models import (
    ExtractedFrame,
    FrameExtractionRequest,
    FrameExtractionResult,
)
from video_autocut.infrastructure.ffmpeg import FFmpegTools
from video_autocut.settings import get_settings

logger = logging.getLogger(__name__)

# Frames smaller than this are almost certainly blank/corrupt.
_MIN_FRAME_BYTES: int = 1024


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------


def extract_frames(
    video_path: str | Path,
    strategy: str = "uniform",
    max_frames: int = 10,
    output_dir: str | Path | None = None,
    deduplicate: bool = True,
) -> FrameExtractionResult:
    """Extract frames from a video file.

    This is the primary entry point for agent tools.  It accepts plain
    strings so callers never need to construct domain objects.

    Args:
        video_path: Path to the video file.
        strategy: One of ``"uniform"``, ``"scene_change"``, ``"keyframe"``.
        max_frames: Maximum frames to extract (1--200).
        output_dir: Where to write frames.  Defaults to
            ``settings.temp_frames_dir / <video_stem>``.
        deduplicate: Remove likely-duplicate frames after extraction.

    Returns:
        FrameExtractionResult with frames list and any errors.
    """
    settings = get_settings()
    tools = FFmpegTools(settings.ffmpeg_path, settings.ffprobe_path)

    video_path = Path(video_path)
    extraction_strategy = ExtractionStrategy(strategy)

    metadata = tools.validate_video(video_path)

    if output_dir is None:
        output_dir = settings.temp_frames_dir / _safe_dirname(video_path.stem)
    else:
        output_dir = Path(output_dir)

    request = FrameExtractionRequest(
        video=metadata,
        strategy=extraction_strategy,
        max_frames=max_frames,
        output_dir=output_dir,
    )

    result = tools.extract_frames(request)

    if deduplicate and len(result.frames) > 1:
        unique = _remove_duplicates(result.frames)
        if len(unique) < len(result.frames):
            logger.info(
                "Deduplication removed %d frame(s)",
                len(result.frames) - len(unique),
            )
            result = FrameExtractionResult(
                video_name=result.video_name,
                strategy=result.strategy,
                frames=unique,
                errors=result.errors,
            )

    return result


@contextmanager
def extraction_context(
    video_path: str | Path,
    strategy: str = "uniform",
    max_frames: int = 10,
    output_dir: str | Path | None = None,
    deduplicate: bool = True,
) -> Iterator[FrameExtractionResult]:
    """Context manager that extracts frames and cleans up on exit.

    Usage::

        with extraction_context("video.mp4", strategy="scene_change") as result:
            for frame in result.frames:
                analyze(frame.path)
        # frames are deleted here
    """
    result = extract_frames(
        video_path, strategy, max_frames, output_dir, deduplicate,
    )
    try:
        yield result
    finally:
        if result.frames:
            directory = result.frames[0].path.parent
            count = cleanup_frames(directory)
            logger.debug("Context cleanup removed %d file(s)", count)


def cleanup_frames(directory: str | Path) -> int:
    """Remove extracted frame images from a directory.

    Thin wrapper around :meth:`FFmpegTools.cleanup_frames` that accepts
    ``str`` paths for convenience.
    """
    return FFmpegTools.cleanup_frames(Path(directory))


# -------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------


def _safe_dirname(stem: str) -> str:
    """Produce a filesystem-safe directory name from a video stem."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)


def _remove_duplicates(frames: list[ExtractedFrame]) -> list[ExtractedFrame]:
    """Remove likely-duplicate frames using file-size comparison.

    Two consecutive frames whose file sizes are identical are considered
    duplicates; the later one is removed and its file is deleted.
    Frames below :data:`_MIN_FRAME_BYTES` are also removed as likely
    corrupt or blank.
    """
    if not frames:
        return frames

    unique: list[ExtractedFrame] = []
    prev_size: int | None = None

    for frame in frames:
        path = frame.path
        if not path.exists():
            continue

        size = path.stat().st_size
        if size < _MIN_FRAME_BYTES:
            logger.debug("Removing tiny frame %s (%d bytes)", path.name, size)
            path.unlink(missing_ok=True)
            continue

        if prev_size is not None and size == prev_size:
            logger.debug(
                "Removing duplicate frame %s (same size as previous)", path.name,
            )
            path.unlink(missing_ok=True)
            continue

        prev_size = size
        unique.append(frame)

    # Re-index to keep frame_index contiguous
    return [
        ExtractedFrame(
            path=f.path,
            timestamp_seconds=f.timestamp_seconds,
            frame_index=i,
        )
        for i, f in enumerate(unique)
    ]
