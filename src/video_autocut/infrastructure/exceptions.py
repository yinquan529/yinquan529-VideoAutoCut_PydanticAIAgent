from __future__ import annotations


class FFmpegError(Exception):
    """Base exception for all ffmpeg/ffprobe errors."""

    def __init__(self, message: str, stderr: str = "") -> None:
        self.stderr = stderr
        super().__init__(message)


class FFmpegNotFoundError(FFmpegError):
    """The ffmpeg or ffprobe binary could not be found on the system."""


class VideoProbeError(FFmpegError):
    """ffprobe failed to read metadata from a video file."""


class FrameExtractionError(FFmpegError):
    """ffmpeg frame extraction command failed."""


class VideoValidationError(FFmpegError):
    """Video file does not exist, has an unsupported extension, or is unreadable."""
