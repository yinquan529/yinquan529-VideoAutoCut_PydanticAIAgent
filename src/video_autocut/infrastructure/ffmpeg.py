"""Windows-friendly wrapper around ffmpeg and ffprobe.

All subprocess calls use argument lists (never ``shell=True``) and
``pathlib.Path`` for every filesystem reference.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path

from video_autocut.domain.enums import ErrorCategory, ExtractionStrategy
from video_autocut.domain.models import (
    ExtractedFrame,
    FrameExtractionRequest,
    FrameExtractionResult,
    PipelineError,
    VideoMetadata,
)
from video_autocut.infrastructure.exceptions import (
    FFmpegError,
    FFmpegNotFoundError,
    VideoProbeError,
    VideoValidationError,
)

logger = logging.getLogger(__name__)

VALID_VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v", ".ts", ".mts"}
)

_IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_frame_rate(rate_str: str) -> float | None:
    """Parse an ffprobe frame-rate string like ``"30000/1001"`` or ``"29.97"``."""
    if "/" in rate_str:
        parts = rate_str.split("/", 1)
        try:
            num, den = float(parts[0]), float(parts[1])
        except ValueError:
            return None
        if den == 0:
            return None
        return num / den
    try:
        value = float(rate_str)
        return value if value > 0 else None
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Main wrapper class
# ---------------------------------------------------------------------------


class FFmpegTools:
    """Safe subprocess wrapper around ffmpeg and ffprobe.

    Every subprocess call goes through :meth:`_run_command`, which is the
    single point to monkeypatch in tests.

    Example usage::

        from video_autocut.settings import get_settings

        settings = get_settings()
        tools = FFmpegTools(settings.ffmpeg_path, settings.ffprobe_path)

        if tools.check_ffmpeg():
            meta = tools.validate_video(Path("input.mp4"))
            result = tools.extract_frames(
                FrameExtractionRequest(
                    video=meta,
                    max_frames=10,
                    output_dir=settings.temp_frames_dir,
                )
            )
    """

    def __init__(
        self,
        ffmpeg_path: Path | None = None,
        ffprobe_path: Path | None = None,
    ) -> None:
        self.ffmpeg_path = ffmpeg_path or Path("ffmpeg")
        self.ffprobe_path = ffprobe_path or Path("ffprobe")

    # ------------------------------------------------------------------
    # Subprocess boundary (mock this in tests)
    # ------------------------------------------------------------------

    def _run_command(
        self,
        args: list[str],
        *,
        timeout: float = 30.0,
    ) -> subprocess.CompletedProcess[str]:
        """Run a command and return its result.

        Raises:
            FFmpegNotFoundError: binary not found on PATH.
            FFmpegError: command timed out.
        """
        logger.debug("Running command: %s", args)
        try:
            return subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except FileNotFoundError as exc:
            raise FFmpegNotFoundError(
                f"Executable not found: {args[0]}. "
                "Ensure it is installed and on your PATH.",
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise FFmpegError(
                f"Command timed out after {timeout}s: {' '.join(args[:3])}...",
            ) from exc

    # ------------------------------------------------------------------
    # Availability checks
    # ------------------------------------------------------------------

    def check_ffmpeg(self) -> bool:
        """Return *True* if ffmpeg is callable and reports a version."""
        return self._check_tool(self.ffmpeg_path, "ffmpeg")

    def check_ffprobe(self) -> bool:
        """Return *True* if ffprobe is callable and reports a version."""
        return self._check_tool(self.ffprobe_path, "ffprobe")

    def _check_tool(self, tool_path: Path, label: str) -> bool:
        try:
            result = self._run_command([str(tool_path), "-version"])
        except FFmpegNotFoundError:
            logger.warning("%s not found at %s", label, tool_path)
            return False

        if result.returncode != 0:
            logger.warning("%s exited with code %d", label, result.returncode)
            return False

        version_line = result.stdout.split("\n", 1)[0].strip()
        logger.info("%s available: %s", label, version_line)
        return True

    # ------------------------------------------------------------------
    # Video probing
    # ------------------------------------------------------------------

    def probe_video(self, video_path: Path) -> VideoMetadata:
        """Run ffprobe and return parsed :class:`VideoMetadata`.

        Raises:
            VideoProbeError: ffprobe failed or produced unparseable output.
        """
        args = [
            str(self.ffprobe_path),
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(video_path),
        ]

        try:
            result = self._run_command(args)
        except FFmpegNotFoundError:
            raise  # let it propagate as-is

        if result.returncode != 0:
            raise VideoProbeError(
                f"ffprobe failed for {video_path.name} (exit {result.returncode})",
                stderr=result.stderr,
            )

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise VideoProbeError(
                f"ffprobe returned invalid JSON for {video_path.name}",
                stderr=result.stdout[:500],
            ) from exc

        return self._parse_probe_output(data, video_path)

    @staticmethod
    def _parse_probe_output(data: dict, video_path: Path) -> VideoMetadata:
        fmt = data.get("format", {})

        duration_raw = fmt.get("duration")
        duration = float(duration_raw) if duration_raw is not None else None

        size_raw = fmt.get("size")
        file_size = int(size_raw) if size_raw is not None else None

        # Find first video stream
        width = height = fps = codec = None
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                width = stream.get("width")
                height = stream.get("height")
                codec = stream.get("codec_name")
                fps = _parse_frame_rate(
                    stream.get("r_frame_rate", "0/0")
                ) or _parse_frame_rate(
                    stream.get("avg_frame_rate", "0/0")
                )
                break

        return VideoMetadata(
            path=video_path.resolve(),
            name=video_path.name,
            duration_seconds=duration,
            width=width,
            height=height,
            fps=fps,
            codec=codec,
            format_name=fmt.get("format_name"),
            file_size_bytes=file_size,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_video(self, video_path: Path) -> VideoMetadata:
        """Check that a file exists, has a video extension, and is probeable.

        Raises:
            VideoValidationError: file missing or unsupported format.
            VideoProbeError: ffprobe could not read the file.
        """
        if not video_path.exists():
            raise VideoValidationError(f"File not found: {video_path}")

        if video_path.suffix.lower() not in VALID_VIDEO_EXTENSIONS:
            raise VideoValidationError(
                f"Unsupported video format '{video_path.suffix}'. "
                f"Supported: {', '.join(sorted(VALID_VIDEO_EXTENSIONS))}"
            )

        return self.probe_video(video_path)

    # ------------------------------------------------------------------
    # Frame extraction
    # ------------------------------------------------------------------

    def extract_frames(
        self,
        request: FrameExtractionRequest,
    ) -> FrameExtractionResult:
        """Extract frames from a video according to the given request."""
        request.output_dir.mkdir(parents=True, exist_ok=True)

        strategy = request.strategy
        if strategy == ExtractionStrategy.UNIFORM:
            frames, errors = self._extract_uniform(request)
        elif strategy == ExtractionStrategy.SCENE_CHANGE:
            frames, errors = self._extract_filter_based(
                request, "select='gt(scene,0.3)',showinfo"
            )
        else:  # KEYFRAME
            frames, errors = self._extract_filter_based(
                request, "select='eq(pict_type,PICT_TYPE_I)',showinfo"
            )

        logger.info(
            "Extracted %d frames (%d errors) from %s using %s strategy",
            len(frames), len(errors), request.video.name, strategy.value,
        )

        return FrameExtractionResult(
            video_name=request.video.name,
            strategy=strategy,
            frames=frames,
            errors=errors,
        )

    def _extract_uniform(
        self, request: FrameExtractionRequest
    ) -> tuple[list[ExtractedFrame], list[PipelineError]]:
        frames: list[ExtractedFrame] = []
        errors: list[PipelineError] = []

        duration = request.video.duration_seconds
        if duration is None or duration <= 0:
            errors.append(PipelineError(
                category=ErrorCategory.FRAME_EXTRACTION,
                message=(
                    f"Cannot extract uniform frames from {request.video.name}: "
                    "duration is unknown"
                ),
            ))
            return frames, errors

        max_frames = request.max_frames
        interval = duration / max_frames

        for i in range(max_frames):
            timestamp = i * interval
            out_path = request.output_dir / f"frame_{i:04d}.jpg"
            args = [
                str(self.ffmpeg_path),
                "-ss", f"{timestamp:.3f}",
                "-i", str(request.video.path),
                "-frames:v", "1",
                "-q:v", "2",
                "-y",
                str(out_path),
            ]

            try:
                result = self._run_command(args, timeout=60.0)
            except FFmpegError as exc:
                errors.append(PipelineError(
                    category=ErrorCategory.FRAME_EXTRACTION,
                    message=f"Frame {i} extraction failed: {exc}",
                    frame_index=i,
                    timestamp_seconds=timestamp,
                ))
                continue

            if result.returncode != 0 or not out_path.exists():
                errors.append(PipelineError(
                    category=ErrorCategory.FRAME_EXTRACTION,
                    message=f"Frame {i} extraction failed (exit {result.returncode})",
                    detail=result.stderr[:300],
                    frame_index=i,
                    timestamp_seconds=timestamp,
                ))
                continue

            frames.append(ExtractedFrame(
                path=out_path,
                timestamp_seconds=timestamp,
                frame_index=i,
            ))

        return frames, errors

    def _extract_filter_based(
        self,
        request: FrameExtractionRequest,
        vf_filter: str,
    ) -> tuple[list[ExtractedFrame], list[PipelineError]]:
        frames: list[ExtractedFrame] = []
        errors: list[PipelineError] = []

        pattern = request.output_dir / "frame_%04d.jpg"
        args = [
            str(self.ffmpeg_path),
            "-i", str(request.video.path),
            "-vf", vf_filter,
            "-frames:v", str(request.max_frames),
            "-vsync", "vfn",
            "-q:v", "2",
            "-y",
            str(pattern),
        ]

        try:
            result = self._run_command(args, timeout=300.0)
        except FFmpegError as exc:
            errors.append(PipelineError(
                category=ErrorCategory.FRAME_EXTRACTION,
                message=f"Filter-based extraction failed: {exc}",
            ))
            return frames, errors

        if result.returncode != 0:
            errors.append(PipelineError(
                category=ErrorCategory.FRAME_EXTRACTION,
                message=(
                    f"Filter-based extraction failed (exit {result.returncode})"
                ),
                detail=result.stderr[:300],
            ))
            return frames, errors

        # Parse real timestamps from showinfo output
        timestamps = [
            float(m) for m in re.findall(r"pts_time:\s*([\d.]+)", result.stderr)
        ]

        # Collect output files
        output_files = sorted(
            f for f in request.output_dir.iterdir()
            if f.is_file() and f.name.startswith("frame_") and f.suffix == ".jpg"
        )

        for i, file_path in enumerate(output_files):
            ts = timestamps[i] if i < len(timestamps) else 0.0
            frames.append(ExtractedFrame(
                path=file_path,
                timestamp_seconds=ts,
                frame_index=i,
            ))

        return frames, errors

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    @staticmethod
    def cleanup_frames(directory: Path) -> int:
        """Remove image files from *directory* and return the count deleted."""
        if not directory.exists():
            logger.warning("Cleanup skipped: directory does not exist: %s", directory)
            return 0

        count = 0
        for file_path in directory.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in _IMAGE_EXTENSIONS:
                file_path.unlink()
                count += 1

        logger.debug("Cleaned up %d frame(s) from %s", count, directory)
        return count
