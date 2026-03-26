"""Tests for the FFmpegTools wrapper.

All subprocess interaction is mocked via ``monkeypatch.setattr`` on the
``FFmpegTools._run_command`` instance method — no global ``subprocess.run``
patches needed.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from video_autocut.domain.enums import ErrorCategory, ExtractionStrategy
from video_autocut.domain.models import (
    FrameExtractionRequest,
    VideoMetadata,
)
from video_autocut.infrastructure.exceptions import (
    FFmpegNotFoundError,
    VideoProbeError,
    VideoValidationError,
)
from video_autocut.infrastructure.ffmpeg import FFmpegTools, _parse_frame_rate

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

SAMPLE_PROBE_JSON = json.dumps({
    "format": {
        "duration": "120.500000",
        "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
        "size": "50000000",
    },
    "streams": [
        {
            "codec_type": "video",
            "codec_name": "h264",
            "width": 1920,
            "height": 1080,
            "r_frame_rate": "30000/1001",
            "avg_frame_rate": "30000/1001",
        },
    ],
})


def _completed(
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr,
    )


@pytest.fixture()
def tools() -> FFmpegTools:
    return FFmpegTools(ffmpeg_path=Path("ffmpeg"), ffprobe_path=Path("ffprobe"))


def _make_metadata(tmp_path: Path, duration: float | None = 120.5) -> VideoMetadata:
    """Create a minimal VideoMetadata pointing at a dummy file."""
    video_file = tmp_path / "test.mp4"
    video_file.write_bytes(b"\x00")
    return VideoMetadata(
        path=video_file,
        name="test.mp4",
        duration_seconds=duration,
    )


# ---------------------------------------------------------------------------
# check_ffmpeg / check_ffprobe
# ---------------------------------------------------------------------------


class TestCheckFFmpeg:
    def test_available(self, tools: FFmpegTools, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            tools, "_run_command",
            lambda args, **kw: _completed(stdout="ffmpeg version 6.0\n"),
        )
        assert tools.check_ffmpeg() is True

    def test_not_found(self, tools: FFmpegTools, monkeypatch: pytest.MonkeyPatch):
        def _raise(*a, **kw):
            raise FFmpegNotFoundError("not found")
        monkeypatch.setattr(tools, "_run_command", _raise)
        assert tools.check_ffmpeg() is False

    def test_nonzero_exit(self, tools: FFmpegTools, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            tools, "_run_command",
            lambda args, **kw: _completed(returncode=1),
        )
        assert tools.check_ffmpeg() is False


class TestCheckFFprobe:
    def test_available(self, tools: FFmpegTools, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            tools, "_run_command",
            lambda args, **kw: _completed(stdout="ffprobe version 6.0\n"),
        )
        assert tools.check_ffprobe() is True

    def test_not_found(self, tools: FFmpegTools, monkeypatch: pytest.MonkeyPatch):
        def _raise(*a, **kw):
            raise FFmpegNotFoundError("not found")
        monkeypatch.setattr(tools, "_run_command", _raise)
        assert tools.check_ffprobe() is False


# ---------------------------------------------------------------------------
# probe_video
# ---------------------------------------------------------------------------


class TestProbeVideo:
    def test_success(self, tools: FFmpegTools, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            tools, "_run_command",
            lambda args, **kw: _completed(stdout=SAMPLE_PROBE_JSON),
        )
        meta = tools.probe_video(Path("/fake/video.mp4"))
        assert meta.name == "video.mp4"
        assert meta.duration_seconds == 120.5
        assert meta.width == 1920
        assert meta.height == 1080
        assert meta.codec == "h264"
        assert meta.file_size_bytes == 50_000_000
        assert meta.fps is not None
        assert abs(meta.fps - 29.97) < 0.01

    def test_failure_nonzero_exit(
        self, tools: FFmpegTools, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(
            tools, "_run_command",
            lambda args, **kw: _completed(returncode=1, stderr="error msg"),
        )
        with pytest.raises(VideoProbeError, match="ffprobe failed"):
            tools.probe_video(Path("/fake/video.mp4"))

    def test_malformed_json(
        self, tools: FFmpegTools, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(
            tools, "_run_command",
            lambda args, **kw: _completed(stdout="not json at all"),
        )
        with pytest.raises(VideoProbeError, match="invalid JSON"):
            tools.probe_video(Path("/fake/video.mp4"))

    def test_no_video_stream(
        self, tools: FFmpegTools, monkeypatch: pytest.MonkeyPatch
    ):
        audio_only = json.dumps({
            "format": {"duration": "60.0", "format_name": "mp3", "size": "1000000"},
            "streams": [{"codec_type": "audio", "codec_name": "mp3"}],
        })
        monkeypatch.setattr(
            tools, "_run_command",
            lambda args, **kw: _completed(stdout=audio_only),
        )
        meta = tools.probe_video(Path("/fake/audio.mp3"))
        assert meta.width is None
        assert meta.height is None
        assert meta.fps is None
        assert meta.codec is None
        assert meta.duration_seconds == 60.0


# ---------------------------------------------------------------------------
# _parse_frame_rate
# ---------------------------------------------------------------------------


class TestParseFrameRate:
    def test_fraction(self):
        fps = _parse_frame_rate("30000/1001")
        assert fps is not None
        assert abs(fps - 29.97) < 0.01

    def test_integer_fraction(self):
        assert _parse_frame_rate("30/1") == 30.0

    def test_zero_denominator(self):
        assert _parse_frame_rate("0/0") is None

    def test_plain_float(self):
        assert _parse_frame_rate("29.97") == 29.97

    def test_invalid_string(self):
        assert _parse_frame_rate("garbage") is None


# ---------------------------------------------------------------------------
# validate_video
# ---------------------------------------------------------------------------


class TestValidateVideo:
    def test_nonexistent_file(self, tools: FFmpegTools, tmp_path: Path):
        with pytest.raises(VideoValidationError, match="File not found"):
            tools.validate_video(tmp_path / "missing.mp4")

    def test_unsupported_extension(self, tools: FFmpegTools, tmp_path: Path):
        txt_file = tmp_path / "file.txt"
        txt_file.write_text("not a video")
        with pytest.raises(VideoValidationError, match="Unsupported"):
            tools.validate_video(txt_file)

    def test_success(
        self,
        tools: FFmpegTools,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"\x00")
        monkeypatch.setattr(
            tools, "_run_command",
            lambda args, **kw: _completed(stdout=SAMPLE_PROBE_JSON),
        )
        meta = tools.validate_video(video_file)
        assert meta.name == "test.mp4"
        assert meta.width == 1920


# ---------------------------------------------------------------------------
# extract_frames
# ---------------------------------------------------------------------------


class TestExtractFrames:
    def test_uniform_success(
        self,
        tools: FFmpegTools,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        out_dir = tmp_path / "frames"
        meta = _make_metadata(tmp_path, duration=30.0)

        def mock_run(args: list[str], **kw):
            # Create the expected output file so the check passes
            for arg in args:
                if arg.endswith(".jpg"):
                    Path(arg).parent.mkdir(parents=True, exist_ok=True)
                    Path(arg).write_bytes(b"\xff\xd8")
            return _completed()

        monkeypatch.setattr(tools, "_run_command", mock_run)
        request = FrameExtractionRequest(
            video=meta, max_frames=3, output_dir=out_dir,
        )
        result = tools.extract_frames(request)
        assert len(result.frames) == 3
        assert len(result.errors) == 0
        assert result.strategy == ExtractionStrategy.UNIFORM

    def test_uniform_no_duration(
        self,
        tools: FFmpegTools,
        tmp_path: Path,
    ):
        out_dir = tmp_path / "frames"
        meta = _make_metadata(tmp_path, duration=None)

        request = FrameExtractionRequest(
            video=meta, max_frames=5, output_dir=out_dir,
        )
        result = tools.extract_frames(request)
        assert len(result.frames) == 0
        assert len(result.errors) == 1
        assert result.errors[0].category == ErrorCategory.FRAME_EXTRACTION

    def test_filter_based_success(
        self,
        tools: FFmpegTools,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        out_dir = tmp_path / "frames"
        out_dir.mkdir(parents=True)
        meta = _make_metadata(tmp_path, duration=60.0)

        # Pre-create output files that ffmpeg would produce
        for i in range(1, 4):
            (out_dir / f"frame_{i:04d}.jpg").write_bytes(b"\xff\xd8")

        stderr_with_pts = (
            "[Parsed_showinfo] pts_time:5.200\n"
            "[Parsed_showinfo] pts_time:15.800\n"
            "[Parsed_showinfo] pts_time:30.100\n"
        )
        monkeypatch.setattr(
            tools, "_run_command",
            lambda args, **kw: _completed(stderr=stderr_with_pts),
        )
        request = FrameExtractionRequest(
            video=meta,
            strategy=ExtractionStrategy.SCENE_CHANGE,
            max_frames=10,
            output_dir=out_dir,
        )
        result = tools.extract_frames(request)
        assert len(result.frames) == 3
        assert result.frames[0].timestamp_seconds == pytest.approx(5.2)
        assert result.frames[2].timestamp_seconds == pytest.approx(30.1)
        assert result.strategy == ExtractionStrategy.SCENE_CHANGE

    def test_creates_output_dir(
        self,
        tools: FFmpegTools,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        out_dir = tmp_path / "deeply" / "nested" / "dir"
        meta = _make_metadata(tmp_path, duration=10.0)

        def mock_run(args: list[str], **kw):
            for arg in args:
                if arg.endswith(".jpg"):
                    Path(arg).parent.mkdir(parents=True, exist_ok=True)
                    Path(arg).write_bytes(b"\xff\xd8")
            return _completed()

        monkeypatch.setattr(tools, "_run_command", mock_run)
        request = FrameExtractionRequest(
            video=meta, max_frames=1, output_dir=out_dir,
        )
        tools.extract_frames(request)
        assert out_dir.is_dir()


# ---------------------------------------------------------------------------
# cleanup_frames
# ---------------------------------------------------------------------------


class TestCleanupFrames:
    def test_removes_images(self, tmp_path: Path):
        (tmp_path / "frame_0001.jpg").write_bytes(b"\xff")
        (tmp_path / "frame_0002.png").write_bytes(b"\x89")
        count = FFmpegTools.cleanup_frames(tmp_path)
        assert count == 2
        assert not list(tmp_path.glob("*.jpg"))
        assert not list(tmp_path.glob("*.png"))

    def test_preserves_non_images(self, tmp_path: Path):
        (tmp_path / "frame.jpg").write_bytes(b"\xff")
        (tmp_path / "notes.txt").write_text("keep me")
        count = FFmpegTools.cleanup_frames(tmp_path)
        assert count == 1
        assert (tmp_path / "notes.txt").exists()

    def test_nonexistent_directory(self, tmp_path: Path):
        assert FFmpegTools.cleanup_frames(tmp_path / "nope") == 0

    def test_empty_directory(self, tmp_path: Path):
        assert FFmpegTools.cleanup_frames(tmp_path) == 0
