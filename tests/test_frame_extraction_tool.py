"""Tests for the frame extraction tool facade.

Mocking strategy mirrors ``test_ffmpeg.py``: monkeypatch
``FFmpegTools._run_command`` for subprocess isolation and
``get_settings`` for directory configuration.
"""

from __future__ import annotations

import json
import subprocess
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch

import pytest

from video_autocut.domain.enums import ExtractionStrategy
from video_autocut.domain.models import ExtractedFrame
from video_autocut.settings import Settings
from video_autocut.tools.frame_extraction import (
    _remove_duplicates,
    _safe_dirname,
    cleanup_frames,
    extract_frames,
    extraction_context,
)

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


def _make_settings(tmp_path: Path) -> Settings:
    """Create a Settings instance with tmp_path-based directories."""
    return Settings(
        hunyuan_api_key="test-key",
        hunyuan_base_url="http://localhost",
        ffmpeg_path=Path("ffmpeg"),
        ffprobe_path=Path("ffprobe"),
        temp_frames_dir=tmp_path / "frames",
        output_dir=tmp_path / "output",
    )


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Ensure get_settings cache is cleared between tests."""
    from video_autocut.settings import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture()
def mock_settings(tmp_path: Path):
    """Patch get_settings to return tmp_path-based settings."""
    settings = _make_settings(tmp_path)

    @lru_cache
    def fake_get_settings() -> Settings:
        return settings

    with patch(
        "video_autocut.tools.frame_extraction.get_settings",
        fake_get_settings,
    ):
        yield settings


@pytest.fixture()
def video_file(tmp_path: Path) -> Path:
    """Create a dummy .mp4 file."""
    f = tmp_path / "sample_video.mp4"
    f.write_bytes(b"\x00" * 100)
    return f


# ---------------------------------------------------------------------------
# TestExtractFrames
# ---------------------------------------------------------------------------


class TestExtractFrames:
    def test_uniform_basic(
        self,
        mock_settings: Settings,
        video_file: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        out_dir = tmp_path / "out"
        call_count = 0

        def mock_run(self_tools, args: list[str], **kw):
            nonlocal call_count
            # First call is ffprobe (validate_video → probe_video)
            if call_count == 0:
                call_count += 1
                return _completed(stdout=SAMPLE_PROBE_JSON)
            # Subsequent calls are ffmpeg frame extractions
            call_count += 1
            for arg in args:
                if arg.endswith(".jpg"):
                    Path(arg).parent.mkdir(parents=True, exist_ok=True)
                    # Each frame gets a distinct size to avoid dedup
                    Path(arg).write_bytes(b"\xff\xd8" + b"\x00" * (2000 + call_count * 100))
            return _completed()

        from video_autocut.infrastructure.ffmpeg import FFmpegTools

        monkeypatch.setattr(FFmpegTools, "_run_command", mock_run)

        result = extract_frames(
            video_file, strategy="uniform", max_frames=3, output_dir=out_dir,
        )
        assert result.strategy == ExtractionStrategy.UNIFORM
        assert len(result.frames) == 3
        assert len(result.errors) == 0
        assert all(f.path.exists() for f in result.frames)

    def test_scene_change_strategy(
        self,
        mock_settings: Settings,
        video_file: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        out_dir = tmp_path / "out"

        def mock_run(self_tools, args: list[str], **kw):
            if "ffprobe" in args[0] or "-show_format" in args:
                return _completed(stdout=SAMPLE_PROBE_JSON)
            # Filter-based: create output files and return pts_time
            out_dir.mkdir(parents=True, exist_ok=True)
            for i in range(1, 3):
                (out_dir / f"frame_{i:04d}.jpg").write_bytes(
                    b"\xff\xd8" + b"\x00" * (2000 + i * 100)
                )
            return _completed(
                stderr=(
                    "[Parsed_showinfo] pts_time:5.200\n"
                    "[Parsed_showinfo] pts_time:15.800\n"
                ),
            )

        from video_autocut.infrastructure.ffmpeg import FFmpegTools

        monkeypatch.setattr(FFmpegTools, "_run_command", mock_run)

        result = extract_frames(
            video_file, strategy="scene_change", max_frames=10, output_dir=out_dir,
        )
        assert result.strategy == ExtractionStrategy.SCENE_CHANGE
        assert len(result.frames) == 2

    def test_keyframe_strategy(
        self,
        mock_settings: Settings,
        video_file: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        out_dir = tmp_path / "out"

        def mock_run(self_tools, args: list[str], **kw):
            if "ffprobe" in args[0] or "-show_format" in args:
                return _completed(stdout=SAMPLE_PROBE_JSON)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "frame_0001.jpg").write_bytes(b"\xff\xd8" + b"\x00" * 3000)
            return _completed(stderr="[Parsed_showinfo] pts_time:1.000\n")

        from video_autocut.infrastructure.ffmpeg import FFmpegTools

        monkeypatch.setattr(FFmpegTools, "_run_command", mock_run)

        result = extract_frames(
            video_file, strategy="keyframe", max_frames=5, output_dir=out_dir,
        )
        assert result.strategy == ExtractionStrategy.KEYFRAME
        assert len(result.frames) == 1

    def test_invalid_strategy_raises(
        self,
        mock_settings: Settings,
        video_file: Path,
    ):
        with pytest.raises(ValueError, match="not a valid"):
            extract_frames(video_file, strategy="bogus")

    def test_default_output_dir(
        self,
        mock_settings: Settings,
        video_file: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        call_count = 0

        def mock_run(self_tools, args: list[str], **kw):
            nonlocal call_count
            if call_count == 0:
                call_count += 1
                return _completed(stdout=SAMPLE_PROBE_JSON)
            call_count += 1
            for arg in args:
                if arg.endswith(".jpg"):
                    Path(arg).parent.mkdir(parents=True, exist_ok=True)
                    Path(arg).write_bytes(b"\xff\xd8" + b"\x00" * 2000 + bytes([call_count]))
            return _completed()

        from video_autocut.infrastructure.ffmpeg import FFmpegTools

        monkeypatch.setattr(FFmpegTools, "_run_command", mock_run)

        result = extract_frames(video_file, max_frames=1)
        assert len(result.frames) == 1
        # Output should be under temp_frames_dir / <safe_video_stem>
        expected_parent = mock_settings.temp_frames_dir / _safe_dirname(video_file.stem)
        assert result.frames[0].path.parent == expected_parent

    def test_custom_output_dir(
        self,
        mock_settings: Settings,
        video_file: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        custom_dir = tmp_path / "custom"
        call_count = 0

        def mock_run(self_tools, args: list[str], **kw):
            nonlocal call_count
            if call_count == 0:
                call_count += 1
                return _completed(stdout=SAMPLE_PROBE_JSON)
            call_count += 1
            for arg in args:
                if arg.endswith(".jpg"):
                    Path(arg).parent.mkdir(parents=True, exist_ok=True)
                    Path(arg).write_bytes(b"\xff\xd8" + b"\x00" * 2000 + bytes([call_count]))
            return _completed()

        from video_autocut.infrastructure.ffmpeg import FFmpegTools

        monkeypatch.setattr(FFmpegTools, "_run_command", mock_run)

        result = extract_frames(video_file, max_frames=1, output_dir=custom_dir)
        assert result.frames[0].path.parent == custom_dir


# ---------------------------------------------------------------------------
# TestDeduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    def _make_frames(self, tmp_path: Path, sizes: list[int]) -> list[ExtractedFrame]:
        """Create frame files with specific sizes and return ExtractedFrame list."""
        frames = []
        for i, size in enumerate(sizes):
            p = tmp_path / f"frame_{i:04d}.jpg"
            p.write_bytes(b"\xff" * size)
            frames.append(ExtractedFrame(
                path=p, timestamp_seconds=float(i), frame_index=i,
            ))
        return frames

    def test_consecutive_same_size_removed(self, tmp_path: Path):
        frames = self._make_frames(tmp_path, [2000, 2000, 2000])
        unique = _remove_duplicates(frames)
        assert len(unique) == 1
        assert unique[0].timestamp_seconds == 0.0

    def test_different_sizes_kept(self, tmp_path: Path):
        frames = self._make_frames(tmp_path, [2000, 3000, 4000])
        unique = _remove_duplicates(frames)
        assert len(unique) == 3

    def test_tiny_frames_removed(self, tmp_path: Path):
        frames = self._make_frames(tmp_path, [500, 2000, 3000])
        unique = _remove_duplicates(frames)
        assert len(unique) == 2
        # The tiny frame's file should be deleted
        assert not (tmp_path / "frame_0000.jpg").exists()

    def test_reindexes_after_removal(self, tmp_path: Path):
        frames = self._make_frames(tmp_path, [2000, 2000, 3000])
        unique = _remove_duplicates(frames)
        assert len(unique) == 2
        assert unique[0].frame_index == 0
        assert unique[1].frame_index == 1

    def test_dedup_disabled(
        self,
        mock_settings: Settings,
        video_file: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """When deduplicate=False, same-size frames are kept."""
        out_dir = tmp_path / "out"
        call_count = 0

        def mock_run(self_tools, args: list[str], **kw):
            nonlocal call_count
            if call_count == 0:
                call_count += 1
                return _completed(stdout=SAMPLE_PROBE_JSON)
            call_count += 1
            for arg in args:
                if arg.endswith(".jpg"):
                    Path(arg).parent.mkdir(parents=True, exist_ok=True)
                    # All frames get the same size
                    Path(arg).write_bytes(b"\xff\xd8" + b"\x00" * 2000)
            return _completed()

        from video_autocut.infrastructure.ffmpeg import FFmpegTools

        monkeypatch.setattr(FFmpegTools, "_run_command", mock_run)

        result = extract_frames(
            video_file, max_frames=3, output_dir=out_dir, deduplicate=False,
        )
        # All 3 frames should be present since dedup is off
        assert len(result.frames) == 3

    def test_missing_file_skipped(self, tmp_path: Path):
        """Frames whose files don't exist are silently dropped."""
        frames = [
            ExtractedFrame(
                path=tmp_path / "gone.jpg", timestamp_seconds=0.0, frame_index=0,
            ),
        ]
        unique = _remove_duplicates(frames)
        assert len(unique) == 0


# ---------------------------------------------------------------------------
# TestExtractionContext
# ---------------------------------------------------------------------------


class TestExtractionContext:
    def test_cleanup_on_exit(
        self,
        mock_settings: Settings,
        video_file: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        out_dir = tmp_path / "ctx"
        call_count = 0

        def mock_run(self_tools, args: list[str], **kw):
            nonlocal call_count
            if call_count == 0:
                call_count += 1
                return _completed(stdout=SAMPLE_PROBE_JSON)
            call_count += 1
            for arg in args:
                if arg.endswith(".jpg"):
                    Path(arg).parent.mkdir(parents=True, exist_ok=True)
                    Path(arg).write_bytes(
                        b"\xff\xd8" + b"\x00" * (2000 + call_count * 100)
                    )
            return _completed()

        from video_autocut.infrastructure.ffmpeg import FFmpegTools

        monkeypatch.setattr(FFmpegTools, "_run_command", mock_run)

        with extraction_context(
            video_file, max_frames=2, output_dir=out_dir,
        ) as result:
            assert len(result.frames) == 2
            frame_paths = [f.path for f in result.frames]
            assert all(p.exists() for p in frame_paths)

        # After exiting context, images should be cleaned up
        remaining = list(out_dir.glob("*.jpg"))
        assert len(remaining) == 0

    def test_cleanup_on_exception(
        self,
        mock_settings: Settings,
        video_file: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        out_dir = tmp_path / "ctx_exc"
        call_count = 0

        def mock_run(self_tools, args: list[str], **kw):
            nonlocal call_count
            if call_count == 0:
                call_count += 1
                return _completed(stdout=SAMPLE_PROBE_JSON)
            call_count += 1
            for arg in args:
                if arg.endswith(".jpg"):
                    Path(arg).parent.mkdir(parents=True, exist_ok=True)
                    Path(arg).write_bytes(
                        b"\xff\xd8" + b"\x00" * (2000 + call_count * 100)
                    )
            return _completed()

        from video_autocut.infrastructure.ffmpeg import FFmpegTools

        monkeypatch.setattr(FFmpegTools, "_run_command", mock_run)

        with pytest.raises(RuntimeError, match="test"):
            with extraction_context(
                video_file, max_frames=2, output_dir=out_dir,
            ) as result:
                assert len(result.frames) == 2
                raise RuntimeError("test")

        remaining = list(out_dir.glob("*.jpg"))
        assert len(remaining) == 0


# ---------------------------------------------------------------------------
# TestCleanupFrames
# ---------------------------------------------------------------------------


class TestCleanupFrames:
    def test_removes_images(self, tmp_path: Path):
        (tmp_path / "frame_0001.jpg").write_bytes(b"\xff")
        (tmp_path / "frame_0002.png").write_bytes(b"\x89")
        count = cleanup_frames(tmp_path)
        assert count == 2

    def test_accepts_string_path(self, tmp_path: Path):
        (tmp_path / "frame.jpg").write_bytes(b"\xff")
        count = cleanup_frames(str(tmp_path))
        assert count == 1


# ---------------------------------------------------------------------------
# TestSafeDirname
# ---------------------------------------------------------------------------


class TestSafeDirname:
    def test_alphanumeric_unchanged(self):
        assert _safe_dirname("my_video") == "my_video"

    def test_spaces_replaced(self):
        assert _safe_dirname("my video (1)") == "my_video__1_"

    def test_hyphens_kept(self):
        assert _safe_dirname("my-video-2024") == "my-video-2024"

    def test_dots_replaced(self):
        assert _safe_dirname("my.video.2024") == "my_video_2024"
