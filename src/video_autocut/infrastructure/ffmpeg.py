from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class FFmpegRunner:
    def __init__(self, ffmpeg_path: Path | None = None) -> None:
        self.ffmpeg_path = ffmpeg_path or Path("ffmpeg")

    def check_available(self) -> bool:
        try:
            result = subprocess.run(
                [str(self.ffmpeg_path), "-version"],
                capture_output=True,
                text=True,
            )
            return result.returncode == 0
        except FileNotFoundError:
            logger.warning("FFmpeg not found at %s", self.ffmpeg_path)
            return False
