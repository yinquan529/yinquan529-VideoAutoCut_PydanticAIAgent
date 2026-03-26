# VideoAutoCut PydanticAI Agent

AI-powered video analysis assistant that extracts frames, analyzes content, and generates shooting scripts using PydanticAI and Tencent Hunyuan LLM.

## Installation Guide

### Prerequisites

- **Python 3.10+** (download from [python.org](https://www.python.org/downloads/))
- **FFmpeg** with ffprobe (download from [ffmpeg.org](https://ffmpeg.org/download.html))
- **Tencent Hunyuan API** key and base URL

### FFmpeg Setup (Windows)

1. Download the FFmpeg release build from https://ffmpeg.org/download.html
2. Extract to a folder, e.g. `C:\ffmpeg`
3. Add `C:\ffmpeg\bin` to your system PATH, or set `FFMPEG_PATH` / `FFPROBE_PATH` in your `.env` file

To verify FFmpeg is available:

```
ffmpeg -version
ffprobe -version
```

### Project Setup

```bash
# Clone the repository
git clone https://github.com/yinquan529/yinquan529-VideoAutoCut_PydanticAIAgent.git
cd yinquan529-VideoAutoCut_PydanticAIAgent

# Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# Install the package in editable mode with dev dependencies
pip install -e ".[dev]"
```

### Configuration

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

Required settings in `.env`:

```ini
HUNYUAN_API_KEY=your-hunyuan-api-key-here
HUNYUAN_BASE_URL=https://api.hunyuan.cloud.tencent.com/v1
```

Optional settings (shown with defaults):

```ini
MODEL_NAME=openai:hunyuan-turbos-latest
FFMPEG_PATH=ffmpeg
FFPROBE_PATH=ffprobe
TEMP_FRAMES_DIR=.video_autocut/frames
OUTPUT_DIR=.video_autocut/output
LOG_LEVEL=INFO
MAX_RETRIES=3
REQUEST_TIMEOUT_SECONDS=60
```

### Running

```bash
video-autocut
```

### Running Tests

```bash
pytest tests/ -v
```

Lint checks:

```bash
ruff check src/ tests/
```

## Construction Overview

### Project Layout

```
src/video_autocut/
├── app/                    # Application entry point
│   └── main.py             # CLI startup, settings validation, agent invocation
├── agent/                  # PydanticAI agent definitions
│   └── video_agent.py      # Agent with system prompt for video analysis
├── domain/                 # Pure data layer (no I/O)
│   ├── enums.py            # ExtractionStrategy, ScriptType, ShotType, SceneType, ErrorCategory
│   ├── models.py           # Internal models (VideoMetadata, ExtractedFrame, etc.)
│   └── script_models.py   # LLM-output models (FrameAnalysis, ShootingScript, etc.)
├── infrastructure/         # External system integrations
│   ├── exceptions.py       # FFmpeg error hierarchy
│   └── ffmpeg.py           # FFmpegTools: subprocess wrapper for ffmpeg/ffprobe
├── tools/                  # Agent-callable tool functions
│   └── frame_extraction.py # High-level frame extraction facade
├── settings.py             # pydantic-settings config with .env support
└── logging_config.py       # Logging setup
```

### Architecture Layers

```
┌─────────────────────────────────────────────┐
│  app/main.py                                │
│  CLI entry point, startup validation        │
├─────────────────────────────────────────────┤
│  agent/video_agent.py                       │
│  PydanticAI Agent with system prompt        │
├─────────────────────────────────────────────┤
│  tools/frame_extraction.py                  │
│  Agent-tool-friendly facade                 │
│  - extract_frames()                         │
│  - extraction_context()  (auto-cleanup)     │
│  - cleanup_frames()                         │
│  - Duplicate-frame detection                │
├─────────────────────────────────────────────┤
│  infrastructure/ffmpeg.py                   │
│  FFmpegTools class                          │
│  - validate_video, probe_video              │
│  - extract_frames (uniform/scene/keyframe)  │
│  - Single subprocess mock boundary          │
├─────────────────────────────────────────────┤
│  domain/                                    │
│  Pure Pydantic models and enums             │
│  - Internal models (use Path)               │
│  - LLM-output models (frozen, str only)     │
├─────────────────────────────────────────────┤
│  settings.py                                │
│  pydantic-settings with .env file support   │
└─────────────────────────────────────────────┘
```

### Key Design Decisions

**Dual model system.** Internal models in `domain/models.py` use `Path` for file references and are mutable. LLM-output models in `domain/script_models.py` are frozen (`ConfigDict(frozen=True)`), use `str` instead of `Path`, and include `Field(description=...)` on every field so they work as PydanticAI `output_type` schemas.

**Single subprocess boundary.** All ffmpeg/ffprobe calls route through `FFmpegTools._run_command()`. Tests monkeypatch this single method per-instance, avoiding global `subprocess.run` patches.

**Facade over strategies.** Frame extraction supports three strategies (uniform, scene_change, keyframe) via the `ExtractionStrategy` enum. The dispatch logic lives in `FFmpegTools.extract_frames()`. The `tools/frame_extraction.py` facade wraps this with a simple API that accepts plain strings, constructs domain objects automatically, and adds deduplication.

**Settings validation at startup.** `validate_settings()` collects all configuration errors before raising a single `SettingsValidationError` with a numbered list, rather than failing on the first missing value.

**Windows-native.** All file paths use `pathlib.Path`. All subprocess calls use argument lists (never `shell=True`). No POSIX-specific assumptions.

### Frame Extraction Strategies

| Strategy | Method | Description |
|---|---|---|
| `uniform` | Per-frame `ffmpeg -ss` | Evenly-spaced timestamps across video duration |
| `scene_change` | `-vf select='gt(scene,0.3)'` | Frames where visual content changes significantly |
| `keyframe` | `-vf select='eq(pict_type,PICT_TYPE_I)'` | I-frames (intra-coded, highest quality) |

Usage from agent tools:

```python
from video_autocut.tools import extract_frames, extraction_context

# Simple extraction
result = extract_frames("input.mp4", strategy="scene_change", max_frames=20)
for frame in result.frames:
    print(frame.path, frame.timestamp_seconds)

# With automatic cleanup
with extraction_context("input.mp4", strategy="keyframe") as result:
    for frame in result.frames:
        analyze(frame.path)
# frame files are deleted here
```

### Test Suite

All tests run without real ffmpeg by monkeypatching `_run_command`:

| Test File | Coverage |
|---|---|
| `test_settings.py` | Settings loading, validation, error collection, directory creation |
| `test_domain_models.py` | Model construction, serialization, enum values, frozen immutability |
| `test_ffmpeg.py` | Availability checks, video probing, validation, frame extraction, cleanup |
| `test_frame_extraction_tool.py` | Facade API, strategy dispatch, deduplication, context manager cleanup |
