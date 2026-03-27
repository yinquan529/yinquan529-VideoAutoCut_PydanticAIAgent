"""Windows-friendly CLI for the VideoAutoCut script assistant.

Built on Typer (which wraps Click) with Rich console output for
progress messages and coloured summaries.

Two sub-commands:

* ``generate`` — deterministic pipeline: analyse video → generate
  script → save outputs.  No LLM "conversation" — each CLI flag maps
  directly to a pipeline parameter.

* ``chat`` — interactive mode backed by the orchestrator agent.  The
  user types natural-language prompts and the agent decides which tools
  to call.

Output file naming
------------------
``{stem}_{script_type}_{timestamp}.{ext}``

where *stem* is the video filename without extension, *script_type* is
the chosen genre, *timestamp* is ``YYYYMMDD_HHMMSS``, and *ext* is one
of ``md``, ``json``, or ``txt``.
"""

from __future__ import annotations

import asyncio
import sys
import time
from datetime import datetime
from enum import Enum
from pathlib import Path

import typer
from rich.console import Console

from video_autocut.agent.orchestrator import VideoDeps, create_orchestrator
from video_autocut.infrastructure.ffmpeg import VALID_VIDEO_EXTENSIONS, FFmpegTools
from video_autocut.settings import (
    SettingsValidationError,
    validate_settings,
)
from video_autocut.tools.script_generator import generate_script
from video_autocut.tools.script_renderer import render_script
from video_autocut.tools.video_analysis import analyze_video

app = typer.Typer(
    name="video-autocut",
    help="AI video analysis assistant — analyse videos and generate shooting scripts.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

console = Console(stderr=True)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class OutputFormat(str, Enum):
    """Supported output file formats."""

    md = "md"
    json = "json"
    txt = "txt"


def _validate_video_path(path: Path) -> Path:
    """Ensure the file exists and has a supported video extension."""
    if not path.exists():
        console.print(f"[red]Error:[/red] File not found: {path}")
        raise typer.Exit(1)
    if path.suffix.lower() not in VALID_VIDEO_EXTENSIONS:
        console.print(
            f"[red]Error:[/red] Unsupported format '{path.suffix}'.\n"
            f"Supported: {', '.join(sorted(VALID_VIDEO_EXTENSIONS))}"
        )
        raise typer.Exit(1)
    return path


def _build_output_path(
    video_path: Path,
    script_type: str,
    output_dir: Path,
    fmt: OutputFormat,
) -> Path:
    """Build a timestamped output filename.

    Pattern: ``{stem}_{script_type}_{YYYYMMDD_HHMMSS}.{ext}``
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = video_path.stem
    name = f"{stem}_{script_type}_{stamp}.{fmt.value}"
    return output_dir / name


def _save_output(
    content: str,
    path: Path,
) -> None:
    """Write content to *path*, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _init_settings():
    """Validate and return settings, printing friendly errors on failure."""
    try:
        return validate_settings()
    except SettingsValidationError as exc:
        console.print(f"[red]Configuration error:[/red]\n{exc}")
        console.print(
            "\n[dim]Set HUNYUAN_API_KEY and HUNYUAN_BASE_URL in your "
            "environment or .env file.[/dim]"
        )
        raise typer.Exit(1) from exc


# ---------------------------------------------------------------------------
# generate command
# ---------------------------------------------------------------------------


@app.command()
def generate(
    video: Path = typer.Argument(
        ...,
        help="Path to the local video file.",
        exists=False,  # we do our own validation for better messages
    ),
    script_type: str = typer.Option(
        "documentary",
        "--type", "-t",
        help="Script type: documentary, promotional, tutorial, social_media, narrative.",
    ),
    duration: float = typer.Option(
        0.0,
        "--duration", "-d",
        help="Target duration in seconds (0 = infer from source).",
    ),
    audience: str = typer.Option(
        "",
        "--audience", "-a",
        help="Target audience (e.g. 'developers', 'general public').",
    ),
    style: str = typer.Option(
        "",
        "--style", "-s",
        help="Visual / editorial style guidance.",
    ),
    prompt: str = typer.Option(
        "",
        "--prompt", "-p",
        help="Custom prompt or focus hint for the analysis.",
    ),
    output_dir: Path = typer.Option(
        None,
        "--output-dir", "-o",
        help="Directory for output files (default: .video_autocut/output).",
    ),
    fmt: OutputFormat = typer.Option(
        OutputFormat.md,
        "--format", "-f",
        help="Output format: md, json, txt.",
    ),
    max_frames: int = typer.Option(
        10,
        "--max-frames",
        help="Maximum frames to extract and analyse (1-200).",
        min=1,
        max=200,
    ),
    strategy: str = typer.Option(
        "uniform",
        "--strategy",
        help="Frame extraction strategy: uniform, scene_change, keyframe.",
    ),
) -> None:
    """Analyse a video and generate a shooting script.

    \b
    Examples:
      video-autocut generate video.mp4
      video-autocut generate video.mp4 --type promotional --audience "teens"
      video-autocut generate clip.mov -t tutorial -d 120 -f json
      video-autocut generate film.mkv --strategy scene_change --max-frames 20
    """
    video = _validate_video_path(video)

    settings = _init_settings()

    if output_dir is None:
        output_dir = settings.output_dir

    console.print(f"[bold]Video:[/bold]  {video}")
    console.print(f"[bold]Type:[/bold]   {script_type}")
    if duration > 0:
        console.print(f"[bold]Duration:[/bold] {duration:.0f}s")
    console.print()

    asyncio.run(
        _run_generate(
            video=video,
            script_type=script_type,
            duration=duration,
            audience=audience,
            style=style,
            prompt=prompt,
            output_dir=output_dir,
            fmt=fmt,
            max_frames=max_frames,
            strategy=strategy,
            settings=settings,
        )
    )


async def _run_generate(
    *,
    video: Path,
    script_type: str,
    duration: float,
    audience: str,
    style: str,
    prompt: str,
    output_dir: Path,
    fmt: OutputFormat,
    max_frames: int,
    strategy: str,
    settings,
) -> None:
    """Async implementation of the generate pipeline."""
    # Phase 1: analyse video
    console.print("[cyan]Analysing video...[/cyan]")
    t0 = time.monotonic()

    try:
        analysis = await analyze_video(
            video,
            strategy=strategy,
            max_frames=max_frames,
            user_prompt=prompt,
            settings=settings,
        )
    except Exception as exc:
        console.print(f"[red]Analysis failed:[/red] {exc}")
        raise typer.Exit(1) from exc

    analysis_secs = time.monotonic() - t0
    console.print(
        f"  Frames analysed: {analysis.stats.frames_analyzed} "
        f"in {analysis_secs:.1f}s"
    )

    if analysis.errors:
        for err in analysis.errors:
            console.print(f"  [yellow]Warning:[/yellow] {err.message}")

    if not analysis.frame_analyses:
        console.print("[red]No frames could be analysed. Aborting.[/red]")
        raise typer.Exit(1)

    # Phase 2: generate script
    console.print("[cyan]Generating script...[/cyan]")
    t1 = time.monotonic()

    try:
        gen_result = await generate_script(
            analysis,
            script_type=script_type,
            target_duration=duration,
            target_audience=audience,
            style=style,
            emphasis=prompt,
            settings=settings,
        )
    except Exception as exc:
        console.print(f"[red]Script generation failed:[/red] {exc}")
        raise typer.Exit(1) from exc

    gen_secs = time.monotonic() - t1

    if gen_result.error:
        console.print(
            f"[red]Script generation error:[/red] {gen_result.error.message}"
        )
        raise typer.Exit(1)

    if gen_result.script is None:
        console.print("[red]Script generation returned no output.[/red]")
        raise typer.Exit(1)

    console.print(f"  Script generated in {gen_secs:.1f}s")

    # Phase 3: render + save
    script = gen_result.script

    if fmt == OutputFormat.json:
        content = script.model_dump_json(indent=2)
    elif fmt == OutputFormat.md:
        content = render_script(script)
    else:
        content = render_script(script)

    out_path = _build_output_path(video, script_type, output_dir, fmt)
    _save_output(content, out_path)

    # Summary
    console.print()
    console.print("[bold green]Done![/bold green]")
    console.print(f"  Script: [bold]{script.title}[/bold]")
    console.print(f"  Scenes: {len(script.scenes)}")
    total_shots = sum(len(s.shots) for s in script.scenes)
    console.print(f"  Shots:  {total_shots}")
    console.print(f"  Output: {out_path}")

    total_secs = analysis_secs + gen_secs
    tokens = analysis.stats.total_prompt_tokens + analysis.stats.total_completion_tokens
    if gen_result.token_usage:
        tokens += gen_result.token_usage.total_tokens
    console.print(f"  Time:   {total_secs:.1f}s  |  Tokens: {tokens}")


# ---------------------------------------------------------------------------
# chat command
# ---------------------------------------------------------------------------


@app.command()
def chat(
    prompt: str = typer.Argument(
        ...,
        help="Your question or instruction for the assistant.",
    ),
) -> None:
    """Send a single prompt to the VideoAutoCut assistant.

    The assistant has access to all video tools and will decide which
    to call based on your prompt.

    \b
    Examples:
      video-autocut chat "What tools do you have?"
      video-autocut chat "Analyse video.mp4 and describe the content"
    """
    settings = _init_settings()
    asyncio.run(_run_chat(prompt, settings))


async def _run_chat(prompt: str, settings) -> None:
    """Async implementation of the chat command."""
    agent = create_orchestrator(settings)
    deps = VideoDeps(
        settings=settings,
        ffmpeg=FFmpegTools(settings.ffmpeg_path, settings.ffprobe_path),
    )

    console.print("[cyan]Thinking...[/cyan]")
    result = await agent.run(prompt, deps=deps)

    # Print to stdout (not stderr via console)
    sys.stdout.write(result.output + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point registered in pyproject.toml."""
    app()
