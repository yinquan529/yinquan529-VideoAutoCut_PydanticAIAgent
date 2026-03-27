"""Markdown renderer for :class:`ShootingScript` objects.

Converts a typed shooting script into a human-readable Markdown document
suitable for review, export, or further editing.

The renderer is a pure function — no I/O, no side effects, no LLM calls.
"""

from __future__ import annotations

from video_autocut.domain.script_models import (
    MusicCue,
    NarrationCue,
    SceneDefinition,
    ShootingScript,
    ShotDefinition,
)


def _fmt_time(seconds: float) -> str:
    """Format seconds as ``MM:SS``."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _fmt_range(start: float, end: float) -> str:
    """Format a timecode range as ``[MM:SS – MM:SS]``."""
    return f"[{_fmt_time(start)} – {_fmt_time(end)}]"


def render_shot(shot: ShotDefinition) -> str:
    """Render a single shot as a Markdown list item."""
    time_range = _fmt_range(shot.start_seconds, shot.end_seconds)
    line = (
        f"- **Shot {shot.shot_number}** {time_range} "
        f"| {shot.shot_type.value.replace('_', ' ')} | "
        f"{shot.description}"
    )
    extras: list[str] = []
    if shot.camera_movement and shot.camera_movement != "static":
        extras.append(f"Camera: {shot.camera_movement}")
    if shot.framing_notes:
        extras.append(f"Framing: {shot.framing_notes}")
    if extras:
        line += "\n  " + " · ".join(extras)
    return line


def render_scene(scene: SceneDefinition) -> str:
    """Render a scene block with its shots."""
    time_range = _fmt_range(scene.start_seconds, scene.end_seconds)
    header = (
        f"### Scene {scene.scene_number}: {scene.title}\n\n"
        f"**{scene.location}** · {scene.scene_type.value} · "
        f"Mood: {scene.mood} · {time_range}\n"
    )
    if scene.scene_direction:
        header += f"\n> {scene.scene_direction}\n"

    shots_md = "\n".join(render_shot(s) for s in scene.shots)
    return f"{header}\n{shots_md}"


def render_narration(cues: list[NarrationCue]) -> str:
    """Render narration cues as a Markdown table."""
    if not cues:
        return ""
    rows = ["| Time | Speaker | Text | Tone |", "| --- | --- | --- | --- |"]
    for cue in cues:
        time_range = _fmt_range(cue.start_seconds, cue.end_seconds)
        tone = cue.tone or "—"
        rows.append(f"| {time_range} | {cue.speaker} | {cue.text} | {tone} |")
    return "## Narration\n\n" + "\n".join(rows)


def render_music(cues: list[MusicCue]) -> str:
    """Render music cues as a Markdown table."""
    if not cues:
        return ""
    rows = ["| Time | Genre | Mood | Tempo | Notes |", "| --- | --- | --- | --- | --- |"]
    for cue in cues:
        time_range = _fmt_range(cue.start_seconds, cue.end_seconds)
        notes = cue.notes or "—"
        rows.append(
            f"| {time_range} | {cue.genre} | {cue.mood} | {cue.tempo} | {notes} |"
        )
    return "## Music\n\n" + "\n".join(rows)


def render_script(script: ShootingScript) -> str:
    """Render a complete shooting script as a Markdown document.

    Args:
        script: A :class:`ShootingScript` instance.

    Returns:
        A multi-section Markdown string ready for display or file export.
    """
    duration = _fmt_time(script.target_duration_seconds)
    header = (
        f"# {script.title}\n\n"
        f"**Type:** {script.script_type.value} · "
        f"**Duration:** {duration}\n\n"
        f"## Synopsis\n\n"
        f"{script.synopsis}\n"
    )

    scenes_md = "\n\n".join(render_scene(s) for s in script.scenes)
    scenes_section = f"\n\n## Scenes\n\n{scenes_md}" if script.scenes else ""

    narration_section = ""
    if script.narration_cues:
        narration_section = "\n\n" + render_narration(script.narration_cues)

    music_section = ""
    if script.music_cues:
        music_section = "\n\n" + render_music(script.music_cues)

    notes_section = ""
    if script.production_notes:
        notes_section = f"\n\n## Production Notes\n\n{script.production_notes}"

    return (
        header
        + scenes_section
        + narration_section
        + music_section
        + notes_section
        + "\n"
    )
