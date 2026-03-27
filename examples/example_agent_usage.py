"""Example: using the VideoAutoCut orchestrator agent.

Before running, set the required environment variables (or create a .env file):

    HUNYUAN_API_KEY=your-key-here
    HUNYUAN_BASE_URL=https://api.hunyuan.cloud.tencent.com/v1

Usage:
    python examples/example_agent_usage.py
"""

from __future__ import annotations

import asyncio

from video_autocut.agent.orchestrator import (
    VideoDeps,
    create_orchestrator,
    run_agent,
)
from video_autocut.infrastructure.ffmpeg import FFmpegTools
from video_autocut.settings import validate_settings


async def one_shot() -> None:
    """Simplest usage: one prompt, one answer."""
    response = await run_agent(
        "What tools do you have available? "
        "Summarize each one in a sentence."
    )
    print("=== One-shot response ===")
    print(response)


async def multi_turn() -> None:
    """Multi-turn conversation using the agent directly."""
    settings = validate_settings()
    agent = create_orchestrator(settings)
    deps = VideoDeps(
        settings=settings,
        ffmpeg=FFmpegTools(settings.ffmpeg_path, settings.ffprobe_path),
    )

    # Turn 1: ask about a video
    r1 = await agent.run("Get the metadata for sample.mp4", deps=deps)
    print("=== Turn 1 ===")
    print(r1.output)

    # Turn 2: follow up with analysis (message_history carries context)
    r2 = await agent.run(
        "Now analyze the video content, focusing on outdoor scenes.",
        deps=deps,
        message_history=r1.all_messages(),
    )
    print("\n=== Turn 2 ===")
    print(r2.output)

    # Turn 3: generate a script from the analysis
    r3 = await agent.run(
        "Generate a documentary shooting script for this video.",
        deps=deps,
        message_history=r2.all_messages(),
    )
    print("\n=== Turn 3 ===")
    print(r3.output)


async def main() -> None:
    print("--- One-shot example ---\n")
    await one_shot()

    print("\n\n--- Multi-turn example ---\n")
    await multi_turn()


if __name__ == "__main__":
    asyncio.run(main())
