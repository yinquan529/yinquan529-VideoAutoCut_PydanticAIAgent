from __future__ import annotations

import asyncio
import logging

from video_autocut.agent.video_agent import create_agent
from video_autocut.logging_config import setup_logging
from video_autocut.settings import get_settings

logger = logging.getLogger(__name__)


async def async_main() -> None:
    settings = get_settings()
    setup_logging(settings.log_level)

    logger.info("VideoAutoCut starting (model=%s)", settings.model_name)

    agent = create_agent(settings.model_name)
    result = await agent.run("Hello! What can you help me with?")
    print(result.output)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
