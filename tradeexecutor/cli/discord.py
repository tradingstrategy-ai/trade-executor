"""Send images to the Discord logger."""

import logging
import os
from io import BytesIO
from typing import Optional

from discord_logging.handler import DiscordHandler
from discord_webhook import DiscordWebhook

from tradeexecutor.cli.log import setup_discord_logging, setup_logging


logger = logging.getLogger(__name__)


def get_discord_logging_handler() -> Optional[DiscordHandler]:
    """See" log.py

    https://stackoverflow.com/a/3630800/315168
    """
    root = logging.getLogger()
    for handler in root.handlers:
        if isinstance(handler, DiscordHandler):
            return handler

    return None


def post_logging_discord_image(image: bytes):
    """Post an image to Discord to the Discord logging channel.

    It's a bit broken abstraction here, as we do not specifically
    pass the Discord webhook data around. We will just grab
    it from the active logger and post using those credentials.

    If no Discord logger is active do nothing.
    """

    assert isinstance(image, bytes), f"Expected bytes, got {type(image)}"

    handler = get_discord_logging_handler()
    if not handler:
        return

    discord = DiscordWebhook(
        url=handler.webhook_url,
        username=handler.service_name,
        rate_limit_retry=handler.rate_limit_retry,
        avatar_url=handler.avatar_url,
        timeout=60,
    )

    discord.add_file(image, "strategy-state.png")

    try:
        discord.execute()
    except Exception:
        logger.error("Failed to post image to Discord", exc_info=True)


if __name__ == "__main__":
    # Manually test image posting
    setup_logging()
    setup_discord_logging("Imaeg log test", os.environ["DISCORD_WEBHOOK_URL"])

    with open("/tmp/test-image.png", "rb") as inp:
        image_data = inp.read()
        post_logging_discord_image(image_data)

