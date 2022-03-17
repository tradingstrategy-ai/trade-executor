"""Discord logging handler.

Based on the orignal work by Copyright (c) 2019 Trayser Cassa (MIT licensed)

See also other inspirations and sources
- https://github.com/TrayserCassa/DiscordHandler/blob/master/discord_handler/DiscordHandler.py
- https://pypi.org/project/discord-webhook/
- https://github.com/chinnichaitanya/python-discord-logger/blob/master/discord_logger/message_logger.py
"""
import logging
import os

from discord_webhook import DiscordEmbed, DiscordWebhook


DEFAULT_COLOURS = {
    None: 2040357,
    logging.CRITICAL: 14362664,  # Red
    logging.ERROR: 14362664,  # Red
    logging.WARNING: 16497928,  # Yellow
    logging.INFO: 2196944,  # Blue
    logging.DEBUG: 8947848,  # Gray
}


DEFAULT_EMOJIS = {
    None: ":loudspeaker:",
    logging.CRITICAL: ":x:",
    logging.ERROR: ":x:",
    logging.WARNING: ":warning:",
    logging.INFO: ":bell:",
    logging.DEBUG: ":microscope:",
}


class DiscordHandler(logging.Handler):
    """
    A handler class which writes logging records, appropriately formatted,
    to a Discord Server using webhooks.
    """

    def __init__(self,
                 service_name: str,
                 webhook_url: str,
                 colours=DEFAULT_COLOURS,
                 emojis=DEFAULT_EMOJIS,
                 rate_limit_retry=True):

        logging.Handler.__init__(self)
        self.webhook_url = webhook_url
        self.service_name = service_name
        self.colours = colours
        self.emojis = emojis
        self.rate_limit_retry = rate_limit_retry
        self.reentry_barrier = False

    def should_format_as_code_block(self, record: logging.LogRecord, msg: str) -> bool:
        """Figure out whether we want to use code block formatting in Discord"""
        return "\n" in msg

    def clip_content(self, content: str, max_len=1024) -> str:
        """Make sure the text fits to a Discord message."""
        if len(content) > max_len - 5:
            return content[0:max_len] + "..."
        else:
            return content

    def emit(self, record: logging.LogRecord):
        """Send a log entry to Discord."""

        if self.reentry_barrier:
            # Don't let Discord and request internals to cause logging
            # and thus infinite recursion
            return

        self.reentry_barrier = True

        try:

            discord = DiscordWebhook(
                url=self.webhook_url,
                username=self.service_name,
                rate_limit_retry=self.rate_limit_retry,
            )

            try:
                msg = self.format(record)

                colour = self.colours.get(record.levelno) or self.colours[None]
                emoji = self.emojis.get(record.levelno) or self.emojis[None]

                if self.should_format_as_code_block(record, msg):
                    first, remainder = msg.split("\n", maxsplit=1)
                    embed_content = f"{emoji} {first}"
                    clipped = self.clip_content(remainder)
                    content = f"```\n{clipped}\n```"

                    discord.content = content

                    embed = DiscordEmbed(description=embed_content, color=colour)
                    embed.set_author(name=self.service_name)
                    discord.add_embed(embed)
                else:
                    payload = emoji + " " + msg
                    embed = DiscordEmbed(description=payload, color=colour)
                    embed.set_author(name=self.service_name)
                    discord.add_embed(embed)

                discord.execute()

            except Exception:
                # We cannot use handleError here, because Discord request may cause
                # infinite recursion when Discord connection fails and
                # it tries to log.
                # We fall back to writing the error to stderr
                self.handleError(record)
        finally:
            self.reentry_barrier = False


if __name__ == "__main__":
    # Run a manual test
    webhook_url = os.environ["DISCORD_WEBHOOK_URL"]
    logger = logging.getLogger()

    stream_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    discord_format = logging.Formatter("%(message)s")

    discord_handler = DiscordHandler("test logger", webhook_url)
    discord_handler.setFormatter(discord_format)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(stream_format)

    # Add the handlers to the Logger
    logger.addHandler(discord_handler)
    logger.addHandler(stream_handler)

    logger.setLevel(logging.DEBUG)
    logger.debug("Debug message %d %d", 1, 2)
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")

    try:
        raise RuntimeError("A bloody exception")
    except Exception as e:
        logger.exception(e)


