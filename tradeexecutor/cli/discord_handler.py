"""Discord logging handler.

Based on the orignal work by Copyright (c) 2019 Trayser Cassa (MIT licensed)

See also other inspirations and sources
- https://github.com/TrayserCassa/DiscordHandler/blob/master/discord_handler/DiscordHandler.py
- https://pypi.org/project/discord-webhook/
- https://github.com/chinnichaitanya/python-discord-logger/blob/master/discord_logger/message_logger.py
"""
import logging
import os
import sys
import textwrap

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
                 avatar_url=None,
                 rate_limit_retry=True,
                 embed_line_wrap_threshold=60):

        logging.Handler.__init__(self)
        self.webhook_url = webhook_url
        self.service_name = service_name
        self.colours = colours
        self.emojis = emojis
        self.rate_limit_retry = rate_limit_retry
        self.avatar_url = avatar_url
        self.reentry_barrier = False
        self.embed_line_wrap_threshold = embed_line_wrap_threshold

    def should_format_as_code_block(self, record: logging.LogRecord, msg: str) -> bool:
        """Figure out whether we want to use code block formatting in Discord"""

        if "\n" not in msg:
            if len(msg) > self.embed_line_wrap_threshold:
                return True

        return "\n" in msg

    def clip_content(self, content: str, max_len=1900, clip_to_end=True) -> str:
        """Make sure the text fits to a Discord message.

        Discord max message length is 2000 chars.
        """
        if len(content) > max_len - 5:
            if clip_to_end:
                return "..." + content[-max_len:]
            else:
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
                avatar_url=self.avatar_url,
            )

            # About the Embed footer trick
            # https://stackoverflow.com/a/65543555/315168

            try:
                msg = self.format(record)

                colour = self.colours.get(record.levelno) or self.colours[None]
                emoji = self.emojis.get(record.levelno)

                # discord.content = msg
                if self.should_format_as_code_block(record, msg):

                    try:
                        first, remainder = msg.split("\n", maxsplit=1)
                    except ValueError:
                        first = msg
                        remainder = ""

                    max_line_length = max([len(l) for l in msg.split("\n")])
                    clipped = self.clip_content(remainder)

                    if max_line_length > self.embed_line_wrap_threshold:
                        # msg_with_bold = f"**{first}**\n```{clipped}```"
                        clipped_msg = self.clip_content(msg)
                        discord.content = f"```{clipped_msg}```"
                    else:
                        embed = DiscordEmbed(title=first, description=clipped, color=colour)
                        discord.add_embed(embed)

                    # Embeds will wrap lines quite early
                    # if True:
                    #     clipped = self.clip_content(remainder)
                    #     content = f"```\n{clipped}\n```"
                    #     if emoji:
                    #         title = f"{emoji} {first}"
                    #     else:
                    #         title = first
                    #
                    #     embed = DiscordEmbed(title=title, description=content, color=colour)
                    #     discord.add_embed(embed)
                    # else:
                    #     # Too long lines, we cannot do fancy formatting
                    #     discord.content = f"{msg}"
                else:
                    # discord.content = content
                    if emoji:
                        title = f"{emoji} {msg}"
                    else:
                        title = msg
                    embed = DiscordEmbed(title=title, color=colour)
                    discord.add_embed(embed)

                discord.execute()

            except Exception as e:
                # We cannot use handleError here, because Discord request may cause
                # infinite recursion when Discord connection fails and
                # it tries to log.
                # We fall back to writing the error to stderr
                print(f"Error from Discord logger {e}", file=sys.stderr)
                self.handleError(record)
        finally:
            self.reentry_barrier = False


if __name__ == "__main__":
    # Run a manual test
    webhook_url = os.environ["DISCORD_TRASH_WEBHOOK_URL"]
    logger = logging.getLogger()

    stream_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    discord_format = logging.Formatter("%(message)s")

    discord_handler = DiscordHandler("Happy Bot", webhook_url, emojis={}, avatar_url="https://i0.wp.com/www.theterminatorfans.com/wp-content/uploads/2012/09/the-terminator3.jpg?resize=900%2C450&ssl=1")
    #discord_handler = DiscordHandler("Happy Bot", webhook_url, emojis={})
    discord_handler.setFormatter(discord_format)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(stream_format)

    # Add the handlers to the Logger
    logger.addHandler(discord_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.DEBUG)

    logger.info("Long line of text Long line of text Long line of text Long line of text Long line of text  Long line of text Long line of text")

    # Test logging output
    # https://docs.python.org/3.9/library/textwrap.html#textwrap.dedent
    detent_text = textwrap.dedent("""\
    Test title
    
    🌲 Item 1     $200,00
    🔻 Item 2     $12,123
    """)
    logger.info(detent_text)

    long_lines_text = textwrap.dedent("""\
    A test with long lines in the content
    
    🌲 Item 1     $200,00 $200,00 $200,00 $200,00 $200,00 $200,00 $200,00 $200,00 $200,00 $200,00 $200,00 $200,00 $200,00 $200,00 $200,00 $200,00
    🔻 Item 2     $12,123 $12,123 $12,123 $12,123 $12,123 $12,123 $12,123 $12,123 $12,123 $12,123 $12,123 $12,123 $12,123 $12,123 $12,123 $12,123 $12,123 $12,123
    
            https://tradingstrategy.ai/trading-view
            https://tradingstrategy.ai/blog 
    """)
    logger.info(long_lines_text)

    logger.info("Line of text")



    logger.debug("Debug message %d %d", 1, 2)
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")

    try:
        raise RuntimeError("A bloody exception")
    except Exception as e:
        logger.exception(e)


