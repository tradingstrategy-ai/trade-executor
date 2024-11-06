"""send-log-message command

"""

import logging
import os
from pathlib import Path
from typing import Optional

import typer


from . import shared_options
from .app import app
from ..bootstrap import prepare_executor_id
from ..log import setup_logging, setup_discord_logging, setup_logstash_logging, setup_file_logging, setup_telegram_logging


logger = logging.getLogger(__name__)


@app.command()
def send_log_message(
    id: str = shared_options.id,
    name: Optional[str] = shared_options.name,
    strategy_file: Path = shared_options.strategy_file,
    # Logging
    log_level: str = shared_options.log_level,
    discord_webhook_url: Optional[str] = typer.Option(None, envvar="DISCORD_WEBHOOK_URL", help="Discord webhook URL for notifications"),
    logstash_server: Optional[str] = typer.Option(None, envvar="LOGSTASH_SERVER", help="LogStash server hostname where to send logs"),
    file_log_level: Optional[str] = typer.Option("info", envvar="FILE_LOG_LEVEL", help="Log file log level. The default log file is logs/id.log."),
    telegram_api_key: Optional[str] = typer.Option(None, envvar="TELEGRAM_API_KEY", help="Telegram bot API key for Telegram logging"),
    telegram_chat_id: Optional[str] = typer.Option(None, envvar="TELEGRAM_CHAT_ID", help="Telegram chat id where the bot will log. Group chats have negative id. Bot must receive command /start in the group chat before it can send messages there."),
):
    """Send a test messages across all configured loggers.

    Allows to manually test if all loggers are correctly configured.
    """
    global logger

    # Guess id from the strategy file
    id = prepare_executor_id(id, strategy_file)

    # We always need a name-*-
    if not name:
        if strategy_file:
            name = os.path.basename(strategy_file)
        else:
            name = "Unnamed backtest"

    # Make sure unit tests run logs do not get polluted
    # Don't touch any log levels, but
    # make sure we have logger.trading() available when
    # log_level is "disabled"
    logger = setup_logging(
        log_level,
        in_memory_buffer=True,
        enable_trade_high=True,
    )

    if discord_webhook_url:
        logger.info("Configuring Discord logging")
        setup_discord_logging(
            name,
            webhook_url=discord_webhook_url,
        )

    if telegram_api_key:
        logger.info("Configuring Telegram logging")
        setup_telegram_logging(
            telegram_api_key,
            telegram_chat_id,
        )

    if logstash_server:
        logger.info("Configuring Logstash logging")
        logger.info("Enabling Logstash logging to %s", logstash_server)
        setup_logstash_logging(
            logstash_server,
            f"executor-{id}",  # Always prefix logged with executor id
            quiet=False,
        )
    else:
        logger.info("Logstash logging disabled")

    logger.info("Configuring file logging")
    setup_file_logging(
        f"logs/{id}.log",
        file_log_level,
        http_logging=True,
    )

    logger.info(f"{name}: INFO log test message")
    logger.trade_high(f"{name}: TRADE (HIGH) log test message")

