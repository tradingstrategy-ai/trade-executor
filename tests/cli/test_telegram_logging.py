"""Manual test for checking Telegram trading log output.

- Need bot key and chat id
"""
import os
import logging

import pytest
from telegram_bot_logger import TelegramMessageHandler

from tradeexecutor.cli.log import setup_telegram_logging, setup_custom_log_levels

TELEGRAM_API_KEY = os.environ.get("TELEGRAM_API_KEY")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

pytestmark = pytest.mark.skipif(not(TELEGRAM_API_KEY and TELEGRAM_CHAT_ID), reason="Missing TELEGRAM_API_KEY and TELEGRAM_CHAT_ID env")

def test_telegram_logging():
    """Send a messagea to TG group chat for manual inspection."""
    setup_custom_log_levels()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.TRADE)
    handler = setup_telegram_logging(
        TELEGRAM_API_KEY,
        TELEGRAM_CHAT_ID,
    )
    assert isinstance(handler, TelegramMessageHandler)
    logger.trade("Test trade output")

