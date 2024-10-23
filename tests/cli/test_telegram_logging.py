"""Manual test for checking Telegram trading log output.

- Need bot key and chat id
"""
import os
import logging
import sys

import pytest
from telegram_bot_logger import TelegramMessageHandler

from tradeexecutor.cli.log import setup_telegram_logging, setup_custom_log_levels

TELEGRAM_API_KEY = os.environ.get("TELEGRAM_API_KEY")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

pytestmark = pytest.mark.skipif(not(TELEGRAM_API_KEY and TELEGRAM_CHAT_ID), reason="Missing TELEGRAM_API_KEY and TELEGRAM_CHAT_ID env")


def test_telegram_logging():
    """Send a messagea to TG group chat for manual inspection.

    Run as to make sense out of this:

    .. code-block:: shell

        pytest --log-cli-level=info -k test_telegram_logging
    """
    logger = logging.getLogger(__name__)

    setup_custom_log_levels()

    # Set up basic sys.stdout config so we know if Telegram itself logs something failed
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    logger.setLevel(logging.TRADE)
    handler = setup_telegram_logging(
        TELEGRAM_API_KEY,
        TELEGRAM_CHAT_ID,
    )

    # Test crahs
    # problem_record = logging.LogRecord(
    #    'test', logging.INFO, 'path', 1,
    #    'msg', (), None)
    # handler.queue.put_nowait(problem_record)

    assert isinstance(handler, TelegramMessageHandler)
    assert handler.listener._thread is not None, "Queue thread crashed on background"
    # import ipdb ; ipdb.set_trace()
    logger.trade("Test trade output (custom logging level)")
    assert handler.listener._thread is not None, "Queue thread crashed on background"
    logger.info("test INFO")
    logger.debug("test DEBUg")
    logger.warning("test WARNING")
    logger.error("test ERROR")
    logger.fatal("test FATAL")

    logger.trade("""
    This is a multiline.
    
    Log output.
    
    Foo
    
    Bar
    
    - Foo
    - Bat
    
    https://tradingstrategy.ai
    """)

    assert handler.listener._thread is not None
    handler.close()

