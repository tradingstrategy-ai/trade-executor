import logging

from tradeexecutor.cli.log import setup_custom_log_levels
from tradeexecutor.utils.ring_buffer_logging_handler import RingBufferHandler


def test_ring_buffer_logger():

    setup_custom_log_levels()

    logger = logging.getLogger(__name__)
    handler = RingBufferHandler()
    logger.addHandler()

    try:
        raise RuntimeError("Abort")
    except Exception as e:
        logger.exception(e)

    logger.warning("warning")
    logger.error("error")
    logger.debug("debug")
    logger.trade("trade")
    logger.info("Info level message")

    data = handler.export()