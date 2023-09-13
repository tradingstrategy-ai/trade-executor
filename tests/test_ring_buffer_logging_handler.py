"""Test in-memory logging solution."""

import json
import logging
from pprint import pprint

from tradeexecutor.cli.log import setup_custom_log_levels
from tradeexecutor.utils.ring_buffer_logging_handler import RingBufferHandler


def test_ring_buffer_logger():
    """Test our in-house in-memory logging solution."""
    setup_custom_log_levels()

    logger = logging.getLogger(__name__)
    handler = RingBufferHandler(level=logging.DEBUG)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    try:
        raise RuntimeError("Big Bada Boom")
    except Exception as e:
        logger.exception(e)

    logger.warning("warning")
    logger.error("error")
    logger.debug("debug")
    logger.trade("trade")  # Check for custom log level support
    logger.info("Info level message")
    logger.info("Foo %d, bar %d", 1, 2)

    data = handler.export()

    # Check we get the correct order
    # pprint(data)
    assert data[-1]["level"] == "info"
    assert data[-1]["message"] == "Foo 1, bar 2"

    assert data[0]["level"] == "error"
    assert data[0]["message"] == "RuntimeError('Big Bada Boom')"
    assert data[0]["level_number"] == 40
    #assert data[0]["formatted_data"] == ['NoneType: None\n']
    
    
    x = data[0]["formatted_data"]
    assert type(x) == list
    assert len(x) == 3
    assert x[0] == 'Traceback (most recent call last):\n'
    
    # Check that we can serialise JSON
    json.dumps(data)

    logger.removeHandler(handler)
