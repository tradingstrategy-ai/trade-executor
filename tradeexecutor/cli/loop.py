import time

import logging

logger = logging.getLogger(__name__)


def run_main_loop():
    """Runs the main loop of the strategy executor"""

    cycle = 1
    while True:
        logger.info("Starting strategy executor main loop cycle %d", cycle)
        time.sleep(10)