"""Function wall clock time tracking.

Provide helpers to understand which parts of the hot code paths take most of the time.
"""

import contextlib
import logging
from contextlib import contextmanager
from eth_defi.compat import native_datetime_utc_now

logger = logging.getLogger(__name__)


@contextmanager
def timed_task(task_name: str, **context_info) -> contextlib.AbstractContextManager[None]:
    """A simple context manger to measure the duration of different tasks.

    Can be later plugged in to a metrics system like Statsd / Grafana / Datadog.
    """
    started = native_datetime_utc_now()
    logger.info("Starting task %s at %s, context is %s", task_name, started, context_info)

    try:
        yield
    finally:
        duration = native_datetime_utc_now() - started
        logger.info("Ended task %s, took %s", task_name, duration)
