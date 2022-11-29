"""Execution state communicates the current trade execution loop state to the webhook."""
import datetime
import sys
from dataclasses import dataclass, field
from typing import Optional, TypedDict

import dataclasses_json
from dataclasses_json import dataclass_json
from tblib import Traceback


class ExceptionData(TypedDict):
    """Serialise exception data using tblib.

    TODO: Figure out what can go here, because tblib does not provide typing.
    """
    exception_message: str
    tb_next: Optional[dict]
    tb_lineno: int



@dataclass
class LatestStateVisualisation:
    """The last visualisation of the strategy state."""

    #: When the execution state was updated last time
    last_refreshed_at: datetime.datetime = field(default_factory=datetime.datetime.utcnow)

    #: 512 x 512 image
    small_image_data: Optional[bytes] = None

    #: 1920 x 1920 image
    large_image_data: Optional[bytes] = None


@dataclass
class ExecutionState:
    """Execution state

    A singleton instance communicates the state between
    the trade executor main loop and the webhook.

    The webhook can display the exception that caused
    the trade executor crash.

    Partially returned by different endpoints in API

    - /status

    - /source

    - /visualisation
    """

    #: When the execution state was updated last time
    last_refreshed_at: datetime.datetime = field(default_factory=datetime.datetime.utcnow)

    #: Is the main loop alive
    #:
    #: Set to false on the crashing exception
    executor_running: bool = True

    #: The last completed trading cycle
    completed_cycle: Optional[int] = None

    #: If the exception has crashed, serialise the exception information here.
    #:
    #: See :py:meth:`serialise_exception`
    exception: Optional[ExceptionData] = None

    #: The strategy source code.
    #:
    #: TODO: Move this to somewhere else long term.
    #: Use /source API endpoint to get this.
    source_code: Optional[str] = None

    #: The strategy visualisation images
    #:
    visualisation: Optional[LatestStateVisualisation] = None

    @staticmethod
    def serialise_exception() -> ExceptionData:
        """Serialised the latest raised Python exception.

        Uses :py:mod:`tblib` to convert the Python traceback
        to something that is serialisable.
        """
        et, ev, tb = sys.exc_info()
        tb = Traceback(tb)
        data = tb.to_dict()

        # tblib loses the actual formatted exception message
        data["exception_message"] = str(ev)
        return data

    def set_fail(self):
        """Set the trade-executor main loop to a failed state."""
        self.exception = self.serialise_exception()
        self.last_refreshed_at = datetime.datetime.utcnow()
        self.executor_running = False

    def update_complete_cycle(self, cycle: int):
        self.last_refreshed_at = datetime.datetime.utcnow()
        self.completed_cycle = cycle
