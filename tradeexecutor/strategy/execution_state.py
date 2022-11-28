"""Execution state communicates the current trade execution loop state to the webhook."""
import datetime
import sys
from dataclasses import dataclass, field
from typing import Optional

from tblib import Traceback


@dataclass
class ExecutionState:
    """Execution state

    A singleton instance communicates the state between
    the trade executor main loop and the webhook.

    The webhook can display the exception that caused
    the trade executor crash.
    """

    #: When the execution state was updated last time
    last_refreshed_at: datetime.datetime = field(default_factory=datetime.datetime.utcnow)

    #: Is the main loop alive
    #:
    #: Set to false on the crashing exception
    executor_running: bool = True

    #: The last completed trading cycle
    completed_cycle: Optional[int] = None

    #: If the exception has crashed, serialise the exception information.
    #:
    #: This is serialised using :py:mod:`tblib`, as native
    exception: Optional[dict] = None


    @staticmethod
    def serialise_exception() -> dict:
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


