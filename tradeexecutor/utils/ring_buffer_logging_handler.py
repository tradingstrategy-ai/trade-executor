"""A Python logger that keeps logs in a ring buffer in-memory.

Allows any process to fetch the latest logs from the process itself.
"""

from collections import deque
from logging import Handler, LogRecord, NOTSET
from typing import Deque, List, TypedDict, Optional

from tblib import Traceback


class ExportedRecord(TypedDict):
    """One exported entry in the ring buffer logs.

    TODO: Add traceback support
    """

    #: UTC unix timestamp when this was recordded
    timestamp: float

    #: Symbolic log level, lowercase
    level: str

    #: Log message, formatted
    message: str

    exception_type: Optional[str]

    #: Log message, formatted
    traceback_data: Optional[dict]

    @staticmethod
    def get_symbolic_log_level(log_level: int) -> str:
        level = LogRecord(log_level)
        return level.name

    @staticmethod
    def export(record: LogRecord) -> "ExportedRecord":
        """Export single log record as dict."""

        # Massage data a bit
        if record.exc_info:
            et, ev, tb = record.exc_info
            traceback_data = Traceback(tb).to_dict()
            exception_type = str(et)
            message = repr(record.msg)  # This is exception message in Python developer format
        else:
            exception_type = traceback_data = None
            message = record.getMessage()  # Expand log args

        return {
            "timestamp": record.created,
            "level": record.levelname.lower(),
            "message": message,
            "exception_type": exception_type,
            "traceback_data": traceback_data,
        }


class RingBufferHandler(Handler):
    """Keep N log entries in the memory."""

    def __init__(self, level=NOTSET, buffer_size: int=2_000):
        """By default, store 2000 log messates."""
        # https://stackoverflow.com/a/4151368/315168
        
        super(RingBufferHandler, self).__init__(level)
        
        self.buffer: Deque[LogRecord] = deque([], maxlen=buffer_size)

    def emit(self, record: LogRecord):
        self.buffer.append(record)

    def export(self) -> List[ExportedRecord]:
        """Export all log entries in a format suitable for JSON serialisation.

        :return:
            Log records sorted by timestamp, for oldest to newest
        """
        records = [ExportedRecord.export(r) for r in self.buffer]
        records.sort(key=lambda r: r["timestamp"])
        return records
