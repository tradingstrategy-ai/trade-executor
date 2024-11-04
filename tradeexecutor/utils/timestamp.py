"""Timestamp  and timedelta helpers"""
import calendar
import datetime
from typing import Union

import pandas as pd


def convert_and_validate_timestamp(timestamp: Union[pd.Timestamp, datetime.datetime]) -> datetime.datetime:
    """Ensure the timestamp is converted to our internal state format.

    - Strategies deal with Pandas timestamps

    - State stores internally datetime.datetime, no timezone, all UTC
    """

    if isinstance(timestamp, pd.Timestamp):
        timestamp = timestamp.to_pydatetime()
    elif isinstance(timestamp, datetime.datetime):
        # Good
        pass
    else:
        raise RuntimeError(f"Unknown timestamp input: {timestamp}")

    assert timestamp.tzinfo is None, f"All timestamps must be naive, got {timestamp.tzinfo}"

    return timestamp


def convert_and_validate_timestamp_as_int(timestamp: Union[pd.Timestamp, datetime.datetime]) -> int:
    """Serialise timestamp as UNIX epoch seconds UTC.

    Needed when we need to have time based keys in our state,
    as the current JSON encoder cannot directly encode datetime.datetime keys.

    :return:
        UNIX UTC timestamp
    """
    timestamp = convert_and_validate_timestamp(timestamp)

    # https://stackoverflow.com/a/5499906/315168
    return int(calendar.timegm(timestamp.utctimetuple()))


def convert_and_validate_timestamp_as_float(timestamp: Union[pd.Timestamp, datetime.datetime]) -> float:
    """Serialise timestamp as UNIX epoch seconds UTC.

    Needed when we need to have time based keys in our state,
    as the current JSON encoder cannot directly encode datetime.datetime keys.

    :return:
        UNIX UTC timestamp
    """
    timestamp = convert_and_validate_timestamp(timestamp)

    # https://stackoverflow.com/a/5499906/315168
    return calendar.timegm(timestamp.utctimetuple())



def json_encode_timedelta(val: datetime.timedelta | None) -> float | None:
    """Encode timestamp objects as number of seconds passed"""

    if val is None:
        return None

    return val.total_seconds()


def json_decode_timedelta(val: float | None) -> datetime.timedelta | None:
    """Decode timestamp objects as number of seconds passed"""

    if val is None:
        return None

    return datetime.timedelta(seconds=val)
    