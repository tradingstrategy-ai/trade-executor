"""Timestamp helpers"""
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

    assert timestamp.tzinfo is None

    return timestamp


def convert_and_validate_timestamp_as_int(timestamp: Union[pd.Timestamp, datetime.datetime]) -> int:
    """Serialise timestamp as UNIX epoch seconds UTC.

    Needed when we need to have time based keys in our state,
    as the current JSON encoder cannot directly encode datetime.datetime keys/
    """
    timestamp = convert_and_validate_timestamp(timestamp)

    # https://stackoverflow.com/a/5499906/315168
    return int(calendar.timegm(timestamp.utctimetuple()))


def json_encode_timedelta(obj: datetime.timedelta) -> dict: 
    return {
        "__type__" : "timedelta",
        "days" : obj.days,
        "seconds" : obj.seconds,
        "microseconds" : obj.microseconds,
    }


def json_decode_timedelta(obj: dict) -> datetime.timedelta:
    type_ = obj.pop("__type__")
    
    return datetime.timedelta(**obj)
    