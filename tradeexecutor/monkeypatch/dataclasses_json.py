"""Fix dataclasses_json to support us.

 - Always serialised and deserialise dates in the timezone-naive format

 - Add support for timedelta objects

 """

import json
import math
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Collection, Mapping
from uuid import UUID

import pandas as pd
from dataclasses_json import core

from tradeexecutor.utils.timestamp import \
    convert_and_validate_timestamp_as_float


# Mankeypatched _ExtendedEncoder.default()
def _patched_default(self, o) -> core.Json:
    result: core.Json

    if core._isinstance_safe(o, Collection):
        if core._isinstance_safe(o, Mapping):
            result = dict(o)
        else:
            result = list(o)
    elif core._isinstance_safe(o, datetime):
        #assert o.tzinfo == None, "Received a datetime with attached tz info: {o}"
        result = convert_and_validate_timestamp_as_float(o)
    elif core._isinstance_safe(o, pd.Timestamp):
        # Automatically convert pd.Timestamps to Python datetimes on write
        dt = o.to_pydatetime()
        result = convert_and_validate_timestamp_as_float(dt)
    #
    # Patch timedelta support
    #
    elif core._isinstance_safe(o, timedelta):
        result = o.total_seconds()
    elif core._isinstance_safe(o, pd.Timedelta):
        result = o.total_seconds()
    elif core._isinstance_safe(o, UUID):
        result = str(o)
    elif core._isinstance_safe(o, Enum):
        result = o.value
    elif core._isinstance_safe(o, Decimal):
        result = str(o)
    elif core._isinstance_safe(o, Decimal):
        result = str(o)
    elif isinstance(o, float):
        if math.isnan(o) or math.isinf(o):
            return None
    else:
        result = json.JSONEncoder.default(self, o)
    return result


def _patched_support_extended_types(field_type, field_value):
    if core._issubclass_safe(field_type, datetime):
        if isinstance(field_value, datetime):
            res = field_value
        else:
            # Fixed here
            # tz = datetime.now(timezone.utc).astimezone().tzinfo
            res = datetime.utcfromtimestamp(field_value)
    #
    # Add timedelta support
    #
    elif core._issubclass_safe(field_type, timedelta):
        if isinstance(field_value, timedelta):
            res = field_value
        else:
            # Fixed here
            # tz = datetime.now(timezone.utc).astimezone().tzinfo
            res = timedelta(seconds=field_value)
    elif core._issubclass_safe(field_type, Decimal):
        res = (field_value
               if isinstance(field_value, Decimal)
               else Decimal(field_value))
    elif core._issubclass_safe(field_type, UUID):
        res = (field_value
               if isinstance(field_value, UUID)
               else UUID(field_value))
    else:
        res = field_value
    return res


def patch_dataclasses_json():
    """Add monkey patched fixes to dataclasses_json package"""
    if core._support_extended_types != _patched_support_extended_types:
        core._support_extended_types = _patched_support_extended_types

    core._ExtendedEncoder.default = _patched_default

