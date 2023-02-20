"""Fix dataclasses_json to always serialised and deserialise dates in the timezone-naive format."""
import json
from datetime import datetime, timedelta
from datetime import timezone
from decimal import Decimal
from enum import Enum
from typing import Collection, Mapping
from uuid import UUID

from dataclasses_json import core



def _patched_default(self, o) -> core.Json:
    result: core.Json
    if core._isinstance_safe(o, Collection):
        if core._isinstance_safe(o, Mapping):
            result = dict(o)
        else:
            result = list(o)
    elif core._isinstance_safe(o, datetime):
        result = o.timestamp()
    # Patch timedelta support
    elif core._isinstance_safe(o, timedelta):
        result = o.total_seconds()
    elif core._isinstance_safe(o, UUID):
        result = str(o)
    elif core._isinstance_safe(o, Enum):
        result = o.value
    elif core._isinstance_safe(o, Decimal):
        result = str(o)
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
            res = datetime.fromtimestamp(field_value, tz=None)
    # Add timedelta support
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
    if core._support_extended_types != _patched_support_extended_types:
        core._support_extended_types = _patched_support_extended_types

    core._ExtendedEncoder.default = _patched_default

