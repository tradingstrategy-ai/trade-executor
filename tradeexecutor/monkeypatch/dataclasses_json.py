"""Fix dataclasses_json to always serialised and deserialise dates in the timezone-naive format."""
from datetime import datetime
from datetime import timezone
from decimal import Decimal
from uuid import UUID

from dataclasses_json import core


def _patched_support_extended_types(field_type, field_value):
    if core._issubclass_safe(field_type, datetime):
        if isinstance(field_value, datetime):
            res = field_value
        else:
            # Fixed here
            # tz = datetime.now(timezone.utc).astimezone().tzinfo
            res = datetime.fromtimestamp(field_value, tz=None)
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

