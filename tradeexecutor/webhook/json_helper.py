import json
import math
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Collection, Mapping
from uuid import UUID

import pandas as pd
from dataclasses_json.core import Json
from dataclasses_json.utils import _isinstance_safe


class NaNToNullEncoder(json.JSONEncoder):
    """NaN fixed dataclasses-json JSON encoder"""

    def default(self, o) -> Json:
        """NaN + timedelta added"""
        result: Json
        if _isinstance_safe(o, Collection):
            if _isinstance_safe(o, Mapping):
                result = dict(o)
            else:
                result = list(o)
        elif _isinstance_safe(o, datetime):
            result = o.timestamp()
        elif _isinstance_safe(o, UUID):
            result = str(o)
        elif _isinstance_safe(o, Enum):
            result = o.value
        elif _isinstance_safe(o, Decimal):
            result = str(o)
        elif _isinstance_safe(o, timedelta):
            result = o.total_seconds()
        elif _isinstance_safe(o, pd.Timedelta):
            result = o.total_seconds()
        else:
            result = json.JSONEncoder.default(self, o)
        return result

    def encode(self, obj):
        # Replace NaN/Inf before encoding
        return super().encode(self._sanitize(obj))

    def _sanitize(self, obj):
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
        elif isinstance(obj, dict):
            return {k: self._sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize(v) for v in obj]
        return obj
