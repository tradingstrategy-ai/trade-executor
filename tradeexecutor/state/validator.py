"""Stateful data validation.

Avoid non-JSON supported types to slipping into to the state.
We have some common culprits we want to catch.

Any error message contains tree presentation of the state,
so you can easily locate any values that are bad, unlike with :py:mod:`json`.
"""
import datetime
import math
from decimal import Decimal
from enum import Enum
from json.encoder import INFINITY
from types import NoneType
from typing import Type

import numpy as np
import pandas as pd

from tradeexecutor.state.state import State


class BadStateData(Exception):
    """Having something we do not support in the state."""

ALLOWED_KEY_TYPES = (
    float,
    int,
    str
)

#: Types we know we can safely pass to JSON serialisation
ALLOWED_VALUE_TYPES = (
    dict,
    list,
    float,
    int,
    str,
    tuple,
    NoneType,
    Enum,  # Supported by dadtaclasses_json
    datetime.datetime,  # Supported by dadtaclasses_json
    Decimal,  # Supported by dadtaclasses_json
    datetime.timedelta,
    pd.Timestamp,
    pd.Timedelta,
)

#: We especially do not want to see these in serialisation.
#: We need to do negative test, because Pandas types to some base class
#: magic.
#:
#: For Pandas float serialisation discussion see https://stackoverflow.com/questions/27098529/numpy-float64-vs-python-float and https://stackoverflow.com/questions/27098529/numpy-float64-vs-python-float
#:
BAD_VALUE_TYPES = (
    # np.float32,
    # np.float64,
    # pd.Timedelta,  fixed in monkeypatch/dataclasses_json.py
    # pd.Timestamp,  fixed in monkeypatch/dataclasses_json.py
)

_inf=INFINITY

_neginf=-INFINITY

# https://www.tutorialspoint.com/what-is-javascript-s-highest-integer-value-that-a-number-can-go-to-without-losing-precision
JS_MAX_INT = 9007199254740991


def validate_state_value(name: str | int, val: object):
    """Check the state value against our whitelist and blacklist."""

    if type(val) == float:
        # JavaScript number compatibility check
        # Note: NaNs are encoded as null now
        pass
    if type(val) == int:
        if val > JS_MAX_INT:
            raise BadStateData(f"{name}: {val} ({type(val)} - larger than JavaScript max int")
    elif isinstance(val, datetime.datetime):
        if val.tzinfo is not None:
            raise BadStateData(f"{name}: {val} ({type(val)} - datetime must be naive, this one contains timezone info")
    elif isinstance(val, BAD_VALUE_TYPES):
        raise BadStateData(f"{name}: {val} ({type(val)} - blacklisted value type")
    elif not isinstance(val, ALLOWED_VALUE_TYPES):
        raise BadStateData(f"{name}: {val} ({type(val)} - value type is not in supported serialisable types")


def walk(name: str | int, val: dict | list | object, key_type: Type):
    """Raise hierarchical exceptions to locate the bad key-value pair in nested data.

    :raise BadStateData:
        In the case we have sneaked something into the state
        that does not belong there.
    """
    try:
        if isinstance(val, dict):
            for k, v in val.items():
                walk(k, v, type(k))
        elif isinstance(val, list):
            for idx, val in enumerate(val):
                walk(idx, val, type(idx))
        else:
            if key_type not in ALLOWED_KEY_TYPES:
                raise BadStateData(f"'{name}' bad key type: {key_type}, allowed {ALLOWED_KEY_TYPES}")
            validate_state_value(name, val)
    except BadStateData as e:
        raise BadStateData(f"'{name}' ({val.__class__}) key has errors") from e


def validate_nested_state_dict(d: dict | list | object):
    """Validate state as serialised to a dictionary tree by dataclasses_json.

    See `to_dict` in `dataclass_json`.

    :raise BadStateData:
        In the case we have sneaked something into the state
        that does not belong there.
    """
    walk("state", d, type(d))


def validate_state_serialisation(state: State):
    """Check that we can write the state to the disk,

    Unlike `json.dump()` gives user friendly error messages.

    :raise BadStateData:
        In the case we have sneaked something into the state
        that does not belong there.
    """
    d = state.to_dict()
    validate_nested_state_dict(d)
