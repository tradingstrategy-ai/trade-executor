"""Stateful data validation.

Avoid non-JSON supported types to slipping into to the state.
We have some common culprits we want to catch.

Any error message contains tree presentation of the state,
so you can easily locate any values that are bad, unlike with :py:mod:`json`.
"""
import datetime
from decimal import Decimal
from enum import Enum
from types import NoneType

import pandas as pd
import numpy as np

from tradeexecutor.state.state import State


class BadStateData(Exception):
    """Having something we do not support in the state."""


#: Types we know we can safely pass to JSON serialisation
ALLOWED_VALUE_TYPES = (
    dict,
    list,
    float,
    int,
    str,
    NoneType,
    Enum,  # Supported by dadtaclasses_json
    datetime.datetime,  # Supported by dadtaclasses_json
    Decimal,  # Supported by dadtaclasses_json
)

#: We especially do not want to see these in serialisation.
#: We need to do negative test, because Pandas types to some base class
#: magic.
BAD_VALUE_TYPES = (
    np.float32,
    np.float64,
    pd.Timedelta,
    pd.Timestamp,
)


def validate_state_value(name: str | int, val: object):
    """Check the state value against our whitelist and blacklist."""

    if isinstance(val, BAD_VALUE_TYPES):
        raise BadStateData(f"{name}: {val} ({type(val)} - blacklisted value type")

    if not isinstance(val, ALLOWED_VALUE_TYPES):
        raise BadStateData(f"{name}: {val} ({type(val)} - value type is not in supported serialisable types")


def walk(name: str | int, val: dict | list | object):
    """Raise hierarchical exceptions to locate the bad key-value pair in nested data.

    :raise BadStateData:
        In the case we have sneaked something into the state
        that does not belong there.
    """
    try:
        if isinstance(val, dict):
            for k, v in val.items():
                walk(k, v)
        elif isinstance(val, list):
            for idx, val in enumerate(val):
                walk(idx, val)
        else:
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
    walk("state", d)


def validate_state_serialisation(state: State):
    """Check that we can write the state to the disk,

    Unlike `json.dump()` gives user friendly error messages.

    :raise BadStateData:
        In the case we have sneaked something into the state
        that does not belong there.
    """
    d = state.to_dict()
    validate_nested_state_dict(d)
