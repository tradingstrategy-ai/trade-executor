"""Type aliases for state data structures.

.. note::

    We are currently supporting Python 3.9.
    Type alias support won't land until Python 3.10,
    so work here is much unfinished.

"""
import datetime

# Wait until NumPy supports Python 3.10 on macOS
# from typing import TypeAlias

#: Represents a US dollar amount used in valuation and prices.
#: This type alias cannot be used for accounting. For accountable amounts always use Decimal.
#: This type is only used for symboling that the function return value will be approximately
#: amount in the US dollar, mostly for being human readable purposes.
from decimal import Decimal


class USDollarAmount(float):
    pass


class NaiveDatetime(datetime.datetime):
    """Always encode datetime with timezones.

    By default datetime.datetime(ts) deserialises to a local timezone.
    We want to force UTC every to make sure state is transferrable between different computers.
    """
    def __init__(self, *args, **kwargs):
        """"""
        if len(args) == 1:
            pass


JSONHexAddress = str
JSONHexBytes = str