"""Type aliases for state data structures.

.. note::

    We are currently supporting Python 3.9.
    Type alias support won't land until Python 3.10,
    so work here is much unfinished.

"""

#: Represents a US dollar amount used in valuation and prices.
#: This type alias cannot be used for accounting. For accountable amounts always use Decimal.
#: This type is only used for symboling that the function return value will be approximately
#: amount in the US dollar, mostly for being human readable purposes.
from typing import TypeAlias


#: Dollar amount that does not need to be accurately amounted
USDollarAmount: TypeAlias = float

#: Dollar price that does not need to be accurately amounted
USDollarPrice: TypeAlias = float

#: Basis points expressed as float
#:
#: 10000 bps = 100 % = 1 float
BPS: TypeAlias = float

JSONHexAddress: TypeAlias = str

JSONHexBytes: TypeAlias = str