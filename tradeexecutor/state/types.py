"""Type aliases for state data structures.

.. note::

    We are currently supporting Python 3.9.
    Type alias support won't land until Python 3.10,
    so work here is much unfinished.

"""
from decimal import DecimalTuple, Decimal
#: Represents a US dollar amount used in valuation and prices.
#: This type alias cannot be used for accounting. For accountable amounts always use Decimal.
#: This type is only used for symboling that the function return value will be approximately
#: amount in the US dollar, mostly for being human readable purposes.
from typing import TypeAlias


#: Dollar amount that does not need to be accurately amounted
USDollarAmount: TypeAlias = float

#: How many tokens we are about to buy/sell
#:
#: Decimal type should be configured to carry accuracy up to 18 decimals,
#: but there is bound to be rounding errors so you need to always use an epsilon check.
#:
TokenAmount: TypeAlias = Decimal

#: Dollar price that does not need to be accurately amounted
USDollarPrice: TypeAlias = float

#: Basis points expressed as float
#:
#: 10000 bps = 100 % = 1 float
#:
#: See also :py:data:`Percent`.
BPS: TypeAlias = float

#: Basis points expressed as float
#:
#: 100 bps = 100
#:
#: See also :py:data:`BPS`.
IntBPS: TypeAlias = int


JSONHexAddress: TypeAlias = str

JSONHexBytes: TypeAlias = str

#: Pair primary key as integer.
#:
#: Note that these are not stable over the time,
#: please use (chain id, address tuple)
PairInternalId: TypeAlias = int


#: Raw Ethereum address as a string
#:
#: - lowercase
#:
#: - starts with 0x
#:
ZeroExAddress: TypeAlias = str


#: Represents percents
#:
#: This is an alias for float used for core readability purposes,
#; to differ function arguments from absolute values.
#:
#: 1.0 = 100%
#:
#: See also :py:data:`BPS`.
Percent: TypeAlias = float


#: Block number.
#:
#: - Cannot be negative
#:
#: - Cannot be zero
BlockNumber: TypeAlias = int

#: Unix timestamp as seconds
UnixTimestamp: TypeAlias = float

#: Leverage multiplier
#:
#: 1.0 is spot market.
#: 20.0 is degen.
#:
#: For over collateralised position. E.g. 8 USDC and then 6 USD worth of ETH this is ``0.8``.
#:
LeverageMultiplier: TypeAlias = float


class LegacyDataException(Exception):
    """We are dealing with old files and cannot complete this calculation because data points where not yet recorded."""