# Wait until NumPy supports Python 3.10 on macOS
# from typing import TypeAlias

#: Represents a US dollar amount used in valuation and prices.
#: This type alias cannot be used for accounting. For accountable amounts always use Decimal.
#: This type is only used for symboling that the function return value will be approximately
#: amount in the US dollar, mostly for being human readable purposes.
class USDollarAmount(float):
    pass

