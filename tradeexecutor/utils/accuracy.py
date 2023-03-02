"""Make sure we use accurate enough Decimal math to cover token asset quantity.

Ethereum assets have 18 decimals.

"""

from decimal import Decimal
from typing import Iterable


#: If sum of token quantities goes below this value assume the sum is zero
SUM_EPSILON = Decimal(10**-18)

#: Preconstruced Decimal Zero
#:
#: Avoid object reinitialisation.
ZERO_DECIMAL = Decimal(0)


def setup_decimal_accuracy():
    """Make sure we can handle Decimals up to 18 digits.

    .. note::

        Currently we assume we can safely trade without worring about the decimal accuracy,
        as we have some special epsilon rules in place to work around the kinks. Increasing the
        decimal accuracy will slow down calculations.

        Also increasing the decimal accuracy does not remedy us from issues.

    """

    # From ethereum.stackexchange.com:
    #
    # > I believe the minimum correct precision is math.ceil(math.log10(2**256)) = 78, no matter how many decimals the token is.
    #
    #decimal.getcontext().prec = 78


def sum_decimal(numbers: Iterable[Decimal]) -> Decimal:
    """Decimal safe sum().

    Looks like Python will fail to sum plus and minus decimals together even if they cancel each out:

    .. code-block:: text

        57602384161.6838278325398013034137975573193695174227082184047361995798525240101
        -57602384161.6838278325398013034137975573193695174227082184047361995798525240101

        0E-67

    :param numbers:
        Incoming Decimals to sum
    """
    total = sum(numbers)
    if total < SUM_EPSILON:
        return ZERO_DECIMAL
    return total





