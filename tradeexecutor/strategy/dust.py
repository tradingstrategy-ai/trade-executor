"""Dust amounts and epsilon rounding.

"""
from _decimal import Decimal

from tradeexecutor.state.identifier import TradingPairIdentifier



#: The absolute number of tokens we consider the value to be zero
#:
#:
#: Because of funny %s of values divided near zero,
#: we cannot use relative comparison near zero values.
#:
#:
DEFAULT_DUST_EPSILON = Decimal(10 ** -4)



def get_dust_epsilon(pair: TradingPairIdentifier) -> Decimal:
    """Get the dust threshold for a trading pair.

    :param pair:
        Trading pair identifier.

    :return:
        Maximum amount of units we consider "zero".

    """

    # Hardcoded rules for now
    if pair.base.token_symbol == "WBTC":
        return Decimal(10 ** -7)
    else:
        return DEFAULT_DUST_EPSILON
