"""Reserve currency options for strategies.

See :ref:`reserve currency` for more information.
"""

import enum


class ReserveCurrency(enum.Enum):
    """Default supported reserve currencies.

    These are the options for strategy module `reverse_currency` value.

    See :ref:`reserve currency` for more information.
    """

    #: Strategy holds its reserves as BUSD on BNB Chain
    busd = "busd"

    #: Strategy holds its reserves USDC (multi-chain)
    usdc = "usdc"

    #: Strategy holds its reserves as USDT (multi-chain)
    usdt = "usdt"

    #: Strategy holds its reserves as DAI (multi-chain)
    dai = "dai"