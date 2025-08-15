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
    #:
    #: For Arbitrum this is native USDC (not bridged)
    #:
    usdc = "usdc"

    #: Strategy holds its reserves USDC bridged on Arbitrum (USDC.e)
    #:
    #: `0xff970a61a04b1ca14834a43f5de4533ebddb5cc8` on Arbitrum.
    usdc_e = "usdc_e"

    #: Strategy holds its reserves as USDT (multi-chain)
    usdt = "usdt"

    #: Strategy holds its reserves as DAI (multi-chain)
    dai = "dai"

