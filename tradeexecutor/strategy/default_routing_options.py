"""Default routing options for trading strategies.

The strategy :ref:`routing model` defines how individual trades
are split to different blockchain transactions. Furthermore,
we might need to do a currency conversion between our strategy :term:`reserve currency`
and the trading pair quote token and this is a part of the routing.

The :py:class:`TradeRouting`, we define the default routing model options a trading strategy can have.
There is also a custom `user_supplied_routing_model` option in which case you need to construct
the routing model in the code yourself.
"""

import enum


class TradeRouting(enum.Enum):
    """What trade routing should the strategy use.

    These values can be given to `trade_routing` variable in a strategy.
    """

    #: Two or three-legged trades on PancakeSwap.
    #: - Open positions with BUSD quote token.
    #: - Open positions with WBNB quote token.
    pancakeswap_busd = "pancakeswap_busd"

    #: Two or three-legged trades on PancakeSwap.
    #: - Open positions with USDC quote token.
    #: - Open positions with WBNB quote token.
    pancakeswap_usdc = "pancakeswap_usdc"

    #: Two or three legged trades on Quickswap
    #: - Open positions with USDC quote token.
    #: - Open positions with WMATIC quote token.
    quickswap_usdc = "quickswap_usdc"

    #: Use user supplied routing model
    #:
    #: The routing table is constructed by the developer in the
    #: Python code.
    #:
    #: Mostly useful for unit testing.
    user_supplied_routing_model = "user_supplied_routing_model"
