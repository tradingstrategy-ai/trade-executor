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

    - These values can be given to `trade_routing` variable in a strategy.

    - Thus option hides the complexity of the actual routing logic
      form an average developer.

    See also :py:mod:`tradeexecutor.ethereum.routing_data`
    for actual routing data implementation.
    """

    #: Two or three-legged trades on PancakeSwap.
    #:
    #: - Open positions with BUSD quote token.
    #:
    #: - Open positions with WBNB quote token.
    pancakeswap_busd = "pancakeswap_busd"

    #: Two or three-legged trades on PancakeSwap.
    #:
    #: - Open positions with USDC quote token.
    #:
    #: - Open positions with WBNB quote token.
    pancakeswap_usdc = "pancakeswap_usdc"

    #: Two or three legged trades on Pancake on BSC
        #:
    #: - Open positions with USDT quote token.
    #:
    #: - Open positions with WBNB quote token.
    pancakeswap_usdt = "pancakeswap_usdt"

    #: Two or three legged trades on Quickswap on Polygon
    #:
    #: - Open positions with USDC quote token.
    #:
    #: - Open positions with WMATIC quote token.
    quickswap_usdc = "quickswap_usdc"

    #: Two or three legged trades on Quickswap on Polygon
    #:
    #: - Open positions with USDT quote token.
    #:
    #: - Open positions with WMATIC quote token.
    quickswap_usdt = "quickswap_usdt"

    #: Two or three legged trades on Quickswap on Polygon
    #:
    #: - Open positions with DAI quote token.
    #:
    #: - Open positions with USDC quote token.
    quickswap_dai = "quickswap_dai"

    #: Two or three legged trades on Trader Joe on Avalanche
    #:
    #: - Open positions with USDC quote token.
    #:
    #: - Open positions with WAVAX quote token.
    trader_joe_usdc = "trader_joe_usdc"

    #: Two or three legged trades on Trader Joe on Avalanche
    #:
    #: - Open positions with USDT quote token.
    #:
    #: - Open positions with WAVAX quote token.
    trader_joe_usdt = "trader_joe_usdt"

    #: Two or three legged trades on Uniswap v2 on Ethereum mainnet
    #:
    #: - Open positions with USDC quote token.
    #:
    #: - Open positions with WETH quote token.
    uniswap_v2_usdc = "uniswap_v2_usdc"

    #: Two or three legged trades on Uniswap V2 on Ethereum
    #:
    #: - Open positions with USDT quote token.
    #:
    #: - Open positions with WETH quote token.
    uniswap_v2_usdt = "uniswap_v2_usdt"

    #: Two or three legged trades on Uniswap V2 on Ethereum
    #:
    #: - Open positions with DAI quote token.
    #:
    #: - Open positions with WETH quote token.
    uniswap_v2_dai = "uniswap_v2_dai"

    #: Two or three legged trades on Uniswap v3 on Ethereum mainnet
    #:
    #: - Open positions with USDC quote token.
    #:
    #: - Open positions with WETH quote token.
    uniswap_v3_usdc = "uniswap_v3_usdc"

    #: Two or three legged trades on Uniswap v3 on Ethereum mainnet
    #:
    #: - Open positions with USDT quote token.
    #:
    #: - Open positions with WETH quote token.
    #:
    #: - Open positions with USDC quote token.
    uniswap_v3_usdt = "uniswap_v3_usdt"

    #: Two or three legged trades on Uniswap v3 on Ethereum mainnet
    #:
    #: - Open positions with DAI quote token.
    #:
    #: - Open positions with WETH quote token.
    #:
    #: - Open positions with USDC quote token.
    uniswap_v3_dai = "uniswap_v3_dai"

    #: Two or three legged trades on Uniswap v3 on Ethereum mainnet
    #:
    #: - Open positions with BUSD quote token.
    #:
    #: - Open positions with USDC quote token.
    #:
    #: - Open positions with USDT quote token.
    #:
    #: - Open positions with DAI quote token.
    uniswap_v3_busd = "uniswap_v3_busd"
    
    #: Two or three legged trades on Uniswap v3 on Polygon mainnet
    #:
    #: - Open positions with USDC quote token.
    #:
    #: - Open positions with WETH quote token.
    uniswap_v3_usdc_poly = "uniswap_v3_usdc_poly"

    #: Two or three legged trades on Uniswap v3 on Polygon mainnet
    #:
    #: - Open positions with USDT quote token.
    #:
    #: - Open positions with WETH quote token.
    #:
    #: - Open positions with USDC quote token.
    uniswap_v3_usdt_poly = "uniswap_v3_usdt_poly"

    #: Use user supplied routing model
    #:
    #: The routing table is constructed by the developer in the
    #: Python code.
    #:
    #: Mostly useful for unit testing.
    user_supplied_routing_model = "user_supplied_routing_model"

    #: Backtesting only
    #:
    #: The order routing is ignored. We use backtest estimations for trading
    #: fees and assume all pairs are tradeable.
    ignore = "ignore"