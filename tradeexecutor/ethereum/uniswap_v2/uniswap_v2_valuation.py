"""Value model based on Uniswap v2 market price.

Value positions based on their "dump" price on Uniswap,
assuming we get the worst possible single trade execution.
"""
import datetime
from typing import Tuple

from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_live_pricing import UniswapV2LivePricing
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.types import USDollarAmount
from tradeexecutor.ethereum.eth_valuation import EthereumPoolRevaluator


class UniswapV2PoolRevaluator(EthereumPoolRevaluator):
    """Re-value assets based on their on-chain price.

    Does directly JSON-RPC call to get the latest price in the Uniswap pools.

    Only uses direct route - mostly useful for testing, may not give a realistic price in real
    world with multiple order routing options.

    .. warning ::

        This valuation metohd always uses the latest price. It
        cannot be used for backtesting.
    """

    def __init__(self, pricing_model: UniswapV2LivePricing):
        
        super().__init__(pricing_model)
        
        assert isinstance(pricing_model, UniswapV2LivePricing)

    def __call__(self,
                 ts: datetime.datetime,
                 position: TradingPosition) -> Tuple[datetime.datetime, USDollarAmount]:

        return super().__call__(ts, position)


def uniswap_v2_sell_valuation_factory(pricing_model):
    return UniswapV2PoolRevaluator(pricing_model)