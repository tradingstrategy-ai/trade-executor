import datetime
from typing import Tuple

from eth_hentai.uniswap_v2 import UniswapV2Deployment
from eth_hentai.uniswap_v2_fees import estimate_sell_price_decimals
from tradeexecutor.state.state import TradingPosition
from tradeexecutor.state.types import USDollarAmount


class UniswapV2PoolRevaluator:
    """Re-value assets based on their on-chain price.

    Does directly JSON-RPC call to get the latest price in the Uniswap pools.

    Only uses direct route - mostly useful for testing, may not give a realistic price in real
    world with multiple order routing options.

    .. warning ::

        This valuation metohd always uses the latest price. It
        cannot be used for backtesting.
    """

    def __init__(self, uniswap: UniswapV2Deployment):
        self.uniswap = uniswap

    def __call__(self, timestamp: datetime.datetime, position: TradingPosition) -> Tuple[datetime.datetime, USDollarAmount]:
        assert isinstance(timestamp, datetime.datetime)
        pair = position.pair
        quantity = position.get_quantity()
        # Cannot do pricing for zero quantity
        if quantity == 0:
            return timestamp, 0.0
        web3 = self.uniswap.web3
        total_price = estimate_sell_price_decimals(self.uniswap, web3.toChecksumAddress(pair.base.address), web3.toChecksumAddress(pair.quote.address), quantity)
        return timestamp, float(total_price / quantity)





