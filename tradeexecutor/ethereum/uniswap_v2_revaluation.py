import pandas as pd
from web3 import Web3

from smart_contracts_for_testing.uniswap_v2 import UniswapV2Deployment, estimate_sell_price_decimals
from tradeexecutor.state.state import TradingPosition
from tradeexecutor.strategy.revaluation import RevaluationFailed


class UniswapV2PoolRevaluator:
    """Re-value assets based on their on-chain price.

    Does directly JSON-RPC call to get the latest price in the Uniswap pools.

    Only uses direct route - mostly useful for testing, may not give a realistic price in real
    world with multiple order routing options.
    """

    def __init__(self, web3: Web3, uniswap: UniswapV2Deployment):
        self.web3 = web3
        self.uniswap = uniswap

    def __call__(self, timestamp: pd.Timestamp, position: TradingPosition):
        pair = position.pair
        price = estimate_sell_price_decimals(self.web3, self.uniswap, pair.base.address, pair.quote.address)
        return price





