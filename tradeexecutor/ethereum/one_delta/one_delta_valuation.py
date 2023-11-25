"""Value model for 1delta trade based on Uniswap v3 market price.
"""
import datetime

from tradeexecutor.ethereum.one_delta.one_delta_live_pricing import OneDeltaLivePricing
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.types import USDollarAmount
from tradeexecutor.ethereum.eth_valuation import EthereumPoolRevaluator


class OneDeltaPoolRevaluator(EthereumPoolRevaluator):
    """Re-value assets based on their on-chain price.

    Does directly JSON-RPC call to get the latest price in the Uniswap pools.

    Only uses direct route - mostly useful for testing, may not give a realistic price in real
    world with multiple order routing options.

    .. warning ::

        This valuation method always uses the latest price. It
        cannot be used for backtesting.
    """

    def __init__(self, pricing_model: OneDeltaLivePricing):
        assert isinstance(pricing_model, OneDeltaLivePricing)
        super().__init__(pricing_model)

    def __call__(
        self,
        ts: datetime.datetime,
        position: TradingPosition,
    ) -> tuple[datetime.datetime, USDollarAmount]:
        pair = position.pair.get_pricing_pair()

        quantity = position.get_quantity()
        
        # Cannot do pricing for zero quantity
        if quantity == 0:
            return ts, 0.0

        price_structure = self.pricing_model.get_sell_price(ts, pair, quantity)

        return ts, price_structure.price


def one_delta_valuation_factory(pricing_model):
    return OneDeltaPoolRevaluator(pricing_model)