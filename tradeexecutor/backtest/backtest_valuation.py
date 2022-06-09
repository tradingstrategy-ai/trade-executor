import datetime
from typing import Tuple

from tradeexecutor.backtest.backtest_pricing import BacktestPricingModel
from tradeexecutor.ethereum.uniswap_v2_live_pricing import UniswapV2LivePricing
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.types import USDollarAmount
from tradeexecutor.strategy.valuation import ValuationModel


class BacktestValuationModel(ValuationModel):
    """Re-value assets based on their on-chain backtest dataset price."""

    def __init__(self, pricing_model: BacktestPricingModel):
        self.pricing_model = pricing_model

    def __call__(self,
                 ts: datetime.datetime,
                 position: TradingPosition) -> Tuple[datetime.datetime, USDollarAmount]:
        assert isinstance(ts, datetime.datetime)
        pair = position.pair

        assert position.is_long(), "Short not supported"
        price = self.pricing_model.get_sell_price(ts, pair, quantity)

        return ts, price


def backtest_valuation_factory(pricing_model):
    return BacktestValuationModel(pricing_model)