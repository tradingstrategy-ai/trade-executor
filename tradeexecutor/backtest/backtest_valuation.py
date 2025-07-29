import datetime
from typing import Tuple

from tradeexecutor.backtest.backtest_pricing import BacktestPricing
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.types import USDollarAmount
from tradeexecutor.state.identifier import TradingPairKind
from tradeexecutor.state.valuation import ValuationUpdate
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.valuation import ValuationModel


class BacktestValuationModel(ValuationModel):
    """Re-value assets based on their on-chain backtest dataset price.

    Each asset is valued at its market sell price estimation.
    """

    def __init__(
        self,
        pricing_model: BacktestPricing,
    ):
        assert pricing_model, "pricing_model missing"
        self.pricing_model = pricing_model

    def __call__(
        self,
        ts: datetime.datetime,
        position: TradingPosition
    ) -> ValuationUpdate:

        assert isinstance(ts, datetime.datetime)

        # Special case for 1s cycle duration used in unit tests
        assert ts.second == 0, f"Timestamp sanity check failed, does not have even seconds: {ts}"

        pair = position.pair

        if position.is_credit_supply():
            # TODO: Assumes stable USD stablecoins
            price = 1.0
        elif position.is_long():
            pricing_pair = pair.get_pricing_pair()
            quantity = position.get_quantity()
            trade_price = self.pricing_model.get_sell_price(ts, pricing_pair, quantity)
            price = trade_price.price
        else:
            # TODO: Use position net asset pricing for leveraged positions
            assert pair.kind == TradingPairKind.lending_protocol_short
            quantity = -position.get_quantity()
            trade_price = self.pricing_model.get_sell_price(ts, pair.underlying_spot_pair, quantity)
            price = trade_price.price

        old_price = position.last_token_price
        old_value = position.get_value()
        position.revalue_base_asset(ts, price)
        new_value = position.get_value()

        return ValuationUpdate(
            position_id=position.position_id,
            created_at=ts,
            valued_at=ts,
            new_price=price,
            new_value=new_value,
            old_value=old_value,
            old_price=old_price,
        )



def backtest_valuation_factory(pricing_model):
    return BacktestValuationModel(pricing_model)