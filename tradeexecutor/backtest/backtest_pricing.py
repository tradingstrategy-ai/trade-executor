import logging
import datetime
import warnings
from decimal import Decimal, ROUND_DOWN
from typing import Optional

from tradeexecutor.backtest.backtest_execution import BacktestExecutionModel
from tradeexecutor.ethereum.uniswap_v2_routing import UniswapV2SimpleRoutingModel
from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.strategy.execution_model import ExecutionModel

from tradeexecutor.state.types import USDollarAmount
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.routing import RoutingModel
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, translate_trading_pair
from tradingstrategy.candle import GroupedCandleUniverse


logger = logging.getLogger(__name__)


class BacktestSimplePricingModel(PricingModel):
    """Look the price without any liquidity data or price impact.

    Use naive closing price.
    """

    def __init__(self,
                 candle_universe: GroupedCandleUniverse,
                 routing_model: RoutingModel,
                 candle_timepoint_kind="close",
                 very_small_amount=Decimal("0.10")):

        # TODO: Remove later - now to support some old code
        if isinstance(candle_universe, TradingStrategyUniverse):
            candle_universe = candle_universe.universe.candles

        assert isinstance(candle_universe, GroupedCandleUniverse), f"Got candles in wrong format: {candle_universe.__class__}"

        self.candle_universe = candle_universe
        self.very_small_amount = very_small_amount
        self.routing_model = routing_model
        self.candle_timepoint_kind = candle_timepoint_kind

    def get_pair_for_id(self, internal_id: int) -> Optional[TradingPairIdentifier]:
        """Look up a trading pair.

        Useful if a strategy is only dealing with pair integer ids.
        """
        warnings.warn("Do not use internal ids as they are not stable ids."
                      "Instead use chain id + address tuples")

        pair = self.universe.pairs.get_pair_by_id(internal_id)
        if not pair:
            return None
        return translate_trading_pair(pair)

    def check_supported_quote_token(self, pair: TradingPairIdentifier):
        assert pair.quote.address == self.routing_model.reserve_token_address, f"Quote token {self.routing_model.reserve_token_address} not supported for pair {pair}, pair tokens are {pair.base.address} - {pair.quote.address}"

    def get_sell_price(self,
                       ts: datetime.datetime,
                       pair: TradingPairIdentifier,
                       quantity: Optional[Decimal],
                       ) -> USDollarAmount:
        # TODO: Include price impact
        pair_id = pair.internal_id
        return self.candle_universe.get_closest_price(pair_id, ts, kind=self.candle_timepoint_kind)

    def get_buy_price(self,
                       ts: datetime.datetime,
                       pair: TradingPairIdentifier,
                       reserve: Optional[Decimal],
                       ) -> USDollarAmount:
        # TODO: Include price impact
        pair_id = pair.internal_id
        return float(self.candle_universe.get_closest_price(pair_id, ts, kind=self.candle_timepoint_kind))

    def quantize_base_quantity(self, pair: TradingPairIdentifier, quantity: Decimal, rounding=ROUND_DOWN) -> Decimal:
        """Convert any base token quantity to the native token units by its ERC-20 decimals."""
        assert isinstance(pair, TradingPairIdentifier)
        decimals = pair.base.decimals
        return Decimal(quantity).quantize((Decimal(10) ** Decimal(-decimals)), rounding=ROUND_DOWN)


def backtest_pricing_factory(
        execution_model: ExecutionModel,
        universe: TradingStrategyUniverse,
        routing_model: UniswapV2SimpleRoutingModel) -> BacktestSimplePricingModel:

    assert isinstance(universe, TradingStrategyUniverse)
    assert isinstance(execution_model, BacktestExecutionModel), f"Execution model not compatible with this execution model. Received {execution_model}"
    assert isinstance(routing_model, UniswapV2SimpleRoutingModel), f"This pricing method only works with Uniswap routing model, we received {routing_model}"

    return BacktestSimplePricingModel(
        universe.universe.candles,
        routing_model)

