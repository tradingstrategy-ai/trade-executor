"""Perform a test trade on a universe."""
import logging
import datetime
from decimal import Decimal

from tradingstrategy.universe import Universe

from tradeexecutor.state.state import State
from tradeexecutor.strategy.execution_model import ExecutionModel
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, translate_trading_pair

logger = logging.getLogger(__name__)


def make_test_trade(
        execution_model: ExecutionModel,
        pricing_model: PricingModel,
        state: State,
        universe: TradingStrategyUniverse,
        amount=Decimal("1.0"),
):
    """Perform a test trade.

    Buy and sell 1 token worth for 1 USD to check that
    our trade routing works.
    """

    ts = datetime.datetime.utcnow()

    data_universe: Universe = universe.universe

    reserve_asset = universe.get_reserve_asset()

    # TODO: Supports single pair universes only for now
    raw_pair = data_universe.pairs.get_single()
    pair = translate_trading_pair(raw_pair)

    assumed_price = pricing_model.get_buy_price(
        ts,
        pair,
        amount,
    )

    logger.info("Making a test trade on pair: %s, for %f %s price is %f %s/%s",
                pair,
                amount,
                reserve_asset.token_symbol,
                assumed_price,
                pair.base.token_symbol,
                reserve_asset.token_symbol,
                )


