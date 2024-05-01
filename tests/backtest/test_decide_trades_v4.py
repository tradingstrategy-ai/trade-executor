"""Test DecideTradesV4 function signature..

"""
import os
import logging
import datetime
from _decimal import Decimal
from typing import List, Dict

import pandas_ta
import pytest
import pandas as pd

from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.strategy_module import StrategyParameters
from tradingstrategy.client import Client
from tradingstrategy.chain import ChainId
from tradingstrategy.lending import LendingProtocolType
from tradingstrategy.timebucket import TimeBucket

from tradeexecutor.backtest.backtest_runner import run_backtest_inline
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, load_partial_data
from tradeexecutor.strategy.execution_context import ExecutionContext, unit_test_execution_context, ExecutionMode
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.reserve_currency import ReserveCurrency
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.universe_model import UniverseOptions, default_universe_options
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.state.state import State


# https://docs.pytest.org/en/latest/how-to/skipping.html#skip-all-test-functions-of-a-class-or-module
pytestmark = pytest.mark.skipif(os.environ.get("TRADING_STRATEGY_API_KEY") is None, reason="Set TRADING_STRATEGY_API_KEY environment variable to run this test")



def create_trading_universe(
    ts: datetime.datetime,
    client: Client,
    execution_context: ExecutionContext,
    universe_options: UniverseOptions,
) -> TradingStrategyUniverse:

    assert universe_options.start_at

    pairs = [
        (ChainId.polygon, "uniswap-v3", "WETH", "USDC", 0.0005)
    ]

    reverses = [
        (ChainId.polygon, LendingProtocolType.aave_v3, "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
    ]

    dataset = load_partial_data(
        client,
        execution_context=unit_test_execution_context,
        time_bucket=TimeBucket.d1,
        pairs=pairs,
        universe_options=default_universe_options,
        start_at=universe_options.start_at,
        end_at=universe_options.end_at,
        lending_reserves=reverses,
    )

    strategy_universe = TradingStrategyUniverse.create_single_pair_universe(dataset)
    return strategy_universe


@pytest.fixture()
def strategy_universe(persistent_test_client: Client):
    return create_trading_universe(
        datetime.datetime.now(),
        persistent_test_client,
        unit_test_execution_context,
        UniverseOptions(start_at=datetime.datetime(2023, 1, 1), end_at=datetime.datetime(2023, 2, 1))
    )


def test_decide_trades_v04(strategy_universe):
    """Test DecideTradesProtocolV4

    - Check that StrategyInput is passed correctly in backtesting (only backtesting, not live trading)
    """

    def decide_trades(input: StrategyInput) -> List[TradeExecution]:
        """A simple strategy that puts all in to our lending reserve."""

        assert input.cycle > 0
        assert input.parameters.test_val == 111
        assert input.execution_context.mode.is_unit_testing()
        assert input.indicators is not None

        position_manager = input.get_position_manager()
        pair = input.get_default_pair()
        cash = input.state.portfolio.get_cash()

        trades = []

        # Check price accessor
        price_value = input.indicators.get_price()
        if price_value is not None:
            assert 0 < price_value < 100_000
        assert price_value == input.indicators.get_price(index=-1)
        previous_price = input.indicators.get_price(index=-2)
        if previous_price is not None:
            assert 0 < previous_price < 100_000
            assert previous_price != price_value

        # Check indicator accessor
        rsi_value = input.indicators.get_indicator_value("rsi")
        if rsi_value is not None:
            assert 0 < rsi_value < 100

        bb_value = input.indicators.get_indicator_value("bb", "BBL_20_2.0")
        if bb_value is not None:
            assert 0 < bb_value < 10_000

        _ = input.indicators.get_indicator_value("rsi", index=-1)
        _ = input.indicators.get_indicator_value("rsi", index=-2)

        # Switch between full spot open and close between cycles
        if not position_manager.is_any_open():
            trades += position_manager.open_spot(pair, cash * 0.99)
        else:
            trades += position_manager.close_all()

        return trades

    def create_indicators(parameters: StrategyParameters, indicators: IndicatorSet, strategy_universe: TradingStrategyUniverse, execution_context: ExecutionContext):
        indicators.add("rsi", pandas_ta.rsi, {"length": parameters.rsi_length})
        indicators.add("bb", pandas_ta.bbands, {"length": parameters.bb_length})

    class MyParameters:
        test_val = 111
        initial_cash = 10_000
        cycle_duration = CycleDuration.cycle_1d
        rsi_length = 21
        bb_length = 20

    # Run the test
    result = run_backtest_inline(
        client=None,
        decide_trades=decide_trades,
        create_indicators=create_indicators,
        universe=strategy_universe,
        reserve_currency=ReserveCurrency.usdc,
        engine_version="0.5",
        parameters=StrategyParameters.from_class(MyParameters),
        mode=ExecutionMode.unit_testing,
    )

    state, universe, debug_dump = result
    assert len(state.portfolio.closed_positions) == 15

    assert result.indicators.available_indicators.has_indicator("rsi")
    assert result.indicators.available_indicators.has_indicator("bb")

    bb = result.indicators.get_indicator_dataframe("bb")
    assert isinstance(bb,  pd.DataFrame)
