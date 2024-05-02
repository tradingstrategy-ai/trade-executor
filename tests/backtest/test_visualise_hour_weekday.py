"""Test weekday/hour trade success rate visualisation with synthetic data.
"""

import logging
import os.path
import random
import datetime
from pathlib import Path
from typing import List, Dict

import pytest

import pandas as pd
from matplotlib.figure import Figure

from tradeexecutor.analysis.single_pair import expand_entries_and_exits
from tradeexecutor.analysis.timemap import ScoringMethod, visualise_weekly_time_heatmap
from tradeexecutor.backtest.backtest_runner import run_backtest_inline
from tradeexecutor.cli.log import setup_pytest_logging
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, \
    create_pair_universe_from_code, translate_trading_pair
from tradeexecutor.testing.synthetic_ethereum_data import generate_random_ethereum_address
from tradeexecutor.testing.synthetic_exchange_data import generate_exchange, generate_simple_routing_model
from tradeexecutor.testing.synthetic_price_data import generate_ohlcv_candles
from tradeexecutor.visual.equity_curve import calculate_equity_curve, calculate_returns, \
    calculate_aggregate_returns, visualise_equity_curve, visualise_returns_over_time, visualise_returns_distribution
from tradingstrategy.candle import GroupedCandleUniverse
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.state.state import State
from tradingstrategy.universe import Universe
from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.reserve_currency import ReserveCurrency
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.visual.web_chart import render_web_chart, WebChartType, WebChartSource


def decide_trades(
        timestamp: pd.Timestamp,
        universe: Universe,
        state: State,
        pricing_model: PricingModel,
        cycle_debug_data: Dict) -> List[TradeExecution]:
    """Cycle between assets to generate trades.

    - On 3rd days buy or sell asset 1

    - On 5rd days buy or sell asset 2

    - But with 25% of cash
    """

    assert timestamp.hour == 0
    assert timestamp.minute == 0

    # The pair we are trading
    asset_1 = translate_trading_pair(universe.pairs.get_pair_by_id(2))
    asset_2 = translate_trading_pair(universe.pairs.get_pair_by_id(3))

    # How much cash we have in the hand
    cash = state.portfolio.get_cash()

    if cash < 0:
        state.portfolio.get_cash()

    trades = []

    position_manager = PositionManager(timestamp, universe, state, pricing_model)

    if timestamp.day % 3 == 0:
        position = position_manager.get_current_position_for_pair(asset_1)
        if position is None:
            trades += position_manager.open_spot(asset_1, cash * 0.25)
        else:
            trades += position_manager.close_position(position)

    if timestamp.day % 5 == 0:
        position = position_manager.get_current_position_for_pair(asset_2)
        if position is None:
            trades += position_manager.open_spot(asset_2, cash * 0.25)
        else:
            trades += position_manager.close_position(position)

    return trades


@pytest.fixture(scope="module")
def universe() -> TradingStrategyUniverse:
    """Set up a mock universe with 6 months of candle data.

    - WETH/USDC pair id 2

    - AAVE/USDC pair id 3
    """

    start_at = datetime.datetime(2021, 6, 1)
    end_at = datetime.datetime(2022, 1, 1)

    # Set up fake assets
    mock_chain_id = ChainId.ethereum
    mock_exchange = generate_exchange(
        exchange_id=random.randint(1, 1000),
        chain_id=mock_chain_id,
        address=generate_random_ethereum_address())
    usdc = AssetIdentifier(ChainId.ethereum.value, generate_random_ethereum_address(), "USDC", 6, 1)
    weth = AssetIdentifier(ChainId.ethereum.value, generate_random_ethereum_address(), "WETH", 18, 2)
    aave = AssetIdentifier(ChainId.ethereum.value, generate_random_ethereum_address(), "AAVE", 18, 3)
    weth_usdc = TradingPairIdentifier(
        weth,
        usdc,
        generate_random_ethereum_address(),
        mock_exchange.address,
        internal_id=2,
        internal_exchange_id=mock_exchange.exchange_id,
        fee=0.0030,
    )

    aave_usdc = TradingPairIdentifier(
        aave,
        usdc,
        generate_random_ethereum_address(),
        mock_exchange.address,
        internal_id=3,
        internal_exchange_id=mock_exchange.exchange_id,
        fee=0.0030,
    )

    time_bucket = TimeBucket.d1

    pair_universe = create_pair_universe_from_code(mock_chain_id, [weth_usdc, aave_usdc])

    weth_candles = generate_ohlcv_candles(time_bucket, start_at, end_at, pair_id=weth_usdc.internal_id)
    aave_candles = generate_ohlcv_candles(time_bucket, start_at, end_at, pair_id=aave_usdc.internal_id)

    candle_universe = GroupedCandleUniverse.create_from_multiple_candle_datafarames([
        weth_candles,
        aave_candles,
    ])

    universe = Universe(
        time_bucket=time_bucket,
        chains={mock_chain_id},
        exchanges={mock_exchange},
        pairs=pair_universe,
        candles=candle_universe,
        liquidity=None
    )

    return TradingStrategyUniverse(data_universe=universe, reserve_assets=[usdc])


@pytest.fixture(scope="module")
def state(universe):
    """Prepare a run backtest.

    Then one can calculate different statistics over this.
    """

    start_at, end_at = universe.data_universe.candles.get_timestamp_range()

    routing_model = generate_simple_routing_model(universe)

    assert universe is not None

    # Run the test
    state, universe, debug_dump = run_backtest_inline(
        start_at=start_at.to_pydatetime(),
        end_at=end_at.to_pydatetime(),
        client=None,  # None of downloads needed, because we are using synthetic data
        cycle_duration=CycleDuration.cycle_1d,  # Override to use 24h cycles despite what strategy file says
        decide_trades=decide_trades,
        create_trading_universe=None,
        universe=universe,
        initial_deposit=10_000,
        reserve_currency=ReserveCurrency.busd,
        trade_routing=TradeRouting.user_supplied_routing_model,
        routing_model=routing_model,
        log_level=logging.WARNING,
    )

    return state


@pytest.mark.parametrize("method", [ScoringMethod.realised_profitability, ScoringMethod.success_rate, ScoringMethod.failure_rate])
def test_visualise_timemap(state, method):
    """Check our generated data looks correct."""
    positions = state.portfolio.get_all_positions()
    fig = visualise_weekly_time_heatmap(positions, method)
    fig.show()
    import ipdb ; ipdb.set_trace()
