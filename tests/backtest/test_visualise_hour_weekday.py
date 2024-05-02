"""Test weekday/hour trade success rate visualisation with synthetic data.
"""

import logging
import random
import datetime
from typing import List, Dict

import pytest

import pandas as pd
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.universe import Universe
from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket

from tradeexecutor.analysis.timemap import ScoringMethod, visualise_weekly_time_heatmap
from tradeexecutor.backtest.backtest_runner import run_backtest_inline
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, \
    create_pair_universe_from_code, translate_trading_pair
from tradeexecutor.testing.synthetic_ethereum_data import generate_random_ethereum_address
from tradeexecutor.testing.synthetic_exchange_data import generate_exchange, generate_simple_routing_model
from tradeexecutor.testing.synthetic_price_data import generate_ohlcv_candles
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.reserve_currency import ReserveCurrency
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.state.state import State



def decide_trades(
        timestamp: pd.Timestamp,
        universe: Universe,
        state: State,
        pricing_model: PricingModel,
        cycle_debug_data: Dict) -> List[TradeExecution]:
    """Generate random trades and positions."""

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

    if not position_manager.is_any_open():
        if random.randint(0, 5) == 0:
            trades += position_manager.open_spot(asset_1, cash * 0.25)
    else:
        if random.randint(0, 5) == 0:
            trades += position_manager.close_all()

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
        cycle_duration=CycleDuration.cycle_1h,  # Override to use 24h cycles despite what strategy file says
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


@pytest.mark.parametrize("method", [ScoringMethod.success_rate, ScoringMethod.failure_rate, ScoringMethod.realised_profitability])
def test_visualise_timemap(state, method):
    """Check weekday heatmap."""
    positions = state.portfolio.get_all_positions()
    fig = visualise_weekly_time_heatmap(positions, method)
    assert fig.data   # We just check code runs, not that it gives good data
