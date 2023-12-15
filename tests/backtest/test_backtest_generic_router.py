"""Generic router for backtesting."""


import os
import datetime
import random

import pytest
from typing import List

import pandas as pd

from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket
from tradeexecutor.backtest.backtest_runner import run_backtest_inline
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.state import State
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.testing.synthetic_ethereum_data import generate_random_ethereum_address
from tradeexecutor.testing.synthetic_exchange_data import generate_exchange
from tradeexecutor.testing.synthetic_price_data import generate_ohlcv_candles
from tradeexecutor.testing.synthetic_universe_data import create_synthetic_single_pair_universe
from tradeexecutor.testing.synthetic_lending_data import generate_lending_reserve, generate_lending_universe

# https://docs.pytest.org/en/latest/how-to/skipping.html#skip-all-test-functions-of-a-class-or-module
pytestmark = pytest.mark.skipif(os.environ.get("TRADING_STRATEGY_API_KEY") is None, reason="Set TRADING_STRATEGY_API_KEY environment variable to run this test")


@pytest.fixture(scope="module")
def universe() -> TradingStrategyUniverse:
    """Set up a mock universe."""


    start_at = datetime.datetime(2023, 1, 1)
    end_at = datetime.datetime(2023, 1, 5)
    candle_end_at = datetime.datetime(2023, 1, 30)

    time_bucket = TimeBucket.d1

    # Set up fake assets
    chain_id = ChainId.polygon
    mock_exchange = generate_exchange(
        exchange_id=random.randint(1, 1000),
        chain_id=chain_id,
        address=generate_random_ethereum_address(),
        exchange_slug="uniswap-v3",
    )
    usdc = AssetIdentifier(chain_id.value, generate_random_ethereum_address(), "USDC", 6, 1)
    weth = AssetIdentifier(chain_id.value, generate_random_ethereum_address(), "WETH", 18, 2)
    weth_usdc = TradingPairIdentifier(
        weth,
        usdc,
        generate_random_ethereum_address(),
        mock_exchange.address,
        internal_id=random.randint(1, 1000),
        internal_exchange_id=mock_exchange.exchange_id,
        fee=0.0005,
    )

    usdc_reserve = generate_lending_reserve(usdc, chain_id, 1)
    weth_reserve = generate_lending_reserve(weth, chain_id, 2)

    _, lending_candle_universe = generate_lending_universe(
        time_bucket,
        start_at,
        candle_end_at,
        reserves=[usdc_reserve, weth_reserve],
        aprs={
            "supply": 2,
            "variable": 5,
        }
    )

    candles = generate_ohlcv_candles(
        time_bucket,
        start_at,
        candle_end_at,
        start_price=1800,
        pair_id=weth_usdc.internal_id,
        exchange_id=mock_exchange.exchange_id,
    )

    return create_synthetic_single_pair_universe(
        candles=candles,
        chain_id=chain_id,
        exchange=mock_exchange,
        time_bucket=time_bucket,
        pair=weth_usdc,
        lending_candles=lending_candle_universe,
    )


@pytest.fixture(scope="module")
def strategy_universe(universe) -> TradingStrategyUniverse:
    return universe


def decide_trades(
        timestamp: pd.Timestamp,
        strategy_universe: TradingStrategyUniverse,
        state: State,
        pricing_model: PricingModel,
        cycle_debug_data: dict
) -> List[TradeExecution]:
    # Every second day buy spot,
    # every second day short

    trades = []
    position_manager = PositionManager(timestamp, strategy_universe, state, pricing_model)
    cycle = cycle_debug_data["cycle"]
    pairs = strategy_universe.data_universe.pairs
    spot_eth = pairs.get_pair_by_human_description((ChainId.polygon, "uniswap-v3", "WETH", "USDC", 0.0005))

    if position_manager.is_any_open():
        trades += position_manager.close_all()

    if cycle % 2 == 0:
        # Spot day
        trades += position_manager.open_spot(spot_eth, 100.0)
    else:
        # Short day
        trades += position_manager.open_short(spot_eth, 150.0)

    return trades


def test_backtest_generic_router(
    strategy_universe,
):
    """Run the strategy backtest using generic routing routing
    """

    start_at, end_at = strategy_universe.data_universe.candles.get_timestamp_range()

    assert isinstance(start_at, pd.Timestamp)

    # Run the test
    state, strategy_universe, debug_dump = run_backtest_inline(
        start_at=start_at.to_pydatetime(),
        end_at=end_at.to_pydatetime(),
        client=None,
        cycle_duration=CycleDuration.cycle_1d,
        decide_trades=decide_trades,
        universe=strategy_universe,
        initial_deposit=10_000,
        trade_routing=TradeRouting.default,
        engine_version="0.3",
    )

    portfolio = state.portfolio
    assert len(portfolio.closed_positions) == 27
