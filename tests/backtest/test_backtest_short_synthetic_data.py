"""Run a backtest where we open a credit supply position, using real oracle data.

"""
import os
import logging
import datetime
import random
from _decimal import Decimal

import pytest
from typing import List, Dict

import pandas as pd

from tradingstrategy.client import Client
from tradingstrategy.chain import ChainId
from tradingstrategy.lending import LendingProtocolType
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.universe import Universe

from tradeexecutor.backtest.backtest_runner import run_backtest_inline
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.state import State
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier, TradingPairKind
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, load_partial_data
from tradeexecutor.strategy.execution_context import ExecutionContext, unit_test_execution_context
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.reserve_currency import ReserveCurrency
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.universe_model import UniverseOptions, default_universe_options
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, create_pair_universe_from_code
from tradeexecutor.testing.synthetic_ethereum_data import generate_random_ethereum_address
from tradeexecutor.testing.synthetic_exchange_data import generate_exchange
from tradeexecutor.testing.synthetic_price_data import generate_ohlcv_candles
from tradeexecutor.testing.synthetic_universe_data import create_synthetic_single_pair_universe
from tradeexecutor.testing.synthetic_lending_data import generate_lending_reserve, generate_lending_universe


# https://docs.pytest.org/en/latest/how-to/skipping.html#skip-all-test-functions-of-a-class-or-module
pytestmark = pytest.mark.skipif(os.environ.get("TRADING_STRATEGY_API_KEY") is None, reason="Set TRADING_STRATEGY_API_KEY environment variable to run this test")

start_at = datetime.datetime(2023, 1, 1)
end_at = datetime.datetime(2023, 2, 1)

@pytest.fixture(scope="module")
def universe() -> TradingStrategyUniverse:
    """Set up a mock universe."""

    time_bucket = TimeBucket.d1

    # Set up fake assets
    chain_id = ChainId.ethereum
    mock_exchange = generate_exchange(
        exchange_id=random.randint(1, 1000),
        chain_id=chain_id,
        address=generate_random_ethereum_address(),
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
        fee=0.0030,
    )

    usdc_reserve = generate_lending_reserve(usdc, chain_id, 1)
    weth_reserve = generate_lending_reserve(weth, chain_id, 2)

    _, lending_candle_universe = generate_lending_universe(
        time_bucket,
        start_at,
        end_at,
        reserves=[usdc_reserve, weth_reserve],
        aprs={
            "supply": 2,
            "variable": 5,
        }
    )

    candles = generate_ohlcv_candles(
        time_bucket,
        start_at,
        end_at,
        start_price=1800,
        pair_id=weth_usdc.internal_id,
        exchange_id=mock_exchange.exchange_id,
    )

    # print(candles.tail(10))
    
    return create_synthetic_single_pair_universe(
        candles=candles,
        chain_id=chain_id,
        exchange=mock_exchange,
        time_bucket=time_bucket,
        pair=weth_usdc,
        lending_candles=lending_candle_universe,
    )


def test_backtest_open_only_short_synthetic_data(
    persistent_test_client: Client,
    universe,
):
    """Run the strategy backtest using inline decide_trades function.

    - Open short position
    - Check unrealised PnL after a month
    """

    capital = 10000
    leverage = 2

    def decide_trades(
        timestamp: pd.Timestamp,
        strategy_universe: TradingStrategyUniverse,
        state: State,
        pricing_model: PricingModel,
        cycle_debug_data: Dict
    ) -> List[TradeExecution]:
        """A simple strategy that opens a single 2x short position."""
        trade_pair = strategy_universe.universe.pairs.get_single()

        cash = state.portfolio.get_cash()
        
        position_manager = PositionManager(timestamp, strategy_universe, state, pricing_model)

        trades = []

        if not position_manager.is_any_open():
            trades += position_manager.open_short(trade_pair, cash, leverage=leverage)

        return trades

    # Run the test
    state, universe, debug_dump = run_backtest_inline(
        start_at=start_at,
        end_at=end_at,
        client=persistent_test_client,
        cycle_duration=CycleDuration.cycle_1d,
        decide_trades=decide_trades,
        universe=universe,
        initial_deposit=capital,
        reserve_currency=ReserveCurrency.usdc,
        trade_routing=TradeRouting.uniswap_v3_usdc_poly,
        engine_version="0.3",
    )

    portfolio = state.portfolio
    assert len(portfolio.open_positions) == 1
    position = portfolio.open_positions[1]
    assert position.is_short()
    assert position.is_open()

    assert position.get_value_at_open() == capital
    assert position.get_collateral() == pytest.approx(capital * leverage)
    assert position.get_borrowed() == pytest.approx(8566.427426301208)
    assert position.get_accrued_interest() == pytest.approx(18.809854404770963)
    assert position.get_realised_profit_usd() is None

    # TODO: following look wrong, verify the calculations
    assert position.get_value() == Decimal(8585.23728070598)
    assert position.get_unrealised_profit_usd() == pytest.approx(1454.7408590503317)
