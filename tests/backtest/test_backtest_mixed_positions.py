"""Run a backtest where we uses mixed long and short positions, using synthetic oracle data."""
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

from tradeexecutor.analysis.trade_analyser import build_trade_analysis
from tradeexecutor.backtest.backtest_pricing import BacktestPricing
from tradeexecutor.backtest.backtest_routing import BacktestRoutingModel
from tradeexecutor.backtest.backtest_runner import run_backtest_inline
from tradeexecutor.backtest.simulated_wallet import SimulatedWallet
from tradeexecutor.ethereum.routing_data import get_backtest_routing_model
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.state import State
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier, TradingPairKind
from tradeexecutor.statistics.statistics_table import serialise_long_short_stats_as_json_table
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
from tradeexecutor.visual.equity_curve import calculate_compounding_realised_trading_profitability, calculate_long_compounding_realised_trading_profitability, calculate_short_compounding_realised_trading_profitability


# https://docs.pytest.org/en/latest/how-to/skipping.html#skip-all-test-functions-of-a-class-or-module
pytestmark = pytest.mark.skipif(os.environ.get("TRADING_STRATEGY_API_KEY") is None, reason="Set TRADING_STRATEGY_API_KEY environment variable to run this test")

start_at = datetime.datetime(2023, 1, 1)
end_at = datetime.datetime(2023, 1, 20)

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

    return create_synthetic_single_pair_universe(
        candles=candles,
        chain_id=chain_id,
        exchange=mock_exchange,
        time_bucket=time_bucket,
        pair=weth_usdc,
        lending_candles=lending_candle_universe,
    )


@pytest.fixture(scope="module")
def strategy_universe(universe):
    """Legacy alias. Use strategy_universe."""
    return universe


def test_backtest_open_both_long_short(
    persistent_test_client: Client,
        strategy_universe,
):
    """Run the strategy backtest using inline decide_trades function.

    - Open both long and short in the same cycle
    - Backtest shouldn't raise any exceptions
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
        trade_pair = strategy_universe.data_universe.pairs.get_single()

        cash = state.portfolio.get_cash()
        
        position_manager = PositionManager(timestamp, strategy_universe, state, pricing_model)

        trades = []

        if not position_manager.is_any_open():
            amount = cash * 0.5

            trades += position_manager.open_short(trade_pair, amount, leverage=leverage)
            
            trades += position_manager.open_spot(trade_pair, amount)

        return trades

    state, strategy_universe, debug_dump = run_backtest_inline(
        start_at=start_at,
        end_at=datetime.datetime(2023, 1, 5),
        client=persistent_test_client,
        cycle_duration=CycleDuration.cycle_1d,
        decide_trades=decide_trades,
        universe=strategy_universe,
        initial_deposit=capital,
        reserve_currency=ReserveCurrency.usdc,
        trade_routing=TradeRouting.uniswap_v3_usdc_poly,
        engine_version="0.3",
    )

    portfolio = state.portfolio
    assert len(portfolio.open_positions) == 2
    assert portfolio.get_cash() == 0
    assert portfolio.get_net_asset_value(include_interest=True) == pytest.approx(10198.205996721665)
    assert portfolio.calculate_total_equity() == pytest.approx(10198.205996721665)
    

def test_backtest_long_short_stats(
    persistent_test_client: Client,
        strategy_universe,
):
    """Run the strategy backtest using inline decide_trades function.

    - Open both long and short in the same cycle
    - Backtest shouldn't raise any exceptions
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
        trade_pair = strategy_universe.data_universe.pairs.get_single()

        cash = state.portfolio.get_cash()
        
        position_manager = PositionManager(timestamp, strategy_universe, state, pricing_model)

        trades = []

        if not position_manager.is_any_open():
            amount = cash * 0.1

            trades += position_manager.open_short(trade_pair, amount, leverage=leverage)
            
            trades += position_manager.open_spot(trade_pair, amount)
        else:
            trades += position_manager.close_all()

        return trades

    state, strategy_universe, debug_dump = run_backtest_inline(
        start_at=start_at,
        end_at=end_at,
        client=persistent_test_client,
        cycle_duration=CycleDuration.cycle_1d,
        decide_trades=decide_trades,
        universe=strategy_universe,
        initial_deposit=capital,
        reserve_currency=ReserveCurrency.usdc,
        trade_routing=TradeRouting.uniswap_v3_usdc_poly,
        engine_version="0.3",
    )

    portfolio = state.portfolio
    # assert len(portfolio.open_positions) == 2
    # assert portfolio.get_cash() == 0
    # assert portfolio.get_net_asset_value(include_interest=True) == pytest.approx(10198.205996721665)
    # assert portfolio.get_total_equity() == pytest.approx(10198.205996721665)
    
    overall_compounding_profit = calculate_compounding_realised_trading_profitability(state)
    long_compounding_profit = calculate_long_compounding_realised_trading_profitability(state)
    short_compounding_profit = calculate_short_compounding_realised_trading_profitability(state)
    
    analysis = build_trade_analysis(state.portfolio)
    summary = analysis.calculate_summary_statistics(state=state, time_bucket=strategy_universe.data_universe.time_bucket)
    long_summary = analysis.calculate_long_summary_statistics(state=state, time_bucket=strategy_universe.data_universe.time_bucket)
    short_summary = analysis.calculate_short_summary_statistics(state=state, time_bucket=strategy_universe.data_universe.time_bucket)
    summary_by_side = analysis.calculate_all_summary_stats_by_side(state=state, time_bucket=strategy_universe.data_universe.time_bucket)

    assert summary.compounding_returns.equals(overall_compounding_profit)
    assert long_summary.compounding_returns.equals(long_compounding_profit)
    assert short_summary.compounding_returns.equals(short_compounding_profit)

    # all longs and shorts opened and closed at same time
    assert summary.time_in_market == pytest.approx(0.5263157894736842)
    assert summary.time_in_market_volatile == pytest.approx(0.5263157894736842)
    assert long_summary.time_in_market == pytest.approx(0.5263157894736842)
    assert short_summary.time_in_market == pytest.approx(0.5263157894736842)
    
    assert summary.return_percent == pytest.approx(overall_compounding_profit.iloc[-1], abs=1e-3)  # TODO make more precise
    assert summary.total_interest_paid_usd == pytest.approx(2.4602719114723985)
    assert summary.median_interest_paid_usd == pytest.approx(0.0)
    assert summary.min_interest_paid_usd == pytest.approx(0.0)
    assert summary.max_interest_paid_usd == pytest.approx(0.2841330557398559)

    assert long_summary.total_interest_paid_usd == 0.0
    assert short_summary.total_interest_paid_usd == pytest.approx(2.4602719114723985)
    assert short_summary.median_interest_paid_usd == pytest.approx(0.2721172702270313)
    assert short_summary.min_interest_paid_usd == pytest.approx(0.0)
    assert short_summary.max_interest_paid_usd == pytest.approx(0.2841330557398559)
 
    assert overall_compounding_profit.iloc[0] == 0
    assert long_compounding_profit.iloc[0] == 0
    assert short_compounding_profit.iloc[0] == 0
    
    assert overall_compounding_profit.iloc[-1] == pytest.approx(-0.018257383118416515)
    assert long_compounding_profit.iloc[-1] == pytest.approx(-0.003555468782310056)
    assert short_compounding_profit.iloc[-1] == pytest.approx(-0.014754373048884384)
    overall = float(summary_by_side.loc['Return %']['All'][:-1])
    long = float(summary_by_side.loc['Return %']['Long'][:-1])
    short = float(summary_by_side.loc['Return %']['Short'][:-1])
    
    assert summary.return_percent * 100 == pytest.approx(overall, abs=1e-2)
    
    # TODO make more precise
    assert overall/100 == pytest.approx(overall_compounding_profit.iloc[-1], abs=1e-3)

    # Not really supposed to match up currently
    assert long == pytest.approx(long_compounding_profit.iloc[-1] * 100, abs=1e-1)
    assert short == pytest.approx(short_compounding_profit.iloc[-1] * 100, abs=1e-1)

    serialise_long_short_stats_as_json_table(backtested_state=state)
