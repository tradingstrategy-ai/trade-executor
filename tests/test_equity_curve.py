"""Equity curve calculations test.

- Calculate trades on a synthetic universe containing two trading pairs

- We have the asset universe of 2 cryptos (WETH, AAVE) and 1 stablecoin (USDC)

- We use synthetic random data of 2 trading pairs

- We test with synthetic daily candles

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

    candle_universe = GroupedCandleUniverse.create_from_multiple_candle_dataframes([
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
def logger(request):
    """Setup test logger."""
    return setup_pytest_logging(request, mute_requests=False)


@pytest.fixture(scope="module")
def state(
        logger: logging.Logger,
        universe,
    ):
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


def test_precheck_universe(universe):
    """Check our generated data looks correct."""
    universe = universe.data_universe
    assert universe.pairs.get_count() == 2
    assert universe.candles.get_pair_count() == 2
    assert len(universe.pairs.pair_map) == 2
    assert universe.pairs.get_pair_by_id(2).base_token_symbol == "WETH"
    assert universe.pairs.get_pair_by_id(3).base_token_symbol == "AAVE"


def test_calculate_equity_curve(state: State):
    """Get the backtest equity curve."""

    # Check that our trades look correct
    assert len(list(state.portfolio.get_all_trades())) > 10

    curve = calculate_equity_curve(state)
    assert type(curve) == pd.Series

    # Check begin and end values of portfolio look correct
    assert state.portfolio.get_total_equity() == pytest.approx(8605.74)
    assert curve[pd.Timestamp("2021-06-01")] == 10_000
    assert curve[pd.Timestamp("2021-12-30")] == pytest.approx(8605.74)


def test_calculate_equity_time_gapped(state: State):
    """Get the backtest equity curve with time gaps filler."""

    # Check that our trades look correct
    assert len(list(state.portfolio.get_all_trades())) > 10

    curve = calculate_equity_curve(state, fill_time_gaps=True)
    assert type(curve) == pd.Series

    # Check begin and end values of portfolio look correct
    assert state.portfolio.get_total_equity() == pytest.approx(8605.74)
    assert curve[pd.Timestamp("2021-06-01")] == 10_000
    assert curve[pd.Timestamp("2021-12-30")] == pytest.approx(8605.74)


def test_calculate_aggregated_returns(state: State):
    """Calculate monthly returns."""

    curve = calculate_equity_curve(state)

    # Calculate raw returns series
    returns = calculate_returns(curve)
    assert returns[pd.Timestamp("2021-06-01")] == 0
    assert returns[pd.Timestamp("2021-12-30")] != 0

    # Calculate monthly aggregate returns.
    # The test strategy should slowly bleed money in trade fees.
    monthly_aggregate = calculate_aggregate_returns(curve, "1M")
    assert monthly_aggregate["2021-07-31"] == pytest.approx(-0.07914776379460564)
    assert monthly_aggregate["2021-11-30"] == pytest.approx(-0.0030962288338045596)

    # December is incomplete so we do not get the date
    assert pd.Timestamp("2021-12-31") not in monthly_aggregate


def test_visualise_equity_performance(state: State):
    """Draw equity curve."""
    curve = calculate_equity_curve(state)
    returns = calculate_returns(curve)
    fig = visualise_equity_curve(returns)
    assert isinstance(fig, Figure)


def test_returns_over_time(state: State):
    """Draw monthly returns grid."""
    curve = calculate_equity_curve(state)
    returns = calculate_returns(curve)
    fig = visualise_returns_over_time(returns)
    assert isinstance(fig, Figure)


def test_returns_distribution(state: State):
    """Draw return distribution over time."""
    curve = calculate_equity_curve(state)
    returns = calculate_returns(curve)
    fig = visualise_returns_distribution(returns)
    assert isinstance(fig, Figure)


def test_web_compounding_realised_profit_export(state: State):
    """Export profit % to the web."""
    chart = render_web_chart(
        state,
        WebChartType.compounding_realised_profitability,
        WebChartSource.backtest,
    )

    assert chart.help_link == 'https://tradingstrategy.ai/glossary/profitability'
    assert chart.title == 'Compounded realised trading position profitability % on close'

    second_tuple = chart.data[1]  # See calculate_compounding_realised_trading_profitability(fill_current_time_gap)
    assert second_tuple[0] == 1623024000.0
    assert second_tuple[1] == -0.0033223057702593817


def test_compounding_unrealised_trading_profitability_sampled(state: State):
    """Export sampled profit % to the web."""
    chart = render_web_chart(
        state,
        WebChartType.compounding_unrealised_trading_profitability_sampled,
        WebChartSource.backtest,
    )

    assert chart.help_link == 'https://tradingstrategy.ai/glossary/profitability'
    assert len(chart.data) == 0  # Only available in live, because sampled hourly

def test_web_equity_curve(state: State):
    """Export equity curve the web."""
    chart = render_web_chart(
        state,
        WebChartType.total_equity,
        WebChartSource.backtest,
    )

    assert chart.help_link == 'https://tradingstrategy.ai/glossary/total-equity'

    first_tuple = chart.data[0]
    assert first_tuple[0] == 1622505600.0
    assert first_tuple[1] == 10000.0  # Initial deposit


def test_web_netflow(state: State):
    """Netflow curve the web."""
    chart = render_web_chart(
        state,
        WebChartType.netflow,
        WebChartSource.backtest,
    )

    assert chart.help_link == 'https://tradingstrategy.ai/glossary/netflow'

    first_tuple = chart.data[0]
    assert first_tuple[0] == 1622505600.0
    assert first_tuple[1] == 10000.0  # Initial deposit


def test_single_pair_timeline():
    """Create timeline DataFrame for a single pair."""

    dump_file = Path(os.path.join(os.path.dirname(__file__), "arbitrum-btc-usd-sls-binance-data-1h.json"))
    state = State.read_json_file(dump_file)
    df = expand_entries_and_exits(state)

    print(df)

