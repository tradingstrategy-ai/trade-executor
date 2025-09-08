"""Backtest that only performs 1 trade and does not exit.
I.e. it if left with an open position at the end of the backtest."""

import pytest

"""Test backtesting where decide_trades and create_universe functions are passed directly.

"""
import logging
import random
import datetime
from typing import List, Dict

import pytest

import pandas as pd
from pandas_ta_classic.overlap import ema

from tradeexecutor.analysis.trade_analyser import build_trade_analysis, expand_timeline, expand_timeline_raw, TimelineRowStylingMode, TradeAnalysis, TradeSummary
from tradeexecutor.backtest.backtest_runner import run_backtest_inline
from tradeexecutor.cli.log import setup_pytest_logging
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.visualisation import PlotKind
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, \
    create_pair_universe_from_code
from tradeexecutor.testing.synthetic_ethereum_data import generate_random_ethereum_address
from tradeexecutor.testing.synthetic_exchange_data import generate_exchange, generate_simple_routing_model
from tradeexecutor.testing.synthetic_price_data import generate_ohlcv_candles
from tradeexecutor.visual.benchmark import visualise_equity_curve_benchmark
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
from tradeexecutor.strategy.strategy_type import StrategyType
from tradeexecutor.strategy.default_routing_options import TradeRouting


# relative tolerance for floating point tests
APPROX_REL = 1e-6


# How much of the cash to put on a single trade
position_size = 0.10

#
# Strategy thinking specific parameter
#

batch_size = 90

slow_ema_candle_count = 20

fast_ema_candle_count = 5


def decide_trades(
        timestamp: pd.Timestamp,
        universe: Universe,
        state: State,
        pricing_model: PricingModel,
        cycle_debug_data: Dict) -> List[TradeExecution]:
    """The brain function to decide the trades on each trading strategy cycle.
    
    In this particular instance, the logic can only ever execute one trade
    """

    # The pair we are trading
    pair = universe.pairs.get_single()

    # How much cash we have in the hand
    cash = state.portfolio.get_cash()

    # Create a position manager helper class that allows us easily to create
    # opening/closing trades for different positions
    position_manager = PositionManager(timestamp, universe, state, pricing_model)
    
    trades: List[TradeExecution] = []

    # Can only ever execute 1 trade with this logic
    if not position_manager.is_any_open():
        buy_amount = cash * position_size
        trades += position_manager.open_spot(pair, buy_amount)

    return trades


@pytest.fixture(scope="module")
def logger(request):
    """Setup test logger."""
    return setup_pytest_logging(request, mute_requests=False)


@pytest.fixture(scope="module")
def universe() -> TradingStrategyUniverse:
    """Set up a mock universe."""

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
    weth_usdc = TradingPairIdentifier(
        weth,
        usdc,
        generate_random_ethereum_address(),
        mock_exchange.address,
        internal_id=random.randint(1, 1000),
        internal_exchange_id=mock_exchange.exchange_id,
        fee=0.0030,
    )

    time_bucket = TimeBucket.d1

    pair_universe = create_pair_universe_from_code(mock_chain_id, [weth_usdc])

    candles = generate_ohlcv_candles(time_bucket, start_at, end_at, pair_id=weth_usdc.internal_id)
    candle_universe = GroupedCandleUniverse.create_from_single_pair_dataframe(candles)

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
def strategy_universe(universe) -> TradingStrategyUniverse:
    return universe


def test_ema_on_universe(strategy_universe):
    """Calculate exponential moving average on single pair candle universe."""
    start_timestamp = pd.Timestamp("2021-6-1")
    batch_size = 20
    candles = strategy_universe.data_universe.candles.get_single_pair_data(start_timestamp, sample_count=batch_size, allow_current=True, raise_on_not_enough_data=False)
    assert len(candles) == 1

    # Not enough data to calculate EMA - we haave only 1 sample
    ema_20_series = ema(candles["close"], length=20)
    assert ema_20_series is None

    end_timestamp = pd.Timestamp("2021-12-31")
    candles = strategy_universe.data_universe.candles.get_single_pair_data(end_timestamp, sample_count=batch_size, allow_current=True, raise_on_not_enough_data=False)
    assert len(candles) == batch_size

    ema_20_series = ema(candles["close"], length=20)
    assert pd.isna(ema_20_series.iloc[-2])
    assert float(ema_20_series.iloc[-1]) == pytest.approx(1955.019773)


# to avoid running backtest multiple times
@pytest.fixture(scope="module")
def backtest_result(
    strategy_universe
) -> tuple[State, TradingStrategyUniverse, dict]:
    start_at, end_at = strategy_universe.data_universe.candles.get_timestamp_range()

    routing_model = generate_simple_routing_model(strategy_universe)

    # Run the test
    state, strategy_universe, debug_dump = run_backtest_inline(
        start_at=start_at.to_pydatetime(),
        end_at=end_at.to_pydatetime(),
        client=None,  # None of downloads needed, because we are using synthetic data
        cycle_duration=CycleDuration.cycle_1d,  # Override to use 24h cycles despite what strategy file says
        decide_trades=decide_trades,
        create_trading_universe=None,
        universe=strategy_universe,
        initial_deposit=10_000,
        reserve_currency=ReserveCurrency.busd,
        trade_routing=TradeRouting.user_supplied_routing_model,
        routing_model=routing_model,
        log_level=logging.WARNING,
    )

    return state, strategy_universe, debug_dump


def test_run_inline_synthetic_backtest(
        logger: logging.Logger,
        backtest_result: tuple[State, TradingStrategyUniverse, dict],
    ):
    """Run the strategy backtest using inline decide_trades function.
    """

    state, universe, debug_dump = backtest_result
    assert len(debug_dump) == 215


@pytest.fixture(scope = "module")
def analysis(
    backtest_result: tuple[State, TradingStrategyUniverse, dict]
) -> TradeAnalysis:
    state, universe, debug_dump = backtest_result
    analysis = build_trade_analysis(state.portfolio)

    return analysis


@pytest.fixture(scope = "module")
def summary(
    backtest_result: tuple[State, TradingStrategyUniverse, dict],
    analysis: TradeAnalysis
) -> TradeSummary:

    state, universe, debug_dump = backtest_result

    summary = analysis.calculate_summary_statistics(state = state)

    # Should not cause exception
    summary.to_dataframe()

    return summary


def test_basic_summary_statistics(
    summary: TradeSummary,
):
    """Analyse synthetic trading strategy adv_stats.

    TODO: Might move this test to its own module.
    # TODO summary stat test with stop losses involved
    """

    assert summary.initial_cash == 10_000
    assert summary.won == 0
    assert summary.lost == 0
    assert summary.realised_profit == 0
    assert summary.open_value == pytest.approx(1054.2372274910792, rel=APPROX_REL)
    assert summary.end_value == pytest.approx(10054.23722749108, rel=APPROX_REL)
    assert summary.win_percent is None
    assert summary.duration == datetime.timedelta(days=213)
    assert summary.trade_volume == pytest.approx(1000.0, rel=APPROX_REL)
    assert summary.uninvested_cash == pytest.approx(9000.0, rel=APPROX_REL)

    assert summary.stop_losses == 0
    assert summary.take_profits == 0
    assert summary.total_positions == 0
    assert summary.undecided == 1
    assert summary.zero_loss == 0


def test_timeline(
    analysis: TradeAnalysis,
    backtest_result: tuple[State, TradingStrategyUniverse, dict],
):
    state, universe, debug_dump = backtest_result

    timeline = analysis.create_timeline()

    # Test expand timeline both colouring modes
    expanded_timeline, apply_styles = expand_timeline(
        universe.data_universe.exchanges,
        universe.data_universe.pairs,
        timeline,
        row_styling_mode=TimelineRowStylingMode.simple,
    )

    # Check HTML output does not crash
    # https://github.com/pandas-dev/pandas/issues/19358#issuecomment-359733504
    apply_styles(expanded_timeline).to_html()

    assert(len(expanded_timeline) == 1)
    

def test_timeline_raw(
    analysis: TradeAnalysis,
    backtest_result: tuple[State, TradingStrategyUniverse, dict],
):
    state, universe, debug_dump = backtest_result

    timeline = analysis.create_timeline()

    expanded_timeline_raw = expand_timeline_raw(
        timeline,
    )

    assert(len(expanded_timeline_raw) == 1)

def test_benchmark_synthetic_trading_portfolio(
    logger: logging.Logger,
    strategy_universe,
):
    """Build benchmark figures.

    TODO: Might move this test to its own module.
    """

    start_at, end_at = strategy_universe.data_universe.candles.get_timestamp_range()

    routing_model = generate_simple_routing_model(strategy_universe)

    # Run the test
    state, strategy_universe, debug_dump = run_backtest_inline(
        start_at=start_at.to_pydatetime(),
        end_at=end_at.to_pydatetime(),
        client=None,  # None of downloads needed, because we are using synthetic data
        cycle_duration=CycleDuration.cycle_1d,  # Override to use 24h cycles despite what strategy file says
        decide_trades=decide_trades,
        create_trading_universe=None,
        universe=strategy_universe,
        initial_deposit=10_000,
        reserve_currency=ReserveCurrency.busd,
        trade_routing=TradeRouting.user_supplied_routing_model,
        routing_model=routing_model,
        log_level=logging.WARNING,
    )

    # Visualise performance
    fig = visualise_equity_curve_benchmark(
        state.name,
        portfolio_statistics=state.stats.portfolio,
        all_cash=100_000,
        buy_and_hold_asset_name="ETH",
        buy_and_hold_price_series=strategy_universe.data_universe.candles.get_single_pair_data()["close"],
    )

    # Check that the diagram has 3 plots
    assert len(fig.data) == 3
