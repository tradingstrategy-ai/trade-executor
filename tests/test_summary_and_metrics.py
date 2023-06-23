"""Summary statistics calculations test.

Test summary calculation formulas on a synthetic backtest with random data.
"""
import datetime
import logging
import os
from pathlib import Path

import pytest

import pandas as pd

from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import Exchange
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe

from tradeexecutor.analysis.advanced_metrics import calculate_advanced_metrics
from tradeexecutor.backtest.backtest_routing import BacktestRoutingModel
from tradeexecutor.backtest.backtest_runner import run_backtest, setup_backtest_for_universe
from tradeexecutor.cli.log import setup_pytest_logging
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.state.statistics import Statistics, calculate_naive_profitability
from tradeexecutor.state.validator import validate_nested_state_dict
from tradeexecutor.statistics.key_metric import calculate_key_metrics
from tradeexecutor.statistics.summary import calculate_summary_statistics
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.execution_context import ExecutionMode
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, create_pair_universe_from_code
from tradeexecutor.testing.synthetic_ethereum_data import generate_random_ethereum_address
from tradeexecutor.testing.synthetic_exchange_data import generate_exchange, generate_simple_routing_model
from tradeexecutor.testing.synthetic_price_data import generate_ohlcv_candles

from tradeexecutor.visual.equity_curve import calculate_equity_curve, calculate_returns


@pytest.fixture(scope="module")
def logger(request):
    """Setup test logger."""
    return setup_pytest_logging(request, mute_requests=False)


@pytest.fixture(scope="module")
def mock_chain_id() -> ChainId:
    """Mock a chai id."""
    return ChainId.ethereum


@pytest.fixture(scope="module")
def mock_exchange(mock_chain_id) -> Exchange:
    """Mock an exchange."""
    return generate_exchange(exchange_id=1, chain_id=mock_chain_id, address=generate_random_ethereum_address())


@pytest.fixture(scope="module")
def usdc() -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(ChainId.ethereum.value, generate_random_ethereum_address(), "USDC", 6, 1)


@pytest.fixture(scope="module")
def weth() -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(ChainId.ethereum.value, generate_random_ethereum_address(), "WETH", 18, 2)


@pytest.fixture(scope="module")
def weth_usdc(mock_exchange, usdc, weth) -> TradingPairIdentifier:
    """Mock some assets"""
    return TradingPairIdentifier(
        weth,
        usdc,
        generate_random_ethereum_address(),
        mock_exchange.address,
        internal_id=555,
        internal_exchange_id=mock_exchange.exchange_id,
        fee=0.0030
    )


@pytest.fixture(scope="module")
def strategy_path() -> Path:
    """Where do we load our strategy file."""
    return Path(os.path.join(os.path.dirname(__file__), "..", "strategies", "test_only", "pandas_buy_every_second_day.py"))


@pytest.fixture(scope="module")
def synthetic_universe(mock_chain_id, mock_exchange, weth_usdc) -> TradingStrategyUniverse:
    """Generate synthetic trading data universe for a single trading pair.

    - Single mock exchange

    - Single mock trading pair

    - Random candles

    - No liquidity data available
    """

    start_date = datetime.datetime(2021, 6, 1)
    end_date = datetime.datetime(2022, 1, 1)

    time_bucket = TimeBucket.d1

    pair_universe = create_pair_universe_from_code(mock_chain_id, [weth_usdc])

    # Generate candles for pair_id = 1
    candles = generate_ohlcv_candles(time_bucket, start_date, end_date, pair_id=weth_usdc.internal_id)
    candle_universe = GroupedCandleUniverse.create_from_single_pair_dataframe(candles)

    universe = Universe(
        time_bucket=time_bucket,
        chains={mock_chain_id},
        exchanges={mock_exchange},
        pairs=pair_universe,
        candles=candle_universe,
        liquidity=None
    )

    return TradingStrategyUniverse(universe=universe, reserve_assets=[weth_usdc.quote])


@pytest.fixture(scope="module")
def routing_model(synthetic_universe) -> BacktestRoutingModel:
    return generate_simple_routing_model(synthetic_universe)


@pytest.fixture(scope="module")
def state(
        logger: logging.Logger,
        strategy_path: Path,
        synthetic_universe: TradingStrategyUniverse,
        routing_model: BacktestRoutingModel,
    ):
    """Run a simple strategy backtest.

    Calculate some statistics based on it.
    """

    # Run the test
    setup = setup_backtest_for_universe(
        strategy_path,
        start_at=datetime.datetime(2021, 6, 1),
        end_at=datetime.datetime(2022, 1, 1),
        cycle_duration=CycleDuration.cycle_1d,  # Override to use 24h cycles despite what strategy file says
        candle_time_frame=TimeBucket.d1,  # Override to use 24h cycles despite what strategy file says
        initial_deposit=10_000,
        universe=synthetic_universe,
        routing_model=routing_model,
    )

    state, universe, debug_dump = run_backtest(setup)

    return state


def test_get_statistics_as_dataframe(state: State):
    """We can convert any of our statistics to dataframes"""

    stats: Statistics = state.stats

    assert len(stats.portfolio) > 0, "Portfolio progress over the history statistics were not available"

    # Create time series of portfolio "total_equity" over its lifetime
    s = stats.get_portfolio_statistics_dataframe("total_equity")

    assert isinstance(s.index, pd.DatetimeIndex)
    assert s.index[0] == pd.Timestamp('2021-06-01 00:00:00')
    assert s.index[-1] == pd.Timestamp('2021-12-31 00:00:00')

    # First value by date
    assert s.loc[pd.Timestamp('2021-06-01 00:00:00')] == 9994.017946161515

    # Last value by index
    assert s.iloc[-1] == 9541.619532761046

    # Number of positoins
    assert len(s) == 214


def test_calculate_profitability_90_days(state: State):
    """Calculate strategy profitability for last 90 days"""

    stats: Statistics = state.stats

    # Create time series of portfolio "total_equity" over its lifetime
    s = stats.get_portfolio_statistics_dataframe("total_equity")
    profitability_90_days, time_window = calculate_naive_profitability(s, look_back=pd.Timedelta(days=90))

    # Calculate last 90 days
    assert profitability_90_days == pytest.approx(-0.019484747529770786)
    assert time_window == pd.Timedelta(days=90)


def test_calculate_profitability_overflow_time_window(state: State):
    """Calculate strategy profitability but do not have enough data.

    """
    stats: Statistics = state.stats
    s = stats.get_portfolio_statistics_dataframe("total_equity")

    # Attempt to calculate last 10 years
    profitability_10_years, time_window = calculate_naive_profitability(s, look_back=pd.Timedelta(days=10*365))

    assert time_window == pd.Timedelta('213 days 00:00:00')
    assert profitability_10_years == pytest.approx(-0.045266920255454)


def test_calculate_profitability_empty():
    """Calculate strategy profitability but we have no data yet.

    """
    s = pd.Series([], index=pd.DatetimeIndex([]))

    profitability_90_days, time_window = calculate_naive_profitability(s, look_back=pd.Timedelta(days=90))

    assert time_window is None
    assert profitability_90_days is None


def test_calculate_all_summary_statistics_empty():
    """Calculate all summary statistics on empty state.

    """
    # Set "last 90 days" to the end of backtest data
    now_ = pd.Timestamp(datetime.datetime(2021, 12, 31, 0, 0))

    state = State()
    calculate_summary_statistics(
        state,
        ExecutionMode.unit_testing_trading,
        now_=now_,
    )

def test_calculate_all_summary_statistics(state: State):
    """Calculate all summary statistics.

    """

    # Set "last 90 days" to the end of backtest data
    now_ = pd.Timestamp(datetime.datetime(2021, 12, 31, 0, 0))

    summary = calculate_summary_statistics(
        state,
        ExecutionMode.unit_testing_trading,
        now_=now_,
    )

    assert summary.calculated_at
    assert summary.enough_data
    assert summary.first_trade_at == datetime.datetime(2021, 6, 1, 0, 0)
    assert summary.last_trade_at == datetime.datetime(2021, 12, 31, 0, 0)
    assert summary.current_value == pytest.approx(9541.619532761046)
    true_profitability = 9541.619532761046 / 10_000 - 1
    assert true_profitability == pytest.approx(-0.045838046723895465)
    assert summary.profitability_90_days == pytest.approx(-0.045838046723895465)

    datapoints = summary.performance_chart_90_days
    assert len(datapoints) == 91

    assert datapoints[0] == (datetime.datetime(2021, 10, 2, 0, 0), -0.02687699056973558)
    assert datapoints[-1] == (datetime.datetime(2021, 12, 31, 0, 0), -0.045838046723895576)

    # Make sure we do not output anything that is not JSON'able
    data = summary.to_dict()
    validate_nested_state_dict(data)


def test_advanced_metrics(state: State):
    """Quantstats metrics calculations."""

    equity = calculate_equity_curve(state)
    returns = calculate_returns(equity)
    metrics = calculate_advanced_metrics(returns)

    # Each metric as a series. Index 0 is our performance,
    # index 1 is the benchmark.
    sharpe = metrics.loc["Sharpe"][0]
    assert sharpe == pytest.approx(-2.09)


def test_calculate_key_metrics():
    """Calculate web frontend key metric for an empty state and no backtest."""

    state = State()
    metrics = {m.kind.value: m for m in calculate_key_metrics(state)}
    import ipdb ; ipdb.set_trace()
