"""Test different profit / equity calculation models.

- We do this by using simulated deposit/redemption events in a backtest run for synthetic trade data
  and a dummy strategy

"""
import datetime
import logging
import os
from decimal import Decimal
from pathlib import Path

import pytest
import pandas as pd

from tradeexecutor.backtest.backtest_routing import BacktestRoutingModel
from tradeexecutor.backtest.backtest_runner import run_backtest, setup_backtest_for_universe
from tradeexecutor.backtest.backtest_sync import BacktestSyncModel
from tradeexecutor.cli.log import setup_pytest_logging
from tradeexecutor.cli.loop import ExecutionTestHook
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.statistics.summary import calculate_summary_statistics
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, create_pair_universe_from_code
from tradeexecutor.testing.synthetic_ethereum_data import generate_random_ethereum_address
from tradeexecutor.testing.synthetic_exchange_data import generate_exchange, generate_simple_routing_model
from tradeexecutor.testing.synthetic_price_data import generate_ohlcv_candles
from tradeexecutor.visual.equity_curve import calculate_investment_flow, calculate_realised_profitability, calculate_deposit_adjusted_returns, \
    calculate_compounding_realised_trading_profitability, calculate_size_relative_realised_trading_returns, calculate_cumulative_daily_returns, calculate_non_cumulative_daily_returns
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import Exchange
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe



class DepositSimulator(ExecutionTestHook):
    """Do FizzBuzz deposits/redemptions simulation."""

    def __init__(self):
        self.deposit_callbacks_done = 0

    def on_before_cycle(
            self,
            cycle: int,
            cycle_st: datetime.datetime,
            state: State,
            sync_model: BacktestSyncModel

    ):
        # Assume the deposit was made 5 minutes before the strategy cycle
        fund_event_ts = cycle_st - datetime.timedelta(minutes=5)

        # Make sure we have some money in the bank on the first day
        if cycle == 1:
            sync_model.simulate_funding(fund_event_ts, Decimal(15))

        if cycle % 3 == 0:
            sync_model.simulate_funding(fund_event_ts, Decimal(100))

        if cycle % 5 == 0:
            sync_model.simulate_funding(fund_event_ts, Decimal(-90))

        self.deposit_callbacks_done += 1


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
        fee=0.0030,
    )


@pytest.fixture(scope="module")
def strategy_path() -> Path:
    """Where do we load our strategy file."""
    return Path(os.path.join(os.path.dirname(__file__), "../..", "strategies", "test_only", "pandas_buy_every_second_day.py"))


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

    return TradingStrategyUniverse(data_universe=universe, reserve_assets=[weth_usdc.quote])


@pytest.fixture(scope="module")
def routing_model(synthetic_universe) -> BacktestRoutingModel:
    return generate_simple_routing_model(synthetic_universe)


@pytest.fixture(scope="module")
def backtest_result(
        logger: logging.Logger,
        strategy_path: Path,
        synthetic_universe: TradingStrategyUniverse,
        routing_model: BacktestRoutingModel,
    ) -> State:
    """Run the strategy backtest.

    - Use synthetic data

    - Run a strategy for 6 months
    """

    # Run the test
    setup = setup_backtest_for_universe(
        strategy_path,
        start_at=datetime.datetime(2021, 6, 1),
        end_at=datetime.datetime(2022, 1, 1),
        cycle_duration=CycleDuration.cycle_1d,  # Override to use 24h cycles despite what strategy file says
        candle_time_frame=TimeBucket.d1,  # Override to use 24h cycles despite what strategy file says
        initial_deposit=0,
        universe=synthetic_universe,
        routing_model=routing_model,
        allow_missing_fees=True,
    )

    deposit_simulator = DepositSimulator()
    state, universe, debug_dump = run_backtest(
        setup,
        allow_missing_fees=True,
        execution_test_hook=deposit_simulator,
    )

    # Some smoke checks we generated good data
    assert deposit_simulator.deposit_callbacks_done > 10, "No deposit/redemption activity detected"

    all_positions = list(state.portfolio.get_all_positions())
    assert len(all_positions) == 107

    return state


@pytest.fixture(scope="module")
def backtest_result_hourly(
        logger: logging.Logger,
        strategy_path: Path,
        synthetic_universe: TradingStrategyUniverse,
        routing_model: BacktestRoutingModel,
    ) -> State:
    """Run the strategy backtest.

    - Use synthetic data

    - Run a strategy for 6 months
    """

    # Run the test
    setup = setup_backtest_for_universe(
        strategy_path,
        start_at=datetime.datetime(2021, 6, 1),
        end_at=datetime.datetime(2021, 7, 1),
        cycle_duration=CycleDuration.cycle_1h,  # Override to use 1h cycles despite what strategy file says
        candle_time_frame=TimeBucket.h1,  # Override to use 1h cycles despite what strategy file says
        initial_deposit=0,
        universe=synthetic_universe,
        routing_model=routing_model,
        allow_missing_fees=True,
    )

    deposit_simulator = DepositSimulator()
    state, universe, debug_dump = run_backtest(
        setup,
        allow_missing_fees=True,
        execution_test_hook=deposit_simulator,
    )

    # Some smoke checks we generated good data
    assert deposit_simulator.deposit_callbacks_done > 10, "No deposit/redemption activity detected"

    all_positions = list(state.portfolio.get_all_positions())
    assert len(all_positions) == 360

    return state


def test_calculate_funding_flow(backtest_result: State):
    """Calculate funding flow for test deposits/redemptions."""
    state = backtest_result
    flow = calculate_investment_flow(state)
    assert isinstance(flow.index, pd.DatetimeIndex)
    assert max(flow) == 100, "Deposit simulation did not work out"
    assert min(flow) == -90, "Redemption simulation did not work out"
    deposits = flow[flow > 0]  # https://stackoverflow.com/a/28272238/315168
    redemptions = flow[flow < 0]
    assert sum(deposits) == 7115.0
    assert sum(redemptions) == -3780.0


def test_calculate_realised_trading_profitability(backtest_result: State):
    """Calculate the realised trading profitability."""
    state = backtest_result
    profitability = calculate_realised_profitability(state)
    assert isinstance(profitability, pd.Series)
    assert 0.04 < max(profitability) < 0.06
    assert -0.06 < min(profitability) < -0.04
    assert isinstance(profitability.index, pd.DatetimeIndex)

    sized_profitability = calculate_size_relative_realised_trading_returns(state)
    assert sized_profitability.iloc[0] == pytest.approx(profitability.iloc[0] / 10, rel=0.01)  # 10% position size is used

    compounded_profitability = calculate_compounding_realised_trading_profitability(state)
    assert isinstance(compounded_profitability, pd.Series)
    assert -0.05 < compounded_profitability.iloc[0] < 0.01
    assert compounded_profitability.iloc[-1] == pytest.approx(-0.04583804672389613)  # Strategy has destroyed 4.5% of capital


def test_calculate_realised_trading_profitability_fill_gap(backtest_result: State):
    """Calculate the realised trading profitability, filling gap to the latest date.

    By default, always insert a bookkeeping market at the last available date,
    so web frontend can deal with the data easier.
    """
    compounded_profitability = calculate_compounding_realised_trading_profitability(backtest_result)
    compounded_profitability.index[-1] == pd.Timestamp('2022-01-01 00:00:00')
    last = compounded_profitability.index[-1]
    second_last = compounded_profitability.index[-2]
    assert last == pd.Timestamp('2022-01-01 00:00:00')
    assert second_last == pd.Timestamp('2021-12-31 00:00:00')
    last_val = compounded_profitability[last]
    second_last_val = compounded_profitability[second_last]
    assert last_val == second_last_val


def test_daily_returns(backtest_result_hourly: State):
    """Test daily returns calculation by using two different methods
    and comparing the results.
    """
    cum_profit = calculate_cumulative_daily_returns(backtest_result_hourly)
    assert isinstance(cum_profit, pd.Series)
    
    # remove last entry, since it was added to fill the gap
    cum_profit = cum_profit[:-1]

    non_cum_profit = calculate_non_cumulative_daily_returns(backtest_result_hourly)
    assert isinstance(non_cum_profit, pd.Series)

    cum_profit_2 = non_cum_profit.add(1).cumprod().sub(1)
    
    assert cum_profit.index.equals(cum_profit_2.index)
    
    assert all(cum_profit - cum_profit_2 < 1e-10)
    
    assert cum_profit_2.iloc[-3] == pytest.approx(cum_profit.iloc[-3], abs=1e-10)
    assert cum_profit_2.iloc[-2] == pytest.approx(cum_profit.iloc[-2], abs=1e-10)
    assert cum_profit_2.iloc[-1] == pytest.approx(cum_profit.iloc[-1], abs=1e-10)
    

def test_profitabilities_are_same(backtest_result_hourly: State):
    """
    Check that two methods of calculating profit yield the same result
    """
    summary_stats = calculate_summary_statistics(backtest_result_hourly, time_window=datetime.timedelta(days=2000), key_metrics_backtest_cut_off=datetime.timedelta(days=0), cycle_duration=CycleDuration.cycle_1h)
    
    assert summary_stats.return_all_time == pytest.approx(-0.1937959274935105)
    assert summary_stats.key_metrics['profitability'].value == pytest.approx(summary_stats.return_all_time, abs=1e-14)
    

def test_key_metrics(backtest_result_hourly: State):
    """Check that the all key metrics are correct"""
    summary_stats = calculate_summary_statistics(backtest_result_hourly, time_window=datetime.timedelta(days=2000), key_metrics_backtest_cut_off=datetime.timedelta(days=0), cycle_duration=CycleDuration.cycle_1h)
    
    # correct since standard dev = 0
    assert summary_stats.key_metrics['sharpe'].value == float('-inf')
    assert summary_stats.key_metrics['sortino'].value == pytest.approx(-19.104973174542796)
    assert summary_stats.key_metrics['max_drawdown'].value == pytest.approx(0.18798605414266745)
    assert summary_stats.key_metrics['profitability'].value == pytest.approx(-0.1937959274935087)


def test_calculate_realised_trading_profitability_no_trades():
    """Do not crash when calculating realised trading profitability if there are no trades."""
    state = State()
    profitability = calculate_realised_profitability(state)
    assert len(profitability) == 0

    compounded_profitability = calculate_compounding_realised_trading_profitability(state)
    assert len(compounded_profitability) == 0


def test_calculate_deposit_adjusted_returns(backtest_result: State):
    """Calculate the deposit adjusted returns."""
    state = backtest_result

    returns = calculate_deposit_adjusted_returns(state)

    # Dollar based profitability,
    # the strategy is not profitable but loses constantly
    # money on fees
    assert -20 < max(returns) <= 90
    assert -150 < min(returns) <= -100


def test_calculate_deposit_adjusted_returns_no_trades():
    """Do not crash calculating the deposit adjusted returns if there are no trades."""
    state = State()
    returns = calculate_deposit_adjusted_returns(state)
    assert len(returns) == 0