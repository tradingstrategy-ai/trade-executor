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

from tradeexecutor.backtest.backtest_routing import BacktestRoutingModel
from tradeexecutor.backtest.backtest_runner import run_backtest, setup_backtest_for_universe
from tradeexecutor.backtest.backtest_sync import BacktestSyncModel
from tradeexecutor.cli.log import setup_pytest_logging
from tradeexecutor.cli.loop import ExecutionTestHook
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, create_pair_universe_from_code
from tradeexecutor.testing.synthetic_ethereum_data import generate_random_ethereum_address
from tradeexecutor.testing.synthetic_exchange_data import generate_exchange, generate_simple_routing_model
from tradeexecutor.testing.synthetic_price_data import generate_ohlcv_candles
from tradeexecutor.visual.equity_curve import calculate_investment_flow, calculate_realised_profitability
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
        # Make sure we have some money in the bank on the first day
        if cycle == 1:
            sync_model.simulate_funding(datetime.datetime.utcnow(), Decimal(15))

        if cycle % 3 == 0:
            sync_model.simulate_funding(datetime.datetime.utcnow(), Decimal(100))

        if cycle % 5 == 0:
            sync_model.simulate_funding(datetime.datetime.utcnow(), Decimal(-90))

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

    return TradingStrategyUniverse(universe=universe, reserve_assets=[weth_usdc.quote])


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


def test_calculate_funding_flow(backtest_result: State):
    """Calculate funding flow for test deposits/redemptions."""
    state = backtest_result
    flow = calculate_investment_flow(state)
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
    assert 0.04 < max(profitability) < 0.06
    assert -0.06 < min(profitability) < -0.04
