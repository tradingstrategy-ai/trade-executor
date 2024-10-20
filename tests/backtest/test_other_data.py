"""Test other_data state data structures."""
import datetime
import random

import pytest

from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.liquidity import GroupedLiquidityUniverse
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe

from tradeexecutor.backtest.backtest_pricing import BacktestPricing
from tradeexecutor.backtest.backtest_routing import BacktestRoutingModel
from tradeexecutor.cli.log import setup_pytest_logging
from tradeexecutor.backtest.backtest_runner import run_backtest_inline
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, create_pair_universe_from_code
from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.reserve_currency import ReserveCurrency
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.strategy_module import StrategyParameters
from tradeexecutor.strategy.tvl_size_risk import USDTVLSizeRiskModel
from tradeexecutor.testing.synthetic_ethereum_data import generate_random_ethereum_address
from tradeexecutor.testing.synthetic_exchange_data import generate_exchange, generate_simple_routing_model
from tradeexecutor.testing.synthetic_price_data import generate_ohlcv_candles, generate_tvl_candles



@pytest.fixture(scope="module")
def logger(request):
    """Setup test logger."""
    return setup_pytest_logging(request, mute_requests=False)


@pytest.fixture(scope="module")
def strategy_universe() -> TradingStrategyUniverse:
    """Create ETH-USDC universe with only increasing data.

    - 1 months of data

    - Close price increase 1% every hour

    - Liquidity is a fixed 150,000 USD for the duration of the test
    """

    start_at = datetime.datetime(2021, 6, 1)
    end_at = datetime.datetime(2021, 7, 1)

    # Set up fake assets
    mock_chain_id = ChainId.ethereum
    mock_exchange = generate_exchange(
        exchange_id=random.randint(1, 1000),
        chain_id=mock_chain_id,
        address=generate_random_ethereum_address(),
        exchange_slug="my-dex",
    )
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

    # Create 1h underlying trade signal
    daily_candles = generate_ohlcv_candles(
        time_bucket,
        start_at,
        end_at,
        pair_id=weth_usdc.internal_id,
        daily_drift=(1.01, 1.01),
        high_drift=1.05,
        low_drift=0.90,
        random_seed=1,
    )
    candle_universe = GroupedCandleUniverse.create_from_single_pair_dataframe(daily_candles)
    universe = Universe(
        time_bucket=TimeBucket.d1,
        chains={mock_chain_id},
        exchanges={mock_exchange},
        pairs=pair_universe,
        candles=candle_universe,
    )
    universe.pairs.exchange_universe = universe.exchange_universe

    return TradingStrategyUniverse(
        data_universe=universe,
        reserve_assets=[usdc],
        backtest_stop_loss_time_bucket=time_bucket,
    )


@pytest.fixture()
def routing_model(synthetic_universe) -> BacktestRoutingModel:
    return generate_simple_routing_model(synthetic_universe)


@pytest.fixture()
def pricing_model(synthetic_universe, routing_model) -> BacktestPricing:
    pricing_model = BacktestPricing(
        synthetic_universe.data_universe.candles,
        routing_model,
        allow_missing_fees=True,
    )
    return pricing_model


def create_indicators(timestamp: datetime.datetime, parameters: StrategyParameters, strategy_universe: TradingStrategyUniverse, execution_context: ExecutionContext):
    # No indicators needed
    return IndicatorSet()


def decide_trades(input: StrategyInput) -> list[TradeExecution]:
    """Example of storing and loading custom variables."""

    cycle = input.cycle
    state = input.state

    # Saving values by cycle
    state.other_data.save(cycle, "my_value", 1)
    state.other_data.save(cycle, "my_value_2", [1, 2])
    state.other_data.save(cycle, "my_value_3", {1: 2})

    if cycle >= 2:
        # Loading latest values
        assert state.other_data.load_latest("my_value") == 1
        assert state.other_data.load_latest("my_value_2") == [1, 2]
        assert state.other_data.load_latest("my_value_3") == {1: 2}

    return []


def test_other_data(strategy_universe, tmp_path):
    """Test state.other_data."""

    # Start with $1M cash, far exceeding the market size
    class Parameters:
        backtest_start = strategy_universe.data_universe.candles.get_timestamp_range()[0].to_pydatetime()
        backtest_end = strategy_universe.data_universe.candles.get_timestamp_range()[1].to_pydatetime()
        initial_cash = 1_000_000
        cycle_duration = CycleDuration.cycle_1d

    # Run the test
    result = run_backtest_inline(
        client=None,
        decide_trades=decide_trades,
        create_indicators=create_indicators,
        universe=strategy_universe,
        reserve_currency=ReserveCurrency.usdc,
        engine_version="0.5",
        parameters=StrategyParameters.from_class(Parameters),
        mode=ExecutionMode.unit_testing,
    )

    # Variables are readable after the backtest
    state = result.state
    assert len(state.other_data.data.keys()) == 29  # We stored data for 29 decide_trades cycles
    assert state.other_data.data[1]["my_value"] == 1      # We can read historic values


labels to trading pairs."""