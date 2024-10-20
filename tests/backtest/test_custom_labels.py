"""Add custom labels to trading pairs."""
import datetime
import random

import pytest

from tradeexecutor.strategy.universe_model import default_universe_options, UniverseOptions
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.liquidity import GroupedLiquidityUniverse
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe

from tradeexecutor.backtest.backtest_pricing import BacktestPricing
from tradeexecutor.backtest.backtest_routing import BacktestRoutingModel
from tradeexecutor.cli.log import setup_pytest_logging
from tradeexecutor.backtest.backtest_runner import run_backtest_inline
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, create_pair_universe_from_code, load_partial_data
from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode, unit_test_execution_context
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



def create_trading_universe(
    ts: datetime.datetime,
    client: Client,
    execution_context: ExecutionContext,
    universe_options: UniverseOptions,
) -> TradingStrategyUniverse:

    assert universe_options.start_at

    pairs = [
        (ChainId.polygon, "uniswap-v3", "WETH", "USDC", 0.0005)
    ]

    dataset = load_partial_data(
        client,
        execution_context=unit_test_execution_context,
        time_bucket=TimeBucket.d1,
        pairs=pairs,
        universe_options=default_universe_options,
        start_at=universe_options.start_at,
        end_at=universe_options.end_at,
    )

    strategy_universe = TradingStrategyUniverse.create_single_pair_universe(dataset)

    # Set custom labels on a trading pair
    weth_usdc = strategy_universe.get_pair_by_human_description(pairs[0])

    # Add some custom tags on the trading pair
    weth_usdc.base.set_tags({"L1", "EVM", "bluechip"})

    return strategy_universe


@pytest.fixture(scope="module")
def logger(request):
    """Setup test logger."""
    return setup_pytest_logging(request, mute_requests=False)


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

    for pair in input.strategy_universe.iterate_pairs():
        if "L1" in pair.get_tags():
            # Do some tradign logic for L1 tokens only
            pass

        # Unit test asserts
        assert pair.get_tags() == {"L1", "evm", "bluechip"}
        assert pair.base.get_tags() == {"L1", "evm", "bluechip"}

    return []


def test_custom_tags(persistent_test_client, tmp_path):
    """Test custom labelling of trading pairs"""

    client = persistent_test_client

    # Start with $1M cash, far exceeding the market size
    class Parameters:
        backtest_start = datetime.datetime(2020, 1, 1)
        backtest_end = datetime.datetime(2020, 1, 7)
        initial_cash = 1_000_000
        cycle_duration = CycleDuration.cycle_1d

    # Run the test
    result = run_backtest_inline(
        client=None,
        decide_trades=decide_trades,
        create_indicators=create_indicators,
        create_trading_universe=create_trading_universe,
        reserve_currency=ReserveCurrency.usdc,
        engine_version="0.5",
        parameters=StrategyParameters.from_class(Parameters),
        mode=ExecutionMode.unit_testing,
    )

    # Variables are readable after the backtest
    state = result.state
    assert len(state.other_data.data.keys()) == 29  # We stored data for 29 decide_trades cycles
    assert state.other_data.data[1]["my_value"] == 1      # We can read historic values

