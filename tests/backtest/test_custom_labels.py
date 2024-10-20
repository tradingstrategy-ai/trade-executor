"""Add custom labels to trading pairs."""
import datetime

import pytest

from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.universe_model import default_universe_options, UniverseOptions
from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client
from tradingstrategy.timebucket import TimeBucket

from tradeexecutor.backtest.backtest_pricing import BacktestPricing
from tradeexecutor.backtest.backtest_routing import BacktestRoutingModel
from tradeexecutor.cli.log import setup_pytest_logging
from tradeexecutor.backtest.backtest_runner import run_backtest_inline
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, load_partial_data
from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode, unit_test_execution_context
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.reserve_currency import ReserveCurrency
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.strategy_module import StrategyParameters
from tradeexecutor.testing.synthetic_exchange_data import generate_simple_routing_model



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

    # Set custom labels on the token WETH on the trading pair
    weth_usdc = strategy_universe.get_pair_by_human_description(pairs[0])
    weth_usdc.base.set_tags({"L1", "EVM", "bluechip"})

    assert strategy_universe.data_universe.pairs.pair_map is not None, "Cache data structure missing?"
    assert len(weth_usdc.base.get_tags()) > 0

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
    # Show how to read pair and asset labels in decide_trade()
    for pair in input.strategy_universe.iterate_pairs():
        if "L1" in pair.get_tags():
            # Do some tradign logic for L1 tokens only
            pass

        # Unit test asserts
        import ipdb ; ipdb.set_trace()
        assert pair.get_tags() == {"L1", "evm", "bluechip"}
        assert pair.base.get_tags() == {"L1", "evm", "bluechip"}

    return []


def test_custom_tags(persistent_test_client, tmp_path):
    """Test custom labelling of trading pairs"""

    client = persistent_test_client

    # Start with $1M cash, far exceeding the market size
    class Parameters:
        backtest_start = datetime.datetime(2023, 1, 1)
        backtest_end = datetime.datetime(2023, 1, 7)
        initial_cash = 1_000_000
        cycle_duration = CycleDuration.cycle_1d
        trade_routing = TradeRouting.uniswap_v2_usdc

    universe = create_trading_universe(
        None,
        client,
        unit_test_execution_context,
        UniverseOptions.from_strategy_parameters_class(Parameters, unit_test_execution_context),
    )

    # Run the test
    result = run_backtest_inline(
        client=None,
        decide_trades=decide_trades,
        create_indicators=create_indicators,
        universe=universe,
        reserve_currency=ReserveCurrency.usdc,
        engine_version="0.5",
        parameters=StrategyParameters.from_class(Parameters),
        mode=ExecutionMode.unit_testing,
        trade_routing=TradeRouting.uniswap_v2_usdc
    )

    # Variables are readable after the backtest
    state = result.state
    assert len(state.other_data.data.keys()) == 29  # We stored data for 29 decide_trades cycles

    assert "L1" in result.strategy_universe.get_single_pair().get_tags()
