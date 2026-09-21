"""Exercise strategy-input recording through the real live CLI call path.

``tests/cli/test_cli_strategy_input_recorder.py`` loads this checked-in module
through normal strategy discovery, bootstrap, scheduling, and
``PandasTraderRunner`` calls. The universe comes from the local Uniswap mock
client supplied by test fixtures, which makes the use case reproducible without
an inline strategy or patched executor methods.
"""

import datetime

from tradingstrategy.chain import ChainId
from tradingstrategy.pair import DEXPair
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.testing.uniswap_v2_mock_client import UniswapV2MockClient

from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.pandas_trader.trading_universe_input import CreateTradingUniverseInput
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.reserve_currency import ReserveCurrency
from tradeexecutor.strategy.strategy_type import StrategyType
from tradeexecutor.strategy.trading_strategy_universe import (
    TradingStrategyUniverse,
    load_all_data,
)


TRADING_STRATEGY_ENGINE_VERSION = "0.5"
TRADING_STRATEGY_TYPE = StrategyType.managed_positions
TRADING_STRATEGY_CYCLE = CycleDuration.cycle_1s
TRADE_ROUTING = TradeRouting.user_supplied_routing_model
RESERVE_CURRENCY = ReserveCurrency.usdc


class Parameters:
    """Configure the shortest realistic live run that enables recording.

    Strategy-module loading passes this conventional class to
    ``StrategyParameters.from_class()``. It intentionally follows the existing
    v0.5 class-style parameter contract rather than being a dataclass, so the
    black-box test exercises the same loader used by production strategies.
    """

    #: Use the local Anvil chain provided by the CLI integration test.
    chain_id = ChainId.anvil
    #: Exercise the recorder on the shortest scheduler cadence supported by the test.
    cycle_duration = CycleDuration.cycle_1s
    #: The mock strategy does not place trades, but the runner still needs routing.
    routing = TradeRouting.user_supplied_routing_model
    #: Required strategy-module cash setting for the local live run.
    initial_cash = 10_000
    #: Required historical range for the strategy-module contract.
    backtest_start = datetime.datetime(2025, 1, 1)
    #: Required historical range for the strategy-module contract.
    backtest_end = datetime.datetime(2025, 1, 2)
    #: Request enough local history for normal universe creation.
    required_history_period = datetime.timedelta(minutes=1)
    #: Enable the framework recorder that this strategy exists to exercise.
    record_strategy_inputs = True


def create_trading_universe(input: CreateTradingUniverseInput) -> TradingStrategyUniverse:
    """Build the one-pair universe consumed by each recorded test decision.

    ``DefaultTradingStrategyUniverseModel`` calls this during CLI bootstrap.
    Using the injected ``UniswapV2MockClient`` proves normal universe creation
    reaches the recorder and avoids substituting fake runner or universe
    methods in this black-box test.

    :param input:
        Standard v0.5 universe-construction inputs created by CLI bootstrap.
    :return:
        A single-pair universe backed by local deterministic fixture data.
    """
    assert isinstance(input.client, UniswapV2MockClient)
    dataset = load_all_data(
        input.client,
        TimeBucket.not_applicable,
        input.execution_context,
        input.universe_options,
    )
    pair_data = dataset.pairs.iloc[0]
    pair = DEXPair.from_dict(pair_data.to_dict())
    return TradingStrategyUniverse.create_single_pair_universe(
        dataset,
        pair.chain_id,
        pair.exchange_slug,
        pair.base_token_symbol,
        pair.quote_token_symbol,
    )


def create_indicators(
    timestamp: datetime.datetime | None,
    parameters: StrategyParameters,
    strategy_universe: TradingStrategyUniverse,
    execution_context: ExecutionContext,
) -> IndicatorSet:
    """Create an intentionally empty indicator set through the normal callback.

    The live runner calls this for each decision cycle. Empty indicators keep
    the test focused on recorder wiring while still exercising indicator
    calculation and capture lifecycle instead of bypassing it.
    """
    del timestamp, parameters, strategy_universe, execution_context
    return IndicatorSet()


def decide_trades(input: StrategyInput) -> list[TradeExecution]:
    """Record one complete live decision without introducing trade execution.

    ``PandasTraderRunner.on_clock()`` calls this twice in the CLI test. The
    explicit begin/record/finish/error pattern demonstrates the intended
    strategy call site, while returning no trades isolates recorder lifecycle
    behaviour from routing and settlement.
    """
    assert input.recorder is not None
    input.recorder.begin(input)
    try:
        input.recorder.record(
            "calculation",
            "cli_cycle",
            {"cycle": input.cycle, "pair_count": input.strategy_universe.get_pair_count()},
        )
        input.recorder.finish([])
        return []
    except Exception as error:
        input.recorder.fail(error)
        raise
