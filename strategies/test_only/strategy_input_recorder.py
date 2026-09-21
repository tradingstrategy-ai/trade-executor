"""Small real strategy used by the strategy-input recorder CLI test.

The test deliberately runs this module through the normal CLI bootstrap.  The
universe is built from the local Uniswap mock client supplied by the test
fixtures, so the test does not need an inline strategy or patched executor
methods.
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
    """Parameters consumed by the real v0.5 runner."""

    chain_id = ChainId.anvil
    cycle_duration = CycleDuration.cycle_1s
    routing = TradeRouting.user_supplied_routing_model
    initial_cash = 10_000
    backtest_start = datetime.datetime(2025, 1, 1)
    backtest_end = datetime.datetime(2025, 1, 2)
    required_history_period = datetime.timedelta(minutes=1)
    record_strategy_inputs = True


def create_trading_universe(input: CreateTradingUniverseInput) -> TradingStrategyUniverse:
    """Build a one-pair universe using the real local mock client."""

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
    """The recorder test needs the normal indicator lifecycle, with no data."""

    del timestamp, parameters, strategy_universe, execution_context
    return IndicatorSet()


def decide_trades(input: StrategyInput) -> list[TradeExecution]:
    """Record one ordinary strategy decision without placing a trade."""

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
