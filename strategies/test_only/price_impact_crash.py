"""See test_price_impact_crash."""

import datetime

from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradingstrategy.pair import HumanReadableTradingPairDescription
from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client
from tradingstrategy.timebucket import TimeBucket

from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode
from tradeexecutor.strategy.trading_strategy_universe import load_partial_data
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput


trading_strategy_engine_version = "0.5"


class Parameters:
    id = "price-impact-crash"
    cycle_duration = CycleDuration.s1
    candle_time_bucket = TimeBucket.h1
    chain_id = ChainId.anvil
    required_history_period = datetime.timedelta(hours=24)
    routing = TradeRouting.default

    # Set price impact parameter so that it always crashes
    max_price_impact = 0.00000001

    # Unused
    backtest_start = datetime.datetime(2024, 5, 10)
    backtest_end = datetime.datetime(2024, 5, 15)
    initial_cash = 10_000



def get_strategy_trading_pairs(execution_mode: ExecutionMode) -> list[HumanReadableTradingPairDescription]:
    trading_pairs = [
        (ChainId.polygon, "uniswap-v3", "WETH", "USDC"),
    ]

    return trading_pairs


def create_trading_universe(
    timestamp: datetime.datetime,
    client: Client,
    execution_context: ExecutionContext,
    universe_options: UniverseOptions,
) -> TradingStrategyUniverse:
    trading_pairs = get_strategy_trading_pairs(execution_context.mode)

    dataset = load_partial_data(
        client=client,
        time_bucket=Parameters.candle_time_bucket,
        pairs=trading_pairs,
        execution_context=execution_context,
        universe_options=UniverseOptions(
            history_period=Parameters.required_history_period,
            start_at=None,
            end_at=None,
        ),
    )
    # Construct a trading universe from the loaded data,
    # and apply any data preprocessing needed before giving it
    # to the strategy and indicators
    strategy_universe = TradingStrategyUniverse.create_from_dataset(
        dataset,
        reserve_asset="USDC",
    )

    return strategy_universe


def create_indicators(
    timestamp: datetime.datetime | None,
    parameters: StrategyParameters,
    strategy_universe: TradingStrategyUniverse,
    execution_context: ExecutionContext
):
    return IndicatorSet()


def decide_trades(input: StrategyInput) -> list[TradeExecution]:
    position_manager = input.get_position_manager()
    pair = input.strategy_universe.get_single_pair()
    # Attempt to open ETH spot position for 100 USD
    # Will always crash because of Parameters.max_price_impact
    trades = position_manager.open_spot(
        pair,
        value=100.00,
    )
    return trades

