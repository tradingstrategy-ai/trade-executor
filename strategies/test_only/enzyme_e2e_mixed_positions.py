"""Dummy strategy used in Enzyme end-to-end test."""

import datetime

import pandas as pd
import pandas_ta

from tradingstrategy.lending import LendingProtocolType, LendingReserveDescription
from tradingstrategy.pair import HumanReadableTradingPairDescription
from tradingstrategy.utils.groupeduniverse import resample_candles
from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client
from tradingstrategy.timebucket import TimeBucket

from tradeexecutor.analysis.regime import Regime
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet, IndicatorSource
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode
from tradeexecutor.strategy.trading_strategy_universe import load_partial_data
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.state.visualisation import PlotKind
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput


trading_strategy_engine_version = "0.5"


class Parameters:
    """Parameteres for this strategy.

    - Collect parameters used for this strategy here

    - Both live trading and backtesting parameters
    """

    id = "enzyme-e2e-mixed" # Used in cache paths

    cycle_duration = CycleDuration.s1
    candle_time_bucket = TimeBucket.h1
    allocation = 0.98
    chain_id = ChainId.anvil
    routing = TradeRouting.default  # Pick default routes for trade execution
    required_history_period = datetime.timedelta(hours=200)

    backtest_start = datetime.datetime(2024, 5, 10)
    backtest_end = datetime.datetime(2024, 5, 15)
    initial_cash = 10_000


def get_strategy_trading_pairs(execution_mode: ExecutionMode) -> tuple[list[HumanReadableTradingPairDescription], list[LendingReserveDescription] | None]:
    trading_pairs = [
        (ChainId.polygon, "uniswap-v3", "WETH", "USDC"),
    ]

    lending_reserves = [
        (ChainId.polygon, LendingProtocolType.aave_v3, "USDC.e"),
    ]

    return trading_pairs, lending_reserves


def create_trading_universe(
    timestamp: datetime.datetime,
    client: Client,
    execution_context: ExecutionContext,
    universe_options: UniverseOptions,
) -> TradingStrategyUniverse:
    trading_pairs, lending_reserves = get_strategy_trading_pairs(execution_context.mode)

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
        liquidity=False,
        lending_reserves=lending_reserves,
    )
    # Construct a trading universe from the loaded data,
    # and apply any data preprocessing needed before giving it
    # to the strategy and indicators
    strategy_universe = TradingStrategyUniverse.create_from_dataset(
        dataset,
        reserve_asset="USDC",
        forward_fill=True,
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
    cash = position_manager.get_current_cash()
    trades = []

    if not position_manager.is_any_credit_supply_position_open():
        amount = cash * 0.9
        trades += position_manager.open_credit_supply_position_for_reserves(amount)
    elif input.cycle % 5 == 0:
        trades += position_manager.close_all()

    return trades  # Return the list of trades we made in this cycle

