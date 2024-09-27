"""Dummy strategy used in integration tests.

- Run single cycle

- Always buy WETH-USDT on Uniswap
"""
import datetime

from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.parameters import StrategyParameters
from tradingstrategy.chain import ChainId

from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, load_pair_data_for_single_exchange, load_partial_data
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradingstrategy.client import Client
from tradingstrategy.timebucket import TimeBucket
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.reserve_currency import ReserveCurrency


trading_strategy_engine_version = "0.5"


class Parameters:
    chain_id = ChainId.arbitrum
    routing = TradeRouting.default
    cycle_duration = CycleDuration.cycle_1s
    reserve_currency = ReserveCurrency.usdt
    candle_time_bucket = TimeBucket.d7
    trading_pair = (ChainId.arbitrum, "uniswap-v3", "WETH", "USDT", 0.0005)
    required_history_period = datetime.timedelta(weeks=2)
    backtest_start = None
    backtest_end = None
    initial_cash = 10_000


def decide_trades(input: StrategyInput) -> list[TradeExecution]:
    position_manager = input.get_position_manager()
    pair = input.strategy_universe.get_single_pair()
    cash = position_manager.get_current_cash()
    trades = []
    if not position_manager.is_any_open():
        position_size = 0.10
        buy_amount = cash * position_size
        trades += position_manager.open_spot(pair, buy_amount)
    return trades


def create_trading_universe(
    ts: datetime.datetime,
    client: Client,
    execution_context: ExecutionContext,
    universe_options: UniverseOptions,
):
    dataset = load_partial_data(
        client=client,
        time_bucket=Parameters.candle_time_bucket,
        pairs=[Parameters.trading_pair],
        execution_context=execution_context,
        universe_options=universe_options,
        required_history_period=Parameters.required_history_period,
    )

    universe = TradingStrategyUniverse.create_from_dataset(
        dataset,
        reserve_asset="USDT",
    )
    return universe


def create_indicators(
    timestamp: datetime.datetime | None,
    parameters: StrategyParameters,
    strategy_universe: TradingStrategyUniverse,
    execution_context: ExecutionContext
):
    indicators = IndicatorSet()
    return indicators