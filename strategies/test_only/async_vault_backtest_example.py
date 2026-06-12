"""Example strategy: two-stage async (ERC-7540) vault deposit and redeem.

Demonstrates how a strategy deposits into an asynchronous vault, waits for the
settlement delay (capital committed but unavailable), then redeems on a later
cycle. Used by ``tests/backtest/test_backtest_async_vault.py`` to exercise the
simulated two-stage settlement in :py:class:`~tradeexecutor.backtest.backtest_execution.BacktestExecution`.

The strategy:

1. On the first cycle, deposits half of the available cash into the single vault
   pair, keeping the other half as a cash buffer.
2. Holds the position. While the deposit is pending settlement ``is_any_open()``
   is already True and ``get_cash()`` is already debited, so it does not
   re-deposit the committed capital.
3. Once the position has been held for :py:data:`HOLD_PERIOD` and is fully
   settled (no ``vault_settlement_pending`` trade), redeems the whole position.
4. Does not re-deposit afterwards, because a closed position then exists.
"""

import datetime

from tradeexecutor.state.trade import TradeExecution, TradeStatus
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.strategy_module import StrategyParameters
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


trading_strategy_engine_version = "0.5"

#: How long to hold the vault position before redeeming.
HOLD_PERIOD = datetime.timedelta(days=4)


class Parameters:
    """Backtest parameters for the example async vault strategy."""

    cycle_duration = CycleDuration.cycle_1d
    initial_cash = 10_000


def create_indicators(
    parameters: StrategyParameters,
    indicators: IndicatorSet,
    strategy_universe: TradingStrategyUniverse,
    execution_context: ExecutionContext,
) -> None:
    """No indicators are needed for this example."""


def decide_trades(input: StrategyInput) -> list[TradeExecution]:
    """Deposit once into the async vault, hold, then redeem after HOLD_PERIOD.

    1. Deposit half of the cash into the vault on the first cycle (only once
       ever), keeping the other half as a buffer.
    2. Do not re-deposit while the deposit is pending: the position is already
       open and the cash ledger is already debited.
    3. Redeem the whole position once held long enough and fully settled.
    """

    state = input.state
    timestamp = input.timestamp
    position_manager = input.get_position_manager()
    pair = next(input.strategy_universe.iterate_pairs())

    trades: list[TradeExecution] = []

    # 1. Deposit half of the cash into the vault on the first cycle (only once ever).
    #    Keeping a cash buffer is realistic and means committed-but-pending capital
    #    never drives the recorded portfolio value to zero.
    if not position_manager.is_any_open() and len(state.portfolio.closed_positions) == 0:
        cash = state.portfolio.get_cash()
        if cash > 1.0:
            trades += position_manager.open_spot(pair, value=cash * 0.5)
        return trades

    # 3. Redeem the whole position once held long enough and fully settled.
    if position_manager.is_any_open():
        position = next(iter(state.portfolio.open_positions.values()))
        # 2. Skip while a deposit/redeem request is still pending settlement.
        has_pending_settlement = any(
            t.get_status() == TradeStatus.vault_settlement_pending
            for t in position.trades.values()
        )
        held_long_enough = (timestamp - position.opened_at) >= HOLD_PERIOD
        if held_long_enough and not has_pending_settlement:
            trades += position_manager.close_all()

    return trades
