"""Strategy running a fixed script of Lighter custody transfers.

Used by the Lighter transfer-accounting tutorial and its simulated test. Each
``decide_trades()`` cycle emits the next step of
:py:func:`tradeexecutor.testing.lighter_scripted_transfers.build_lighter_transfer_schedule`:
deposit the Safe cash to the master, fund both sub-accounts, move capital
from one sub-account to the other, move it back to the master, then withdraw
everything. Every movement is its own trade moving value between two
positions.

Transfers sent from a sub-account are signed with that sub-account's key, so
``start`` needs ``LIGHTER_SUB_ACCOUNT_OPERATOR_RECORD_FILES`` as well as the
master operator record.

The module is loaded fresh by every Typer command and reads:

``LIGHTER_ACCOUNT_INDEXES``
    Comma-separated master account index followed by sub-account indices,
    from the public deployment report. Required.

``LIGHTER_SCRIPTED_UNIT_USDC``
    Base amount of the fixed movements. Defaults to 1, Lighter's minimum.
"""

import datetime
import os
from decimal import Decimal

from tradingstrategy.chain import ChainId

from tradeexecutor.exchange_account.lighter import parse_lighter_account_indices
from tradeexecutor.exchange_account.lighter_strategy import create_lighter_account_universe
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.pandas_trader.trading_universe_input import CreateTradingUniverseInput
from tradeexecutor.strategy.reserve_currency import ReserveCurrency
from tradeexecutor.strategy.strategy_module import StrategyParameters
from tradeexecutor.strategy.strategy_type import StrategyType
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.testing.lighter_scripted_transfers import build_lighter_transfer_schedule, decide_lighter_scripted_transfers

trading_strategy_engine_version = "0.5"
trading_strategy_type = StrategyType.managed_positions
trading_strategy_cycle = CycleDuration.cycle_1d
trade_routing = TradeRouting.default
reserve_currency = ReserveCurrency.usdc


class Parameters:
    """Parameters for the scripted Lighter transfer strategy."""

    chain_id = ChainId.ethereum
    initial_cash = 100_000
    cycle_duration = CycleDuration.cycle_1d
    routing = TradeRouting.default
    required_history_period = datetime.timedelta(days=1)
    backtest_start = None
    backtest_end = None
    #: Enables Lighter custody routing in ``start``.
    lighter_cash_management = True
    #: Lighter's transfer, withdrawal and deposit minimum.
    lighter_min_transfer_usd = Decimal("1")
    lighter_scripted_unit_usd = Decimal(os.environ.get("LIGHTER_SCRIPTED_UNIT_USDC", "1"))


def create_trading_universe(input: CreateTradingUniverseInput) -> TradingStrategyUniverse:
    """Create one exchange-account pair per Lighter account.

    :param input:
        Unused strategy-universe construction input.
    :return:
        Static universe with the master account, the sub-accounts and the USDC reserve.
    """
    del input
    return create_lighter_account_universe(parse_lighter_account_indices(os.environ.get("LIGHTER_ACCOUNT_INDEXES", "")))


def create_indicators(
    parameters: StrategyParameters,
    indicators: IndicatorSet,
    strategy_universe: TradingStrategyUniverse,
    execution_context: ExecutionContext,
) -> None:
    """Declare no market indicators for this custody-only strategy.

    :param parameters:
        Unused strategy parameters.
    :param indicators:
        Empty indicator collection retained by the engine interface.
    :param strategy_universe:
        Static Lighter strategy universe.
    :param execution_context:
        Current execution context.
    """
    del parameters, indicators, strategy_universe, execution_context


def decide_trades(input: StrategyInput) -> list[TradeExecution]:
    """Return the next scripted step's custody trades.

    :param input:
        Current strategy state, parameters and treasury-synchronised reserve.
    :return:
        Trades of one schedule step, or none.
    """
    parameters = input.parameters
    schedule = build_lighter_transfer_schedule(Decimal(str(parameters.lighter_scripted_unit_usd)))
    return decide_lighter_scripted_transfers(input, schedule, Decimal(str(parameters.lighter_min_transfer_usd)))
