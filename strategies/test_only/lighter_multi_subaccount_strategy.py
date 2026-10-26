"""Strategy exercising Lighter master and sub-account cash management.

Used by the fixed-fork black-box test and the mainnet sub-account tutorial.
The Safe-owned Lighter master account is the treasury hub and each
sub-account is traded by an external process; this strategy only moves
custody capital between the Safe, the master and the sub-accounts, see
:py:func:`tradeexecutor.exchange_account.lighter_strategy.decide_lighter_cash_management`.

The sub-accounts are declared in :py:class:`Parameters`; ``init`` records
which Lighter account each id maps to in the state. The module is loaded
fresh by every Typer command, so the black-box test and the tutorial can
change the policy between ``start`` runs:

``LIGHTER_SUB_ACCOUNT_WEIGHTS``
    Comma-separated target weights in sub-account order, for example
    ``0.5,0.5``. Empty means no sub-account rebalancing.

``LIGHTER_SAFE_CASH_BUFFER_USD``
    USDC kept in the Safe. Defaults to 20.

``LIGHTER_TRANSFER_FEE_USD``
    Fee budgeted on top of each master to sub-account transfer. Defaults to 0,
    as Lighter charges nothing for transfers between accounts of one owner.
"""

import datetime
import os
from decimal import Decimal

from tradingstrategy.chain import ChainId

from tradeexecutor.exchange_account.lighter_strategy import (
    create_lighter_account_universe,
    decide_lighter_cash_management,
    parse_lighter_sub_account_weights,
)
from tradeexecutor.exchange_account.sub_account import SubAccount
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

trading_strategy_engine_version = "0.5"
trading_strategy_type = StrategyType.managed_positions
trading_strategy_cycle = CycleDuration.cycle_1d
trade_routing = TradeRouting.default
reserve_currency = ReserveCurrency.usdc


class Parameters:
    """Parameters for the Lighter sub-account cash-management strategy."""

    chain_id = ChainId.ethereum
    initial_cash = 100_000
    cycle_duration = CycleDuration.cycle_1d
    routing = TradeRouting.default
    required_history_period = datetime.timedelta(days=1)
    backtest_start = None
    backtest_end = None
    lighter_cash_management = True
    #: Sub-strategies run in their own Lighter sub-accounts.
    lighter_sub_accounts = [
        SubAccount(id="trend", name="Trend following"),
        SubAccount(id="arb", name="Funding-rate arbitrage"),
    ]
    lighter_safe_cash_buffer_usd = Decimal(os.environ.get("LIGHTER_SAFE_CASH_BUFFER_USD", "20"))
    lighter_free_collateral_buffer_usd = Decimal("0")
    lighter_min_transfer_usd = Decimal("1")
    lighter_sub_account_weights = parse_lighter_sub_account_weights(os.environ.get("LIGHTER_SUB_ACCOUNT_WEIGHTS", ""))
    lighter_rebalance_band_usd = Decimal("0")
    lighter_transfer_fee_usd = Decimal(os.environ.get("LIGHTER_TRANSFER_FEE_USD", "0"))


def create_trading_universe(input: CreateTradingUniverseInput) -> TradingStrategyUniverse:
    """Create one exchange-account pair per declared Lighter account.

    :param input:
        Strategy parameters declaring the sub-accounts and the state resolving them.
    :return:
        Static universe with the master account, the sub-accounts and the USDC reserve.
    """
    return create_lighter_account_universe(input, Parameters.lighter_sub_accounts)


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
    """Return this cycle's Lighter custody trades.

    :param input:
        Current strategy state, parameters and treasury-synchronised reserve.
    :return:
        Trades of at most one movement kind.
    """
    return decide_lighter_cash_management(input)
