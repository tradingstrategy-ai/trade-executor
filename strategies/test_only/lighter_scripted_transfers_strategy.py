"""Example strategy moving funds in, out and between Lighter sub-accounts.

A Lagoon vault's Safe owns a Lighter master account and sub-accounts. The
strategy declares its sub-accounts in :py:class:`Parameters` as
:py:class:`~tradeexecutor.exchange_account.sub_account.SubAccount` values:
the sub-strategies it runs, each under a stable id. Deployment creates one
Lighter account per declared sub-account and ``init`` records which account
each id maps to in the state. Every account becomes its own trading position
when funds first move into it, and every movement is one trade:

- ``position_manager.deposit_to_lighter()`` moves Safe cash to the master.
- ``position_manager.transfer_lighter_funds(source, destination, amount)``
  moves USDC between two accounts, signed with the source account's key.
- ``position_manager.withdraw_from_lighter(account)`` moves an account's
  free collateral back to the Safe.

``decide_trades()`` runs a fixed script, one step per cycle, so the
transfer-accounting tutorial and its black-box test can check the resulting
state. Transfers sent from a sub-account need that sub-account's key, given
to ``start`` with ``LIGHTER_SUB_ACCOUNT_OPERATOR_RECORD_FILES``.

No Lighter account index appears in the strategy: the same module deploys
and runs on a fork or on mainnet.
"""

import datetime

from tradingstrategy.chain import ChainId

from tradeexecutor.exchange_account.lighter_strategy import create_lighter_account_universe
from tradeexecutor.exchange_account.sub_account import SubAccount
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.pandas_trader.trading_universe_input import CreateTradingUniverseInput
from tradeexecutor.strategy.strategy_module import StrategyParameters
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.testing.lighter_scripted_transfers import SCRIPTED_TRANSFER_STEPS, SCRIPTED_TRANSFER_UNIT_USDC, get_next_scripted_step, tag_scripted_trades

trading_strategy_engine_version = "0.5"

class Parameters:
    """Parameters of the scripted Lighter transfer strategy."""

    chain_id = ChainId.ethereum
    initial_cash = 100_000
    cycle_duration = CycleDuration.cycle_1d
    routing = TradeRouting.default
    required_history_period = datetime.timedelta(days=1)
    backtest_start = None
    backtest_end = None
    #: Enables Lighter custody routing in ``start``.
    lighter_cash_management = True
    #: Sub-strategies run in their own Lighter sub-accounts; the id is the stable key the strategy moves funds with.
    lighter_sub_accounts = [
        SubAccount(id="trend", name="Trend following"),
        SubAccount(id="arb", name="Funding-rate arbitrage"),
    ]


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


def decide_trades(input: StrategyInput) -> list[TradeExecution]:
    """Run the next step of the transfer script.

    :param input:
        Current strategy state, parameters and treasury-synchronised reserve.
    :return:
        The step's custody trades, or none while a transfer is in flight or
        the script is complete.
    """
    if input.execution_context.mode.is_backtesting():
        return []
    position_manager = input.get_position_manager()
    if position_manager.is_lighter_transfer_pending():
        return []
    step = get_next_scripted_step(input.state, SCRIPTED_TRANSFER_STEPS)
    if step is None:
        return []

    unit = SCRIPTED_TRANSFER_UNIT_USDC
    trend, arb = (sub_account.id for sub_account in input.parameters.lighter_sub_accounts)
    match step:
        case 1:
            # Move in: all idle Safe cash goes to the master account.
            trades = [position_manager.deposit_to_lighter()]
        case 2:
            # Fund both sub-accounts from the master.
            trades = [
                position_manager.transfer_lighter_funds("master", trend, 2 * unit),
                position_manager.transfer_lighter_funds("master", arb, 2 * unit),
            ]
        case 3:
            # Move between sub-accounts, signed with the sending sub-account's key.
            trades = [position_manager.transfer_lighter_funds(trend, arb, unit)]
        case 4:
            # Move from a sub-account back to the master.
            trades = [position_manager.transfer_lighter_funds(arb, "master", unit)]
        case _:
            # Move out: every account's free collateral returns to the Safe.
            trades = [position_manager.withdraw_from_lighter(account) for account in (trend, arb, "master")]
    return tag_scripted_trades(trades, step)
