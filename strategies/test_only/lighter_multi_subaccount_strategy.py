"""Strategy exercising Lighter master and sub-account cash management.

Used by the fixed-fork black-box test and the mainnet sub-account tutorial.
The Safe-owned Lighter master account is the treasury hub and each
sub-account is traded by an external process; this strategy only moves
custody capital between the Safe, the master and the sub-accounts with the
production policy,
:py:func:`tradeexecutor.exchange_account.lighter_strategy.decide_lighter_cash_management`.
The sub-accounts, their target weights and the policy buffers are declared
in :py:class:`Parameters`; the test and the tutorial drive the policy by
changing balances, as traders would, not by changing the strategy.
"""

import datetime
from decimal import Decimal

from tradingstrategy.chain import ChainId

from tradeexecutor.exchange_account.lighter_strategy import create_lighter_account_universe, decide_lighter_cash_management
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

trading_strategy_engine_version = "0.5"

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
    #: Sub-strategies run in their own Lighter sub-accounts, each with its target share of the capital.
    lighter_sub_accounts = [
        SubAccount(id="trend", name="Trend following", weight=0.5),
        SubAccount(id="arb", name="Funding-rate arbitrage", weight=0.5),
    ]
    #: Nothing stays in the Safe: all capital goes to Lighter and the tutorial runs at Lighter's minimums.
    lighter_safe_cash_buffer_usd = Decimal(0)
    lighter_free_collateral_buffer_usd = Decimal(0)


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
    """Return this cycle's Lighter custody trades.

    :param input:
        Current strategy state, parameters and treasury-synchronised reserve.
    :return:
        Trades of at most one movement kind.
    """
    return decide_lighter_cash_management(input)
