"""Twenty-pair inspection fixture using the deployed mixed-vault universe.

The extra tokens exercise bounded account discovery. They have no contracts
and must never be traded; custody and tracked holdings use the real fixture.
"""

# Re-export the base settings and indicator hook for the strategy loader.
from strategies.test_only.multi_vault_arbitrum import Parameters as Parameters
from strategies.test_only.multi_vault_arbitrum import create_indicators as create_indicators
from strategies.test_only.multi_vault_arbitrum import create_trading_universe as create_small_universe
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.pandas_trader.trading_universe_input import CreateTradingUniverseInput
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.testing.multi_vault_start import expand_account_check_universe


trading_strategy_engine_version = "0.5"


def create_trading_universe(input: CreateTradingUniverseInput) -> TradingStrategyUniverse:
    """Expand the existing fixture without rewriting its strategy source."""
    return expand_account_check_universe(create_small_universe(input))


def decide_trades(input: StrategyInput) -> list[TradeExecution]:
    """Place no orders: this fixture exists only for account inspection.

    The input is required by the strategy interface and is unused.
    """
    return []
