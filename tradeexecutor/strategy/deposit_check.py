"""Compatibility imports; use :mod:`tradeexecutor.strategy.vault_closure` for new code.

All historical and live deposit interpretations are maintained in that module.
"""
from tradeexecutor.strategy.vault_closure import (
    BACKTEST_ALWAYS_OPEN_DEPOSIT_FEATURES,
    BACKTEST_ALWAYS_OPEN_DEPOSIT_PROTOCOLS,
    is_backtesting_deposit_open,
    check_backtesting_deposit,
    get_live_max_deposit,
    check_live_deposit,
)
