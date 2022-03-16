"""Execution mode modes."""
import enum


class ExecutionMode(enum.Enum):
    """How trade execution is handled."""

    #: Real thing
    live_trade = "live_trade"

    #: Calculate positions with live ticks, but do not execute any trades
    paper_trade = "paper_trade"

    #: Calculate positions with past ticks and simulate past trades
    backtest = "backtest"