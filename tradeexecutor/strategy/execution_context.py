"""Execution modes.

- Are we doing live trading or backtesting

- Any instrumentation like task duration tracing needed for the run
"""
import enum
from dataclasses import dataclass
from typing import Callable


class ExecutionMode(enum.Enum):
    """Different execution modes the strategy engine can hvae."""

    #: We are live trading with real assets
    real_trading = "real_trading"

    #: We are live trading with mock assets
    #: TODO: This mode is not yet supported
    paper_trading = "paper_trading"

    #: We are backtesting
    backtesting = "backtesting"

    #: We are loading and caching datasets before a backtesting session can begin.
    #: We call create_trading_universe() and assume :py:class:`tradingstrategy.client.Client`
    #: class is set to a such state it can display nice progress bar when loading
    #: data in a Jupyter notebook.
    data_preload = "data_preload"

    def is_live_trading(self) -> bool:
        """Are we trading with real money or paper money real time?"""
        return self.value in (self.real_trading, self.paper_trading)


@dataclass
class ExecutionContext:
    """Information about the strategy execution environment.

    This is passed to `create_trading_universe` and couple of other
    functions and they can determine and take action based
    the mode of strategy execution.
    """

    #: What is the current mode of the execution.
    mode: ExecutionMode

    #: Python context manager for timed tasks.
    #: Functions can use this context manager to add them to the tracing.
    #: Used for profiling the strategy code run-time performance.
    #: See :py:mod:`tradeexecutor.utils.timer`.
    timed_task_context_manager: Callable

    @property
    def live_trading(self) -> bool:
        """Are we doing live trading.

        :return:
            True if we doing live trading or paper trading.
             False if we are operating on backtesting data.
        """
        return self.mode in (ExecutionMode.real_trading, ExecutionMode.paper_trading)