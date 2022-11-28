"""Execution modes.

- Are we doing live trading or backtesting

- Any instrumentation like task duration tracing needed for the run
"""
import enum
from dataclasses import dataclass
from typing import Callable

from tradeexecutor.utils.timer import timed_task


class ExecutionMode(enum.Enum):
    """Different execution modes the strategy engine can hvae."""

    #: We are live trading with real assets
    real_trading = "real_trading"

    #: We are live trading with mock assets
    #: TODO: This mode is not yet supported
    paper_trading = "paper_trading"

    #: We are backtesting
    #: When backtesting mode is selected, we can skip most of the statistical calculations that would otherwise be calculated during live-trade. 
    #: This offers great performance benefits for backtesting. 
    backtesting = "backtesting"

    #: We are loading and caching datasets before a backtesting session can begin.
    #: We call create_trading_universe() and assume :py:class:`tradingstrategy.client.Client`
    #: class is set to a such state it can display nice progress bar when loading
    #: data in a Jupyter notebook.
    data_preload = "data_preload"

    #: We are operating on real datasets like :py:data:`real_trading`
    #: but we do not want to purge caches.
    #:
    #: This mode is specially used to test some live trading features.
    #:
    unit_testing_trading = "unit_testing_trading"

    #: Simulated trading: Blockchain we are connected is not real.
    #:
    #: We are trading against a simulated step-by-step blockchain
    #: like EthereumTester. This allows us to control
    #: block production, but otherwise behave as
    #: live trading.
    #:
    #: In this mode, we are also not using any dataset loading features,
    #: but the trading universe and price feeds are typed in the test code.
    #:
    simulated_trading = "simulated_trading"

    #: Prefilight checks
    #:
    #: In this execution mode, we are invoked from the command line
    #: to check that all of our files and connections are intact.
    preflight_check = "preflight_check"

    def is_live_trading(self) -> bool:
        """Are we trading  real time?"""
        return self in (self.real_trading, self.paper_trading, self.unit_testing_trading, self.simulated_trading)

    def is_fresh_data_always_needed(self):
        """Should we purge caches for each trade cycle.

        This will force the redownload of data on each cycle.
        """
        return self in (self.real_trading, self.paper_trading, self.simulated_trading)


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
    #:
    #: Functions can use this context manager to add them to the tracing.
    #: Used for profiling the strategy code run-time performance.
    #:
    #: Set default to :py:func:`tradeexecutor.utils.timer.timed_task`.
    #: which logs task duration using logging.INFO level.
    #:
    timed_task_context_manager: Callable = timed_task

    @property
    def live_trading(self) -> bool:
        """Are we doing live trading.

        :return:
            True if we doing live trading or paper trading.
             False if we are operating on backtesting data.
        """
        return self.mode in (ExecutionMode.real_trading, ExecutionMode.paper_trading, ExecutionMode.simulated_trading)