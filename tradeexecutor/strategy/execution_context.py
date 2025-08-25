"""Execution modes.

- Are we doing live trading or backtesting

- Any instrumentation like task duration tracing needed for the run
"""
import enum
from dataclasses import dataclass
from typing import Callable
from packaging import version

from tradeexecutor.strategy.engine_version import TradingStrategyEngineVersion
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.utils.timer import timed_task


class ExecutionMode(enum.Enum):
    """Different execution modes the strategy engine can handle.

    Depending on how we are using the engine, we might enable and disable
    additional checks and features.

    - In unit testing execution mode we can skip
      all kind of delays when we need to wait a blockchain chain tip to stabilise

    - In backtesting execution mode we skip calculation of statistics
      between strategy decision cycles, as these statistics are discarted
      and calculations slows us down

    """

    #: We are live trading with real assets
    real_trading = "real_trading"

    #: We are live trading with mock assets
    #:
    #: TODO: This mode is not yet supported
    paper_trading = "paper_trading"

    #: We are backtesting
    #: When backtesting mode is selected, we can skip most of the statistical calculations that would otherwise be calculated during live-trade.
    #: This offers great performance benefits for backtesting.
    backtesting = "backtesting"

    #: We are doing data research.
    #:
    #: There is not going to be any trading,
    #: we are only interested in datsets.
    data_research = "data_research"

    #: We are loading and caching datasets before a backtesting session can begin.
    #: We call create_trading_universe() and assume :py:class:`tradingstrategy.client.Client`
    #: class is set to a such state it can display nice progress bar when loading
    #: data in a Jupyter notebook.
    data_preload = "data_preload"

    #: Internal unit testing of modules
    #:
    #: This specifically refers to unit testing that uses backtesting data.
    #: See :py:attr:`unit_testing_trading` as well.
    #:
    unit_testing = "unit_Testing"

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

    #: One off diagnostic and scripts
    #:
    #: Used in the interactive :ref:`console.
    #: and debugging scripts.
    one_off = "one_off"

    #: Web server execution context
    #:
    #: Used when rendering charts and serving data
    #: via the web API endpoints in live trading.
    web = "web"

    def is_live_trading(self) -> bool:
        """Are we trading real time?

        - For live trading, we are operating on real-time price feeds based on `required_history_period` distance,
          instead of explicit start and end dates of backtesting

        - Preflight check is considered live trading, because strategy modules
          are not in backtesting when doing preflight checks
        """
        return self in (self.real_trading, self.paper_trading, self.unit_testing_trading, self.simulated_trading, self.preflight_check, self.one_off, self.web)

    def is_backtesting(self) -> bool:
        """The strategy is running for backtesting.

        """
        return self in (self.backtesting, self.unit_testing,)

    def is_fresh_data_always_needed(self):
        """Should we purge caches for each trade cycle.

        This will force the redownload of data on each cycle.
        """
        return self in (self.real_trading, self.paper_trading, self.simulated_trading)

    def is_unit_testing(self) -> bool:
        """Are we executing unit tests."""
        return self in (self.unit_testing_trading, self.simulated_trading, self.unit_testing,)


@dataclass(slots=True)
class ExecutionContext:
    """Information about the strategy execution environment.

    Execution context helps the strategy to decide different code paths,
    whether we are doing backtesting, live trading and what is the
    target Trading Strategy engine version.

    - This is passed to `create_trading_universe` and other
      functions. they can determine and take action based
      the mode of strategy execution. For example,
      we may load pair and candle data differently in live trading.
      See :py:attr:`mode`.

    - Some function signatures have been made better, but we still
      need to maintain backwards compatibility, see :py:attr:`engine_version`.

    - We also pass the Python profiler hook :py:attr:`timed_task_context_manager`.

    Example how to create for backtests:

    .. code-block:: python

        from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode

        execution_context = ExecutionContext(
            mode=ExecutionMode.backtesting,
        )

    See also

    - :py:data:`unit_test_execution_context`

    - :py:data:`notebook_execution_context`
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

    #: What TS engine the strategy is using.
    #:
    #: `None` means 0.1.
    #:
    #: See :py:mod:`tradeexecutor.strategy.engine_version`.
    #:
    engine_version: TradingStrategyEngineVersion = None

    #: Strategy parameters
    #:
    #: For backtesting/grid search cycle.
    #:
    #: v0.4 only
    #:
    parameters: StrategyParameters | None = None

    #: Is this backtest run part of a grid search group
    #:
    grid_search: bool = False

    #: Is this backtest run part of a optimiser
    #:
    optimiser: bool = False

    #: Are we running inside Jupyter notebook.
    #:
    #: - We might have HTML widgets available like HTML progress bar
    #: - We have interactive prompts available
    #:
    jupyter: bool = False

    #: Render tqdm progress bars when running
    #:
    #: Disabled in child worker process context.
    #:
    progress_bars: bool = True

    #: Force collection of visualisation and draw data during grid search and optimisation.
    #:
    #: Slows down the process a bit.
    #: Note that changing this parameter means that you need to manually deleted previous cached results,
    #: as this is stored as the part of the backtest state.
    #:
    #: See `perform_optimisation(draw_visualisation)` argument.
    #:
    force_visualisation: bool = False

    def __repr__(self):
        version_str = f"v{self.engine_version}" if self.engine_version else "unspecified engine version"
        return f"<ExecutionContext {self.mode.name}, {version_str}>"

    def is_version_greater_or_equal_than(self, major: int, minor: int, patch: int) -> bool:
        """Check that we are runing engine as the minimum required version."""
        running_version = self.engine_version or "0.1"
        required_version = f"{major}.{minor}.{patch}"
        return version.parse(running_version) >= version.parse(required_version)

    def has_visualisation(self) -> bool:
        """Should backtest spend time to draw custom visualisations.

        - By default, disabled for the grid search/optimiser for the speed
        """
        return self.force_visualisation or not(self.grid_search or self.optimiser)

    @property
    def live_trading(self) -> bool:
        """Are we doing live trading.

        See :py:method:`ExecutionMode.is_live_trading`.
        """
        return self.mode.is_live_trading()


#: Shorthand for unit testing
unit_test_execution_context = ExecutionContext(ExecutionMode.unit_testing)

unit_test_trading_execution_context = ExecutionContext(ExecutionMode.unit_testing_trading)

#: Shorthand for notebooks
notebook_execution_context = ExecutionContext(ExecutionMode.backtesting, jupyter=True)

#: Shorthand for doing a grid search within Jupyter
grid_search_execution_context = ExecutionContext(ExecutionMode.backtesting, grid_search=True)

#: Shorthand for Python scripts
python_script_execution_context = ExecutionContext(ExecutionMode.backtesting)

#: Standalone backtest (not within a notebook)
standalone_backtest_execution_context = ExecutionContext(ExecutionMode.backtesting)

#: Special context that we run the code within a web server HTTP API handler
web_server_execution_context = ExecutionContext(ExecutionMode.web)

#: Shorthand when running a indicator parameter optimizer using scikit-optimizer.
#:
#: We are inside a child worker process spawned by scikit,
scikit_optimizer_context = ExecutionContext(
    ExecutionMode.backtesting,
    jupyter=False,
    progress_bars=False,
    optimiser=True,
)

# trade-execution console commands
console_command_execution_context = ExecutionContext(ExecutionMode.real_trading)

# trade-execution console commands
one_off_execution_context = ExecutionContext(ExecutionMode.one_off)

# trade-execution console commands
preflight_execution_context = ExecutionContext(ExecutionMode.preflight_check)


#: Shorthand for unit testing
ExecutionContext.unit_test = unit_test_execution_context
