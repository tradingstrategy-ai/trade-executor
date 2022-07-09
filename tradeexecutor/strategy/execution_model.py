"""Strategy execution model.

Currently supported models

- Backtesting via :py:mod:`tradeexecutor.backtest.backtest_execution`

- Live execution against Uniswap v2 via :py:mod:`tradeexecutor.ethereum.uniswap_v2_execution`
"""
import abc
import datetime
import enum
from dataclasses import dataclass
from typing import List, Callable

from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.mode import ExecutionMode
from tradeexecutor.strategy.routing import RoutingModel, RoutingState


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



class ExecutionModel(abc.ABC):
    """Define how trades are executed.

    See also :py:class:`tradeexecutor.strategy.mode.ExecutionMode`.
    """

    @abc.abstractmethod
    def preflight_check(self):
        """Check that we can start the trade executor

        :raise: AssertionError if something is a miss
        """

    @abc.abstractmethod
    def initialize(self):
        """Read any on-chain, etc., data to get synced.
        """

    @abc.abstractmethod
    def get_routing_state_details(self) -> object:
        """Get needed details to establish a routing state.

        TODO: API Unfinished
        """

    @abc.abstractmethod
    def execute_trades(self,
                       ts: datetime.datetime,
                       state: State,
                       trades: List[TradeExecution],
                       routing_model: RoutingModel,
                       routing_state: RoutingState,
                       max_slippage=0.005,
                       check_balances=False,
                       ):
        """Execute the trades determined by the algo on a designed Uniswap v2 instance.

        :param ts:
            Timestamp of the trade cycle.

        :param universe:
            Current trading universe for this cycle.

        :param state:
            State of the trade executor.

        :param trades:
            List of trades decided by the strategy.
            Will be executed and modified in place.

        :param routing_model:
            Routing model how to execute the trades

        :param routing_state:
            State of already made on-chain transactions and such on this cycle

        :param max_slippage:
            Max slippage % allowed on trades before trade execution fails.

        :param check_balances:
            Check that on-chain accounts have enough balance before creating transaction objects.
            Useful during unit tests to spot issues in trade routing.
        """


class TradeExecutionType(enum.Enum):
    """Default execution options.

    What kind of trade instruction execution model the strategy does.

    Give options for command line parameters and such.

    TODO: Clean up unused options.
    """

    #: Does not make any trades, just captures and logs them
    dummy = "dummy"

    #: Server-side normal Ethereum private eky account
    uniswap_v2_hot_wallet = "uniswap_v2_hot_wallet"

    #: Trading using Enzyme Protocol pool, single oracle mode
    single_oracle_pooled = "single_oracle_pooled"

    #: Trading using oracle network, oracles form a consensus using a judge smart contract
    multi_oracle_judged = "multi_oracle_judged"

    #: Simulate execution using backtest data
    backtest = "backtest"