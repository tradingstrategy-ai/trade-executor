"""Strategy execution model.

Currently supported models

- Backtesting via :py:mod:`tradeexecutor.backtest.backtest_execution`

- Live execution against Uniswap v2 via :py:mod:`tradeexecutor.ethereum.uniswap_v2_execution`
"""
import abc
import datetime
import enum
from typing import List

from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.routing import RoutingModel, RoutingState


class AutoClosingOrderUnsupported(Exception):
    """Raised when trade execution does not support stop loss/take profit.

    Stop loss handling requires special support from the trade execution engine.
    See :py:meth:`ExecutionModel.is_stop_loss_supported` for more details.
    """


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
        """Set up the execution model ready to make trades.

        Read any on-chain, etc., data to get synced.

        - Read EVM nonce for the hot wallet from the chain
        """

    @abc.abstractmethod
    def get_routing_state_details(self) -> object:
        """Get needed details to establish a routing state.

        TODO: API Unfinished
        """

    @abc.abstractmethod
    def is_stop_loss_supported(self) -> bool:
        """Do we support stop-loss/take profit functionality with this execution model?

        - For backtesting we need to have data stream for candles used to calculate stop loss

        - For production execution, we need to have special oracle data streams
          for checking real-time stop loss
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