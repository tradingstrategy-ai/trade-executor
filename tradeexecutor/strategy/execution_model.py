"""Strategy execution model.

Currently supported models

- Backtesting via :py:mod:`tradeexecutor.backtest.backtest_execution`

- Live execution against Uniswap v2 via :py:mod:`tradeexecutor.ethereum.uniswap_v2_execution`
"""
import abc
import datetime
import enum
from typing import List, Dict, Tuple, TypedDict, Optional
from web3 import Web3
from hexbytes import HexBytes

from eth_defi.hotwallet import HotWallet
from tradeexecutor.ethereum.tx import TransactionBuilder
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.routing import RoutingModel, RoutingState
from tradeexecutor.state.blockhain_transaction import BlockchainTransaction


class AutoClosingOrderUnsupported(Exception):
    """Raised when trade execution does not support stop loss/take profit.

    Stop loss handling requires special support from the trade execution engine.
    See :py:meth:`ExecutionModel.is_stop_loss_supported` for more details.
    """


class RoutingStateDetails(TypedDict):
    """Detailts a trade router needs from the execeution mode to set its internal state.

    The content may differ if we are doing backtesting (no live execution objects needed),
    live trading or running some very legacy code.

    TODO: API unfinished. Needs to be cleaned up.

    TODO: Mark everything `NotRequired` if Python 3.11 migrated
    """

    tx_builder: TransactionBuilder

    #: TODO: Legacy - moved to Txbuilder
    web3: Web3

    #: TODO: Legacy - moved to Txbuilder
    wallet: HotWallet

    #: TODO: Legacy - moved to Txbuilder
    hot_wallet: HotWallet


class ExecutionModel(abc.ABC):
    """Define how trades are executed.

    See also :py:class:`tradeexecutor.strategy.mode.ExecutionMode`.
    
    Used directly by BacktestExecutionModel, and indirectly (through EthereumExecutionModel) by UniswapV2ExecutionModel and UniswapV3ExecutionModel
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
    def get_routing_state_details(self) -> RoutingStateDetails:
        """Get needed details to establish a routing state.
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

    @abc.abstractmethod
    def repair_unconfirmed_trades(self, state: State) -> List[TradeExecution]:
        """Repair unconfirmed trades.

        Repair trades that failed to properly broadcast or confirm due to
        blockchain node issues.

        :return:
            List of fixed trades
        """


class AssetManagementMode(enum.Enum):
    """Default execution options.

    What kind of trade instruction execution model the strategy does.

    Give options for command line parameters and such.

    TODO: Clean up unused options.
    """

    #: Does not make any trades, just captures and logs them
    dummy = "dummy"

    #: Server-side normal Ethereum private eky account
    hot_wallet = "hot_wallet"

    #: Trading using Enzyme Protocol pool, single oracle mode
    enzyme = "enzyme"

    #: Simulate execution using backtest data
    #:
    #: - Does not make any real trades
    #:
    #: - Does not connect to any network or blockchain
    #:
    backtest = "backtest"
