"""A testing executoin model without actual execution."""
import datetime
from typing import List

from web3 import Web3

from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.execution_model import ExecutionModel
from tradeexecutor.strategy.routing import RoutingModel, RoutingState


class DummyExecutionModel(ExecutionModel):
    """Trade executor that does not connect to anything.

    Used in testing.
    """

    def __init__(self, web3: Web3):
        self.web3 = web3

    def get_safe_latest_block(self):
        return None

    def get_balance_address(self):
        return None

    def preflight_check(self):
        """Check that we can start the trade executor

        :raise: AssertionError if something is a miss
        """

    def initialize(self):
        """Set up the execution model ready to make trades.

        Read any on-chain, etc., data to get synced.

        - Read EVM nonce for the hot wallet from the chain
        """

    def get_routing_state_details(self) -> object:
        """Get needed details to establish a routing state.

        """
        return {
            "web3": self.web3,
        }

    def is_stop_loss_supported(self) -> bool:
        """Do we support stop-loss/take profit functionality with this execution model?

        - For backtesting we need to have data stream for candles used to calculate stop loss

        - For production execution, we need to have special oracle data streams
          for checking real-time stop loss
        """

    def execute_trades(
        self,
        ts: datetime.datetime,
        state: State,
        trades: List[TradeExecution],
        routing_model: RoutingModel,
        routing_state: RoutingState,
        max_slippage=0.005,
        check_balances=False,
        triggered=False,
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

    def repair_unconfirmed_trades(self, state: State) -> List[TradeExecution]:
        """Repair unconfirmed trades.

        Repair trades that failed to properly broadcast or confirm due to
        blockchain node issues.

        :return:
            List of fixed trades
        """

    def create_default_routing_model(self, universe):
        """Return None - no routing needed for dummy execution.

        Used for exchange account strategies that don't do any on-chain trading.
        """
        return None