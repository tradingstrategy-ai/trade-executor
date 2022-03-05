import datetime
from typing import List
import logging

from eth_hentai.hotwallet import HotWallet
from eth_hentai.uniswap_v2 import UniswapV2Deployment
from tradeexecutor.ethereum.execution import approve_tokens, prepare_swaps, confirm_approvals, broadcast, \
    wait_trades_to_complete, resolve_trades
from tradeexecutor.state.state import TradeExecution, State
from tradeexecutor.strategy.execution_model import ExecutionModel


logger = logging.getLogger(__name__)


class UniswapV2ExecutionModel(ExecutionModel):
    """Run order execution on a single Uniswap v2 style exchanges."""

    def __init__(self, uniswap: UniswapV2Deployment, hot_wallet: HotWallet, stop_on_execution_failure=True):
        """
        :param state:
        :param uniswap:
        :param hot_wallet:
        :param stop_on_execution_failure: Raise an exception if any of the trades fail top execute
        """
        self.web3 = uniswap.web3
        self.uniswap = uniswap
        self.hot_wallet = hot_wallet
        self.stop_on_execution_failure = stop_on_execution_failure

    def preflight_check(self):
        """Check that we can connect to the web3 node"""

        # Check JSON-RPC works
        assert self.web3.eth.block_number > 1

        # Check we have money for gas fees
        balance = self.wallet.get_native_currency_balance(self.web3)
        assert balance > self.min_balance_threshold, f"At least {self.min_balance_threshold} native currency need, {self.wallet.address} has {balance}"

        # Check Uniswap instance is valid
        try:
            self.uniswap.factory.functions.pairCode().call()
        except Exception as e:
            raise AssertionError(f"Uniswap does not function at factory address {self.uniswap.factory.address}") from e

    def initialize(self):
        """Set up the wallet"""
        logger.info("Initialising Uniswap v2 execution model")
        self.hot_wallet.sync_nonce(self.web3)
        logger.info("Our hot wallet is %s with nonce %d", self.hot_wallet.address, self.hot_wallet.current_nonce)

    def execute_trades(self, ts: datetime.datetime, state: State, trades: List[TradeExecution]):
        """Execute the trades determined by the algo on a designed Uniswap v2 instance."""
        assert isinstance(ts, datetime.datetime)

        # 2. Capital allocation
        # Approvals
        approvals = approve_tokens(
            self.web3,
            self.uniswap,
            self.hot_wallet,
            trades
        )

        # 2: prepare
        # Prepare transactions
        prepare_swaps(
            self.web3,
            self.hot_wallet,
            self.uniswap,
            ts,
            self.state,
            trades,
            underflow_check=False,
        )

        #: 3 broadcast

        # Handle approvals separately for now
        confirm_approvals(self.web3, approvals)

        broadcasted = broadcast(self.web3, ts, trades)
        #assert trade.get_status() == TradeStatus.broadcasted

        # Resolve
        receipts = wait_trades_to_complete(self.web3, trades)
        resolve_trades(
            self.web3,
            self.uniswap,
            ts,
            self.state,
            broadcasted,
            receipts)
