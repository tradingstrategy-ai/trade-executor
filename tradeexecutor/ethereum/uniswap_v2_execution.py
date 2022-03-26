import datetime
from decimal import Decimal
from typing import List, Tuple
import logging

from eth_hentai.hotwallet import HotWallet
from eth_hentai.uniswap_v2.deployment import UniswapV2Deployment
from tradeexecutor.ethereum.execution import approve_tokens, prepare_swaps, confirm_approvals, broadcast, \
    wait_trades_to_complete, resolve_trades
from tradeexecutor.state.freeze import freeze_position_on_failed_trade
from tradeexecutor.state.state import TradeExecution, State
from tradeexecutor.strategy.execution_model import ExecutionModel


logger = logging.getLogger(__name__)


class UniswapV2ExecutionModel(ExecutionModel):
    """Run order execution on a single Uniswap v2 style exchanges."""

    def __init__(self,
                 uniswap: UniswapV2Deployment,
                 hot_wallet: HotWallet,
                 min_balance_threshold=Decimal("0.5"),
                 confirmation_block_count=6,
                 confirmation_timeout=datetime.timedelta(minutes=5),
                 stop_on_execution_failure=True):
        """
        :param state:
        :param uniswap:
        :param hot_wallet:
        :param min_balance_threshold: Abort execution if our hot wallet gas fee balance drops below this
        :param confirmation_block_count: How many blocks to wait for the receipt confirmations to mitigate unstable chain tip issues
        :param confirmation_timeout: How long we wait transactions to clear
        :param stop_on_execution_failure: Raise an exception if any of the trades fail top execute
        """
        self.web3 = uniswap.web3
        self.uniswap = uniswap
        self.hot_wallet = hot_wallet
        self.stop_on_execution_failure = stop_on_execution_failure
        self.min_balance_threshold = min_balance_threshold
        self.confirmation_block_count = confirmation_block_count
        self.confirmation_timeout = confirmation_timeout

    @property
    def chain_id(self) -> int:
        """Which chain the live execution is connected to."""
        return self.web3.eth.chain_id

    def preflight_check(self):
        """Check that we can connect to the web3 node"""

        # Check JSON-RPC works
        assert self.web3.eth.block_number > 1

        # Check we have money for gas fees
        balance = self.hot_wallet.get_native_currency_balance(self.web3)
        assert balance > self.min_balance_threshold, f"At least {self.min_balance_threshold} native currency need, our wallet {self.hot_wallet.address} has {balance:.8f}"

        # Check Uniswap v2 instance is valid.
        # Different factories (Sushi, Pancake) share few common public accessors we can call here.
        try:
            self.uniswap.factory.functions.allPairsLength().call()
        except Exception as e:
            raise AssertionError(f"Uniswap does not function at chain {self.chain_id}, factory address {self.uniswap.factory.address}") from e

    def initialize(self):
        """Set up the wallet"""
        logger.info("Initialising Uniswap v2 execution model")
        self.hot_wallet.sync_nonce(self.web3)
        balance = self.hot_wallet.get_native_currency_balance(self.web3)
        logger.info("Our hot wallet is %s with nonce %d and balance %s", self.hot_wallet.address, self.hot_wallet.current_nonce, balance)

    def execute_trades(self, ts: datetime.datetime, state: State, trades: List[TradeExecution]) -> Tuple[List[TradeExecution], List[TradeExecution]]:
        """Execute the trades determined by the algo on a designed Uniswap v2 instance.

        :return: Tuple List of succeeded trades, List of failed trades
        """
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
            state,
            trades,
            underflow_check=False,
        )

        #: 3 broadcast

        # Handle approvals separately for now.
        # We do not need to wait these to confirm.
        confirm_approvals(self.web3, approvals, confirmation_block_count=0)

        broadcasted = broadcast(self.web3, ts, trades)
        #assert trade.get_status() == TradeStatus.broadcasted

        # Resolve
        receipts = wait_trades_to_complete(
            self.web3,
            trades,
            confirmation_block_count=self.confirmation_block_count,
            max_timeout=self.confirmation_timeout)

        resolve_trades(
            self.web3,
            self.uniswap,
            ts,
            state,
            broadcasted,
            receipts,
            stop_on_execution_failure=False)

        # Clean up failed trades
        return freeze_position_on_failed_trade(ts, state, trades)
