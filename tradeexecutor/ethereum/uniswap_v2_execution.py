"""Execution model where trade happens directly on Uniswap v2 style exchange."""

import datetime
from decimal import Decimal
from typing import List
import logging

from web3 import Web3

from eth_defi.hotwallet import HotWallet
from tradeexecutor.ethereum.execution import broadcast_and_resolve
from tradeexecutor.ethereum.uniswap_v2_routing import UniswapV2SimpleRoutingModel, UniswapV2RoutingState
from tradeexecutor.state.freeze import freeze_position_on_failed_trade
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.execution_model import ExecutionModel


logger = logging.getLogger(__name__)


class UniswapV2ExecutionModel(ExecutionModel):
    """Run order execution on a single Uniswap v2 style exchanges."""

    def __init__(self,
                 web3: Web3,
                 hot_wallet: HotWallet,
                 min_balance_threshold=Decimal("0.5"),
                 confirmation_block_count=6,
                 confirmation_timeout=datetime.timedelta(minutes=5),
                 max_slippage: float = 0.01,
                 stop_on_execution_failure=True,
                 swap_gas_fee_limit=2_000_000):
        """
        :param web3:
            Web3 connection used for this instance

        :param hot_wallet:
            Hot wallet instance used for this execution

        :param min_balance_threshold:
            Abort execution if our hot wallet gas fee balance drops below this

        :param confirmation_block_count:
            How many blocks to wait for the receipt confirmations to mitigate unstable chain tip issues

        :param confirmation_timeout:
            How long we wait transactions to clear

        :param stop_on_execution_failure:
            Raise an exception if any of the trades fail top execute

        :param max_slippage:
            Max slippage tolerance per trade. 0.01 is 1%.
        """
        self.web3 = web3
        self.hot_wallet = hot_wallet
        self.stop_on_execution_failure = stop_on_execution_failure
        self.min_balance_threshold = min_balance_threshold
        self.confirmation_block_count = confirmation_block_count
        self.confirmation_timeout = confirmation_timeout
        self.swap_gas_fee_limit = swap_gas_fee_limit
        self.max_slippage = max_slippage

    @property
    def chain_id(self) -> int:
        """Which chain the live execution is connected to."""
        return self.web3.eth.chain_id

    def is_live_trading(self) -> bool:
        return True

    def is_stop_loss_supported(self) -> bool:
        # TODO: fix this when we want to use stop loss in real strategy
        return False

    def preflight_check(self):
        """Check that we can connect to the web3 node"""

        # Check JSON-RPC works
        assert self.web3.eth.block_number > 1

        # Check we have money for gas fees
        if self.min_balance_threshold > 0:
            balance = self.hot_wallet.get_native_currency_balance(self.web3)
            assert balance > self.min_balance_threshold, f"At least {self.min_balance_threshold} native currency need, our wallet {self.hot_wallet.address} has {balance:.8f}"

    def initialize(self):
        """Set up the wallet"""
        logger.info("Initialising Uniswap v2 execution model")
        self.hot_wallet.sync_nonce(self.web3)
        balance = self.hot_wallet.get_native_currency_balance(self.web3)
        logger.info("Our hot wallet is %s with nonce %d and balance %s", self.hot_wallet.address, self.hot_wallet.current_nonce, balance)

    def execute_trades(self,
                       ts: datetime.datetime,
                       state: State,
                       trades: List[TradeExecution],
                       routing_model: UniswapV2SimpleRoutingModel,
                       routing_state: UniswapV2RoutingState,
                       check_balances=False):
        """Execute the trades determined by the algo on a designed Uniswap v2 instance.

        :return: Tuple List of succeeded trades, List of failed trades
        """
        assert isinstance(ts, datetime.datetime)
        assert isinstance(routing_model, UniswapV2SimpleRoutingModel)
        assert isinstance(routing_state, UniswapV2RoutingState)

        state.start_trades(datetime.datetime.utcnow(), trades, max_slippage=self.max_slippage)

        routing_model.setup_trades(
            routing_state,
            trades,
            check_balances=check_balances)

        broadcast_and_resolve(self.web3, state, trades)

        # Clean up failed trades
        freeze_position_on_failed_trade(ts, state, trades)

    def get_routing_state_details(self) -> dict:
        return {
            "web3": self.web3,
            "hot_wallet": self.hot_wallet,
        }

