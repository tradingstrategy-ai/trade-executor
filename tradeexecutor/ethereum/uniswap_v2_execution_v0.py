"""Execution model where trade happens directly on Uniswap v2 style exchange.

TODO: Prototype code path. Only preserved for having unit test suite green.
It has slight API incompatibilities in the later versions.
"""

import datetime
from decimal import Decimal
from typing import List, Tuple, Optional
import logging

from eth_defi.gas import estimate_gas_fees
from eth_defi.hotwallet import HotWallet
from eth_defi.uniswap_v2.deployment import UniswapV2Deployment
from tradeexecutor.ethereum.execution import broadcast_and_resolve
from tradeexecutor.ethereum.tx import TransactionBuilder
from tradeexecutor.ethereum.uniswap_v2_routing import UniswapV2RoutingState, UniswapV2SimpleRoutingModel
from tradeexecutor.state.freeze import freeze_position_on_failed_trade
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.execution_model import ExecutionModel
from tradeexecutor.strategy.routing import RoutingModel, RoutingState
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse

logger = logging.getLogger(__name__)


class UniswapV2RoutingInstructions:
    """Helper class to router Uniswap trades.

    - Define allowed routes to use

    - Define routing for three way trades
    """

    def __init__(self, routing_table: dict):
        """

        :param routing_table: Exchange factory address -> router address data
        """


class UniswapV2ExecutionModelVersion0(ExecutionModel):
    """Run order execution on a single Uniswap v2 style exchanges.

    TODO: This model was used in the first prototype and later discarded.
    """

    def __init__(self,
                 uniswap: UniswapV2Deployment,
                 hot_wallet: HotWallet,
                 min_balance_threshold=Decimal("0.5"),
                 confirmation_block_count=6,
                 confirmation_timeout=datetime.timedelta(minutes=5),
                 max_slippage=0.01,
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
        self.max_slippage = max_slippage

    @property
    def chain_id(self) -> int:
        """Which chain the live execution is connected to."""
        return self.web3.eth.chain_id

    def is_live_trading(self):
        return True

    def is_stop_loss_supported(self) -> bool:
        return False

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

    def execute_trades(self,
                       ts: datetime.datetime,
                       state: State,
                       trades: List[TradeExecution],
                       routing_model: Optional[RoutingModel],
                       routing_state: Optional[RoutingState],
                       check_balances=False) -> Tuple[List[TradeExecution], List[TradeExecution]]:
        """Execute the trades determined by the algo on a designed Uniswap v2 instance.

        :param routing_model:
            Ignored.

        :return: Tuple List of succeeded trades, List of failed trades
        """
        assert isinstance(ts, datetime.datetime)
        assert isinstance(state, State), f"Received {state.__class__}"
        assert isinstance(routing_model, UniswapV2SimpleRoutingModel), f"Received {routing_state.__class__}"
        assert isinstance(routing_state, UniswapV2RoutingState), f"Received {routing_state.__class__}"

        fees = estimate_gas_fees(self.web3)

        tx_builder = TransactionBuilder(
            self.web3,
            self.hot_wallet,
            fees,
        )

        reserve_asset, rate = state.portfolio.get_default_reserve_currency()

        # We know only about one exchange
        routing_model = UniswapV2SimpleRoutingModel(
            factory_router_map={
                self.uniswap.factory.address: (self.uniswap.router.address, self.uniswap.init_code_hash),
            },
            allowed_intermediary_pairs={},
            reserve_token_address=reserve_asset.address,
        )

        state.start_trades(datetime.datetime.utcnow(), trades)

        routing_model.setup_trades(routing_state, trades, check_balances=check_balances)
        broadcast_and_resolve(self.web3, state, trades, stop_on_execution_failure=self.stop_on_execution_failure)

        # Clean up failed trades
        freeze_position_on_failed_trade(ts, state, trades)

        success = [t for t in trades if t.is_success()]
        failed = [t for t in trades if t.is_failed()]

        return success, failed

        # # 2. Capital allocation
        # # Approvals
        # approvals = approve_tokens(
        #     self.web3,
        #     self.uniswap,
        #     self.hot_wallet,
        #     trades
        # )
        #
        # # 2: prepare
        # # Prepare transactions
        # prepare_swaps(
        #     self.web3,
        #     self.hot_wallet,
        #     self.uniswap,
        #     ts,
        #     state,
        #     trades,
        #     underflow_check=False,
        # )
        #
        # #: 3 broadcast
        #
        # # Handle approvals separately for now.
        # # We do not need to wait these to confirm.
        # confirm_approvals(
        #     self.web3,
        #     approvals,
        #     confirmation_block_count=self.confirmation_block_count,
        #     max_timeout=self.confirmation_timeout)
        #
        # broadcasted = broadcast(
        #     self.web3,
        #     ts,
        #     trades,
        #     confirmation_block_count=self.confirmation_block_count,
        # )
        # #assert trade.get_status() == TradeStatus.broadcasted
        #
        # # Resolve
        # receipts = wait_trades_to_complete(
        #     self.web3,
        #     trades,
        #     confirmation_block_count=self.confirmation_block_count,
        #     max_timeout=self.confirmation_timeout)
        #
        # resolve_trades(
        #     self.web3,
        #     self.uniswap,
        #     ts,
        #     state,
        #     broadcasted,
        #     receipts,
        #     stop_on_execution_failure=False)
        #
        # # Clean up failed trades
        # return freeze_position_on_failed_trade(ts, state, trades)

    def get_routing_state_details(self) -> object:
        """Prototype does not know much about routing."""
        return {
            "web3": self.web3,
            "hot_wallet": self.hot_wallet,
        }