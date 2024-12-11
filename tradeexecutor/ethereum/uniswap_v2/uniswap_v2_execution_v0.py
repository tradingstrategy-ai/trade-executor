"""Execution model where trade happens directly on Uniswap v2 style exchange.

TODO: Prototype code path. Only preserved for having unit test suite green.
It has slight API incompatibilities in the later versions.
"""

import datetime
from decimal import Decimal
from typing import List, Tuple, Optional
import logging

from eth_defi.provider.broken_provider import get_almost_latest_block_number
from hexbytes import HexBytes
from web3 import Web3

from eth_defi.gas import estimate_gas_fees
from eth_defi.hotwallet import HotWallet
from eth_defi.uniswap_v2.deployment import UniswapV2Deployment
from eth_defi.token import fetch_erc20_details
from eth_defi.trade import TradeSuccess
from eth_defi.uniswap_v2.deployment import mock_partial_deployment_for_analysis
from eth_defi.uniswap_v2.analysis import analyse_trade_by_receipt
from tradeexecutor.ethereum.tx import HotWalletTransactionBuilder
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_routing import UniswapV2RoutingState, UniswapV2Routing
from tradeexecutor.ethereum.execution import broadcast, wait_trades_to_complete, update_confirmation_status
from tradeexecutor.ethereum.swap import report_failure, get_swap_transactions
from tradeexecutor.state.freeze import freeze_position_on_failed_trade
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.blockhain_transaction import BlockchainTransaction
from tradeexecutor.state.types import BlockNumber
from tradeexecutor.strategy.execution_model import ExecutionModel
from tradeexecutor.strategy.routing import RoutingModel, RoutingState

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

    TODO: This model was used in the first prototype and will be later discarded.
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

    def get_balance_address(self):
        return None

    def get_safe_latest_block(self) -> BlockNumber:
        web3 = self.web3
        return get_almost_latest_block_number(web3)

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
        assert isinstance(routing_model, UniswapV2Routing), f"Received {routing_state.__class__}"
        assert isinstance(routing_state, UniswapV2RoutingState), f"Received {routing_state.__class__}"

        fees = estimate_gas_fees(self.web3)

        tx_builder = HotWalletTransactionBuilder(
            self.web3,
            self.hot_wallet,
        )

        reserve_asset, rate = state.portfolio.get_default_reserve_asset()

        # We know only about one exchange
        routing_model = UniswapV2Routing(
            factory_router_map={
                self.uniswap.factory.address: (self.uniswap.router.address, self.uniswap.init_code_hash),
            },
            allowed_intermediary_pairs={},
            reserve_token_address=reserve_asset.address,
            trading_fee=0.0,
        )

        state.start_execution_all(datetime.datetime.utcnow(), trades)

        routing_model.setup_trades(
            state,
            routing_state,
            trades,
            check_balances=check_balances
        )
        broadcast_and_resolve(self.web3, state, trades, stop_on_execution_failure=self.stop_on_execution_failure) # TODO fix if needs be? deprecated? 

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

    def repair_unconfirmed_trades(self, state: State) -> List[TradeExecution]:
        raise NotImplementedError()
    
    @staticmethod
    def is_v3():
        return False
    
    
def broadcast_and_resolve(
    web3: Web3,
    state: State,
    trades: List[TradeExecution],
    confirmation_timeout: datetime.timedelta = datetime.timedelta(minutes=1),
    confirmation_block_count: int=0,
    stop_on_execution_failure=False,
):
    """Do the live trade execution.

    - Push trades to a live blockchain

    - Wait transactions to be mined

    - Based on the transaction result, update the state of the trade if it was success or not

    :param confirmation_block_count:
        How many blocks to wait until marking transaction as confirmed

    :confirmation_timeout:
        Max time to wait for a confirmation.

        We can use zero or negative values to simulate unconfirmed trades.
        See `test_broadcast_failed_and_repair_state`.

    :param stop_on_execution_failure:
        If any of the transactions fail, then raise an exception.
        Set for unit test.
    """

    assert isinstance(confirmation_timeout, datetime.timedelta)

    broadcasted = broadcast(web3, datetime.datetime.utcnow(), trades)

    if confirmation_timeout > datetime.timedelta(0):

        receipts = wait_trades_to_complete(
            web3,
            trades,
            max_timeout=confirmation_timeout,
            confirmation_block_count=confirmation_block_count,
        )

        resolve_trades(
            web3,
            datetime.datetime.now(),
            state,
            broadcasted,
            receipts,
            stop_on_execution_failure=stop_on_execution_failure)
        
def resolve_trades(
    web3: Web3,
    ts: datetime.datetime,
    state: State,
    tx_map: dict[HexBytes, Tuple[TradeExecution, BlockchainTransaction]],
    receipts: dict[HexBytes, dict],
    stop_on_execution_failure=True):
    """Resolve trade outcome.

    Read on-chain Uniswap swap data from the transaction receipt and record how it went.

    Mutates the trade objects in-place.

    :param tx_map:
        tx hash -> (trade, transaction) mapping

    :param receipts:
        tx hash -> receipt object mapping

    :param stop_on_execution_failure:
        Raise an exception if any of the trades failed"""

    trades = update_confirmation_status(web3, ts, tx_map, receipts)

    # Then resolve trade status by analysis the tx receipt
    # if the blockchain transaction was successsful.
    # Also get the actual executed token counts.
    for trade in trades:
        base_token_details = fetch_erc20_details(web3, trade.pair.base.checksum_address)
        quote_token_details = fetch_erc20_details(web3, trade.pair.quote.checksum_address)
        reserve = trade.reserve_currency
        swap_tx = get_swap_transactions(trade)
        uniswap = mock_partial_deployment_for_analysis(web3, swap_tx.contract_address)

        tx_dict = swap_tx.get_transaction()
        receipt = receipts[HexBytes(swap_tx.tx_hash)]

        result = analyse_trade_by_receipt(web3, uniswap, tx_dict, swap_tx.tx_hash, receipt)

        if isinstance(result, TradeSuccess):
            
            # v3 path includes fee (int) as well
            path = [a.lower() for a in result.path if type(a) == str]
            
            if trade.is_buy():
                assert path[0] == reserve.address, f"Was expecting the route path to start with reserve token {reserve}, got path {result.path}"

                price = 1 / result.price

                executed_reserve = result.amount_in / Decimal(10**quote_token_details.decimals)
                executed_amount = result.amount_out / Decimal(10**base_token_details.decimals)
            else:
                # Ordered other way around
                assert path[0] == base_token_details.address.lower(), f"Path is {path}, base token is {base_token_details}"
                assert path[-1] == reserve.address
                
                price = result.price
                
                executed_amount = -result.amount_in / Decimal(10**base_token_details.decimals)
                executed_reserve = result.amount_out / Decimal(10**quote_token_details.decimals)

            assert (executed_reserve > 0) and (executed_amount != 0) and (price > 0), f"Executed amount {executed_amount}, executed_reserve: {executed_reserve}, price: {price}, tx info {trade.tx_info}"

            # Mark as success
            state.mark_trade_success(
                ts,
                trade,
                executed_price=float(price),
                executed_amount=executed_amount,
                executed_reserve=executed_reserve,
                lp_fees=0,
                native_token_price=1.0,
            )
        else:
            report_failure(ts, state, trade, stop_on_execution_failure)