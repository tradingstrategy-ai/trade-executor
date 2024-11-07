"""Dealing with Ethereum low level tranasctions."""

import logging
import datetime
import warnings
from collections import Counter
from decimal import Decimal
from itertools import chain
from typing import List, Dict, Set, Tuple

from eth_account.datastructures import SignedTransaction
from eth_defi.provider.broken_provider import get_almost_latest_block_number
from eth_typing import HexAddress, HexStr
from hexbytes import HexBytes
from web3 import Web3

from eth_defi.abi import get_deployed_contract
from eth_defi.deploy import get_or_create_contract_registry
from eth_defi.gas import GasPriceSuggestion, apply_gas, estimate_gas_fees
from eth_defi.hotwallet import HotWallet, SignedTransactionWithNonce
from eth_defi.provider.fallback import FallbackProvider
from eth_defi.provider.mev_blocker import MEVBlockerProvider, get_mev_blocker_provider
from eth_defi.token import fetch_erc20_details, TokenDetails
from eth_defi.confirmation import wait_transactions_to_complete, \
    broadcast_and_wait_transactions_to_complete, broadcast_transactions, wait_and_broadcast_multiple_nodes, wait_and_broadcast_multiple_nodes_mev_blocker
from eth_defi.trace import trace_evm_transaction, print_symbolic_trace
from eth_defi.uniswap_v2.deployment import UniswapV2Deployment, FOREVER_DEADLINE
from eth_defi.revert_reason import fetch_transaction_revert_reason


from tradingstrategy.chain import ChainId
from tradeexecutor.ethereum.tx import TransactionBuilder
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution, TradeStatus
from tradeexecutor.state.blockhain_transaction import BlockchainTransaction
from tradeexecutor.state.freeze import freeze_position_on_failed_trade
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_routing import UniswapV2Routing, UniswapV2RoutingState
from tradeexecutor.ethereum.uniswap_v3.uniswap_v3_routing import UniswapV3Routing, UniswapV3RoutingState
from tradeexecutor.state.types import BlockNumber
from tradeexecutor.strategy.execution_model import ExecutionModel, RoutingStateDetails
from tradeexecutor.strategy.generic.generic_router import GenericRouting
from tradeexecutor.strategy.routing import RoutingModel, RoutingState
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.ethereum.ethereum_protocol_adapters import EthereumPairConfigurator


logger = logging.getLogger(__name__)


# TODO check with tradeexuctor.strategy.execution ExecutionModel
class EthereumExecution(ExecutionModel):
    """Run order execution on a single Uniswap v2 style exchanges."""

    def __init__(
        self,
        tx_builder: TransactionBuilder,
        min_balance_threshold=Decimal("0.5"),
        confirmation_block_count=6,
        confirmation_timeout=datetime.timedelta(minutes=5),
        max_slippage: float = 0.01,
        stop_on_execution_failure=True,
        swap_gas_fee_limit=2_000_000,
        mainnet_fork=False,
        force_sequential_broadcast=False,
    ):
        """
        :param tx_builder:
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

            TODO: No longer used. Set in `TradeExecution` object.

            Max slippage tolerance per trade. 0.01 is 1%.

        :param mainnet_fork:
            Is this Anvil in automining mainnet fork mode
        """
        assert isinstance(tx_builder, TransactionBuilder), f"Got {tx_builder} {tx_builder.__class__}"
        assert isinstance(confirmation_timeout, datetime.timedelta), f"Got {confirmation_timeout} {confirmation_timeout.__class__}"
        self.tx_builder = tx_builder
        self.stop_on_execution_failure = stop_on_execution_failure
        self.min_balance_threshold = min_balance_threshold
        self.confirmation_block_count = confirmation_block_count
        self.confirmation_timeout = confirmation_timeout
        self.swap_gas_fee_limit = swap_gas_fee_limit
        self.max_slippage = max_slippage
        self.mainnet_fork = mainnet_fork
        self.force_sequential_broadcast = force_sequential_broadcast
        logger.info(
            "Execution model %s created.\n confirmation_block_count: %s, confirmation_timeout: %s, mainnet_fork: %s, force_sequential_broadcast: %s",
            self.__class__.__name__,
            self.confirmation_block_count,
            self.confirmation_timeout,
            self.mainnet_fork,
            self.force_sequential_broadcast,
        )

    @property
    def web3(self):
        return self.tx_builder.web3

    @property
    def chain_id(self) -> int:
        """Which chain the live execution is connected to."""
        return self.web3.eth.chain_id

    def get_balance_address(self) -> str:
        return self.tx_builder.get_erc_20_balance_address()

    def get_safe_latest_block(self) -> BlockNumber:
        web3 = self.web3
        return get_almost_latest_block_number(web3)

    @staticmethod
    def pre_execute_assertions(
        ts: datetime.datetime, 
        routing_model: UniswapV2Routing | UniswapV3Routing,
        routing_state: UniswapV2RoutingState | UniswapV3RoutingState
    ):
        assert isinstance(ts, datetime.datetime)

        if isinstance(routing_model, UniswapV2Routing):
            assert isinstance(routing_state, UniswapV2RoutingState), "Incorrect routing_state specified"
        elif isinstance(routing_model, UniswapV3Routing):
            assert isinstance(routing_state, UniswapV3RoutingState), "Incorrect routing_state specified"
        else:
            raise ValueError("Incorrect routing model specified")

    def repair_unconfirmed_trades(self, state: State) -> List[TradeExecution]:
        """Repair unconfirmed trades.

        Repair trades that failed to properly broadcast or confirm due to
        blockchain node issues.
        """

        repaired = []

        logger.info("Repairing trades, using execution model %s", self)

        assert self.confirmation_timeout > datetime.timedelta(0), \
                "Make sure you have a good tx confirmation timeout setting before attempting a repair"

        # Check if we are on a live chain, not Ethereum Tester
        if self.web3.eth.chain_id != 61:
            assert self.confirmation_block_count > 0, \
                    "Make sure you have a good confirmation_block_count setting before attempting a repair"

        trades_to_be_repaired = []

        total_trades = 0

        logger.info("Strategy has %d frozen positions", len(list(state.portfolio.frozen_positions)))

        for p in chain(state.portfolio.open_positions.values(), state.portfolio.frozen_positions.values()):
            t: TradeExecution
            for t in p.trades.values():
                if t.is_unfinished() or t.is_failed():
                    logger.info("Found a trade: %s", t)
                    trades_to_be_repaired.append(t)
                total_trades += 1

        assert total_trades > 0, "No executed trades found on the strategy"

        logger.info("Strategy has total %d trades", total_trades)

        if not trades_to_be_repaired:
            return []

        print("Found %d trades to be repaired", len(trades_to_be_repaired))
        confirmation = input("Attempt to repair [y/n]").lower()

        if confirmation != "y":
            return []

        for t in trades_to_be_repaired:
            assert t.get_status() == TradeStatus.broadcasted

            receipt_data = wait_trades_to_complete(
                self.web3,
                [t],
                max_timeout=self.confirmation_timeout,
                confirmation_block_count=self.confirmation_block_count,
            )

            assert len(receipt_data) > 0, f"Got bad receipts: {receipt_data}"

            tx_data = {tx.tx_hash: (t, tx) for tx in t.blockchain_transactions}

            self.resolve_trades(
                datetime.datetime.now(),
                state,
                tx_data,
                receipt_data,
                stop_on_execution_failure=True)

            t.repaired_at = datetime.datetime.utcnow()
            t.add_note(f"Failed broadcast repaired at {t.repaired_at}")

            repaired.append(t)

        return repaired

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
            balance = self.tx_builder.get_gas_wallet_balance()
            assert balance > self.min_balance_threshold, f"At least {self.min_balance_threshold} native currency need, our wallet {self.tx_builder.get_gas_wallet_address()} has {balance:.8f}\n" \
                                                         f"If you think believe this balance is enough consider setting MIN_GAS_BALANCE environment variable."

    def initialize(self):
        """Set up the wallet"""
        logger.info("Initialising %s execution model", self.__class__.__name__)
        self.tx_builder.init()
        logger.info("Hot wallet %s has balance %s", self.tx_builder.get_gas_wallet_address(), self.tx_builder.get_gas_wallet_balance())

    def broadcast_and_resolve_old(
        self,
        state: State,
        trades: List[TradeExecution],
        routing_model: RoutingModel,
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

        web3 = self.web3

        logger.info("Using legacy broadcast")
        
        assert isinstance(confirmation_timeout, datetime.timedelta)

        broadcasted = broadcast(web3, datetime.datetime.utcnow(), trades)

        # if confirmation_timeout > datetime.timedelta(0):
        receipts = wait_trades_to_complete(
            web3,
            trades,
            max_timeout=confirmation_timeout,
            confirmation_block_count=confirmation_block_count,
        )

        self.resolve_trades(
            datetime.datetime.now(),
            routing_model,
            state,
            broadcasted,
            receipts,
            stop_on_execution_failure=stop_on_execution_failure)

    def broadcast_and_resolve_mev_blocker(
        self,
        routing_model: RoutingModel,
        state: State,
        trades: List[TradeExecution],
        confirmation_timeout: datetime.timedelta = datetime.timedelta(minutes=1),
        stop_on_execution_failure=False,
    ):
        """Sequential broadcast of txs.

        - Cannot broadcast another tx until previous is confirmed
        """

        web3 = self.web3

        mev_blocker = get_mev_blocker_provider(web3)

        assert mev_blocker or self.force_sequential_broadcast

        # Special path needed on Ethereum mainnet and MEV Blocker
        logger.info(
            "MEV blocker/sequential tx enabled broadcast, mev blocker: %s, sequential: %s",
            mev_blocker,
            self.force_sequential_broadcast,
        )

        if not mev_blocker:
            # Anvil test path
            mev_blocker = web3.provider

        # Broadcast and resolve one by one
        for t in trades:

            current_trade_tx_map = {}
            current_trade_receipts = {}

            for tx in t.blockchain_transactions:

                logger.info(
                    "MEV blocker resolve, tx: %s, nonce %d, trade: #%d, timeout %s",
                    tx.hash.hex(),
                    tx.nonce,
                    t.trade_id,
                    confirmation_timeout,
                )

                signed_tx = SignedTransactionWithNonce(
                    rawTransaction=HexBytes(tx.signed_bytes),
                    hash=HexBytes(tx.tx_hash),
                    r=0,  # Not needed in this stage
                    s=0,  # Not needed in this stage
                    v=0,  # Not needed in this stage
                    address=tx.from_address,
                    nonce=tx.nonce,
                    source=tx.details,
                )

                receipts  = wait_and_broadcast_multiple_nodes_mev_blocker(
                    mev_blocker,
                    [signed_tx],
                    max_timeout=confirmation_timeout,
                )

                current_trade_tx_map[signed_tx.hash] = (t, tx)
                current_trade_receipts.update(receipts)

            self.resolve_trades(
                datetime.datetime.now(),
                routing_model,
                state,
                current_trade_tx_map,
                current_trade_receipts,
                stop_on_execution_failure=stop_on_execution_failure
            )

    def broadcast_and_resolve_multiple_nodes(
        self,
        routing_model: RoutingModel,
        state: State,
        trades: List[TradeExecution],
        confirmation_timeout: datetime.timedelta = datetime.timedelta(minutes=1),
        confirmation_block_count: int = 0,
        stop_on_execution_failure=False,
        rebroadcast=False,
    ):
        """Do the live trade execution using multiple nodes.

        See :py:func:`eth_defi.confirmation.wait_and_broadcast_multiple_nodes`

        :param state:
            The current state of the strategy

        :param trades:
            List of trades we need to execute on-chain

        :param confirmation_block_count:
            How many blocks to wait until marking transaction as confirmed

        :param confirmation_timeout:
            Max time to wait for a confirmation.

            We can use zero or negative values to simulate unconfirmed trades.
            See `test_broadcast_failed_and_repair_state`.

        :param stop_on_execution_failure:
            If any of the transactions fail, then raise an exception.
            Set for unit test.
        """
        assert isinstance(confirmation_timeout, datetime.timedelta)
        assert isinstance(routing_model, RoutingModel)

        web3 = self.web3
        logger.info(
            "Using multi-node broadcast for %s, mainnet fork flag is %s",
            web3.provider,
            self.mainnet_fork,
        )

        # Uncofirmed trade test, never confirm anything
        if confirmation_timeout == datetime.timedelta(0):
            logger.info("Unit test path of no confirmation")
            return

        txs: Set[SignedTransactionWithNonce] = set()
        tx_map: Dict[HexStr, tuple] = dict()

        for t in trades:
            assert len(t.blockchain_transactions) > 0, f"Trade {t} does not have any blockchain transactions prepared"
            for tx in t.blockchain_transactions:
                assert tx.signed_bytes, f"Unsigned transaction: {tx}"
                assert tx.tx_hash is not None
                signed_tx = SignedTransactionWithNonce(
                    rawTransaction=HexBytes(tx.signed_bytes),
                    hash=HexBytes(tx.tx_hash),
                    r=0,  # Not needed in this stage
                    s=0,  # Not needed in this stage
                    v=0,  # Not needed in this stage
                    address=tx.from_address,
                    nonce=tx.nonce,
                    source=tx.details,
                )
                txs.add(signed_tx)
                logger.info("Broadcasting transaction %s, nonce %s, for trade\n:%s", signed_tx.hash.hex(), signed_tx.nonce, t)
                tx_map[signed_tx.hash.hex()] = (t, tx)

            t.mark_broadcasted(datetime.datetime.utcnow(), rebroadcast=rebroadcast)

        logger.info("Normal tx broadcast")
        receipts = wait_and_broadcast_multiple_nodes(
            web3,
            txs,
            max_timeout=confirmation_timeout,
            confirmation_block_count=confirmation_block_count,
            node_switch_timeout=datetime.timedelta(minutes=1),  # Rebroadcast every 1 minute
            check_nonce_validity=not rebroadcast,
            mine_blocks=self.mainnet_fork,
        )

        self.resolve_trades(
            datetime.datetime.now(),
            routing_model,
            state,
            tx_map,
            receipts,
            stop_on_execution_failure=stop_on_execution_failure
        )

    def execute_trades(
        self,
        ts: datetime.datetime,
        state: State,
        trades: List[TradeExecution],
        routing_model: RoutingModel,
        routing_state: RoutingState,
        check_balances=False,
        rebroadcast=False,
        triggered=False,
    ):

        if self.web3.eth.chain_id not in (ChainId.ethereum_tester.value, ChainId.anvil.value):
            if not self.mainnet_fork:
                assert self.confirmation_block_count > 0, f"confirmation_block_count set to {self.confirmation_block_count} "

        if not rebroadcast:
            state.start_execution_all(
                datetime.datetime.utcnow(),
                trades,
                max_slippage=self.max_slippage,
                rebroadcast=rebroadcast,
            )

        routing_model.setup_trades(
            routing_state,
            trades,
            check_balances=check_balances,
            rebroadcast=rebroadcast,
        )

        if isinstance(self.web3.provider, (FallbackProvider, MEVBlockerProvider)) or self.force_sequential_broadcast:
            self.broadcast_and_resolve_mev_blocker(
                routing_model,
                state,
                trades,
                confirmation_timeout=self.confirmation_timeout,
                confirmation_block_count=self.confirmation_block_count,
                rebroadcast=rebroadcast,
            )
        elif isinstance(self.web3.provider, (FallbackProvider)):
            # Multi node broadcast
            self.broadcast_and_resolve_multiple_nodes(
                routing_model,
                state,
                trades,
                confirmation_timeout=self.confirmation_timeout,
                confirmation_block_count=self.confirmation_block_count,
                rebroadcast=rebroadcast,
            )

        else:
            # Rebroadcast not supported for the old code path
            self.broadcast_and_resolve_old(
                state,
                trades,
                routing_model,
                confirmation_timeout=self.confirmation_timeout,
                confirmation_block_count=self.confirmation_block_count,
            )

        # Clean up failed trades
        freeze_position_on_failed_trade(ts, state, trades)

    def get_routing_state_details(self) -> RoutingStateDetails:
        return {
            "tx_builder": self.tx_builder,
        }

    def update_confirmation_status(
        self, 
        ts: datetime.datetime,
        tx_map: Dict[HexBytes, Tuple[TradeExecution, BlockchainTransaction]],
        receipts: Dict[HexBytes, dict]
    ) -> set[TradeExecution]:
        """First update the state of all transactions, as we now have receipt for them. Update the transaction confirmation status"""
        return update_confirmation_status(self.web3, ts, tx_map, receipts)
    
    def resolve_trades(
        self,
        ts: datetime.datetime,
        routing_model: RoutingModel,
        state: State,
        tx_map: Dict[HexStr, Tuple[TradeExecution, BlockchainTransaction]],
        receipts: Dict[HexBytes, dict],
        stop_on_execution_failure=True
    ):
        """Resolve trade outcome.

        Read on-chain Uniswap swap data from the transaction receipt and record how it went.

        Mutates the trade objects in-place.

        :param tx_map:
            tx hash -> (trade, transaction) mapping

        :param receipts:
            tx hash -> receipt object mapping

        :param stop_on_execution_failure:
            Raise an exception if any of the trades failed"""

        assert isinstance(state, State)
        assert isinstance(routing_model, RoutingModel)

        web3 = self.web3

        trades = self.update_confirmation_status(ts, tx_map, receipts)

        # Then resolve trade status by analysis the tx receipt
        # if the blockchain transaction was successsful.
        # Also get the actual executed token counts.
        for trade in trades:
            routing_model.settle_trade(
                web3,
                state,
                trade,
                receipts,
                stop_on_execution_failure=stop_on_execution_failure,
            )

    def create_default_routing_model(
        self,
        strategy_universe: TradingStrategyUniverse,
    ) -> RoutingModel:
        """Get the default routing model for this executor.

        :return:

        """
        web3 = self.web3
        configurator = EthereumPairConfigurator(web3, strategy_universe)
        return GenericRouting(configurator)


# Only usage outside this module is UniswapV2ExecutionModelV0
def update_confirmation_status(
        web3: Web3,
        ts: datetime.datetime,
        tx_map: Dict[HexBytes, Tuple[TradeExecution, BlockchainTransaction]],
        receipts: Dict[HexBytes, dict]
    ) -> set[TradeExecution]:
        """First update the state of all transactions, as we now have receipt for them. Update the transaction confirmation status"""
        
        trades: set[TradeExecution] = set()

        # First update the state of all transactions,
        # as we now have receipt for them
        for tx_hash, receipt in receipts.items():
            trade, tx = tx_map[tx_hash.hex()]
            # Update the transaction confirmation status
            status = receipt["status"] == 1
            block_number = receipt["blockNumber"]
            logger.info(
                f"Resolved tx %s as %s at block {block_number:,} for trade\n%s",
                tx_hash.hex(),
                "success" if status else "reverted",
                trade)
            reason = None
            stack_trace = None

            # Transaction failed,
            # try to get as much as information possible
            if status == 0:
                reason = fetch_transaction_revert_reason(web3, tx_hash)

                if web3.eth.chain_id == ChainId.anvil.value:
                    trace_data = trace_evm_transaction(web3, tx_hash)
                    stack_trace = print_symbolic_trace(get_or_create_contract_registry(web3), trace_data)

            tx.set_confirmation_information(
                ts,
                receipt["blockNumber"],
                receipt["blockHash"].hex(),
                receipt.get("effectiveGasPrice", 0),
                receipt["gasUsed"],
                status,
                revert_reason=reason,
                stack_trace=stack_trace,
            )
            trades.add(trade)
        
        return trades                


# Only usage outside this module is UniswapV2ExecutionModelV0


def translate_to_naive_swap(
        web3: Web3,
        deployment: UniswapV2Deployment,
        hot_wallet: HotWallet,
        t: TradeExecution,
        gas_fees: GasPriceSuggestion,
        base_token_details: TokenDetails,
        quote_token_details: TokenDetails,
    ):
    """Creates an AMM swap tranasction out of buy/sell.

    If buy tries to do the best execution for given `planned_reserve`.

    If sell tries to do the best execution for given `planned_quantity`.

    Route only between two pools - stablecoin reserve and target buy/sell.

    Any gas price is set by `web3` instance gas price strategy.

    :param t:
    :return: Unsigned transaction
    """

    if t.is_buy():
        amount0_in = int(t.planned_reserve * 10**quote_token_details.decimals)
        path = [quote_token_details.address, base_token_details.address]
        t.reserve_currency_allocated = t.planned_reserve
    else:
        # Reverse swap
        amount0_in = int(-t.planned_quantity * 10**base_token_details.decimals)
        path = [base_token_details.address, quote_token_details.address]
        t.reserve_currency_allocated = 0

    args = [
        amount0_in,
        0,
        path,
        hot_wallet.address,
        FOREVER_DEADLINE,
    ]

    # https://docs.uniswap.org/protocol/V2/reference/smart-contracts/router-02#swapexacttokensfortokens
    # https://web3py.readthedocs.io/en/stable/web3.eth.account.html#sign-a-contract-transaction
    tx = deployment.router.functions.swapExactTokensForTokens(
        *args,
    ).build_transaction({
        'chainId': web3.eth.chain_id,
        'gas': 350_000,  # Estimate max 350k gas per swap
        'from': hot_wallet.address,
    })

    apply_gas(tx, gas_fees)

    signed = hot_wallet.sign_transaction_with_new_nonce(tx)
    selector = deployment.router.functions.swapExactTokensForTokens

    # Create record of this transaction
    tx_info = t.tx_info = BlockchainTransaction()
    tx_info.set_target_information(
        web3.eth.chain_id,
        deployment.router.address,
        selector.fn_name,
        args,
        tx,
    )

    tx_info.set_broadcast_information(tx["nonce"], signed.hash.hex(), signed.rawTransaction.hex())


def prepare_swaps(
        web3: Web3,
        hot_wallet: HotWallet,
        uniswap: UniswapV2Deployment,
        ts: datetime.datetime,
        state: State,
        instructions: List[TradeExecution],
        underflow_check=True) -> Dict[HexAddress, int]:
    """Prepare multiple swaps to be breoadcasted parallel from the hot wallet.

    :param underflow_check: Do we check we have enough cash in hand before trying to prepare trades.
        Note that because when executing sell orders first, we will have more cash in hand to make buys.

    :return: Token approvals we need to execute the trades
    """

    # Get our starting nonce
    gas_fees = estimate_gas_fees(web3)

    for idx, t in enumerate(instructions):

        base_token_details = fetch_erc20_details(web3, t.pair.base.checksum_address)
        quote_token_details = fetch_erc20_details(web3, t.pair.quote.checksum_address)

        assert base_token_details.decimals is not None, f"Bad token at {t.pair.base.address}"
        assert quote_token_details.decimals is not None, f"Bad token at {t.pair.quote.address}"

        state.portfolio.check_for_nonce_reuse(hot_wallet.current_nonce)

        translate_to_naive_swap(
            web3,
            uniswap,
            hot_wallet,
            t,
            gas_fees,
            base_token_details,
            quote_token_details,
        )

        if t.is_buy():
            state.portfolio.move_capital_from_reserves_to_spot_trade(t, underflow_check=underflow_check)

        t.started_at = datetime.datetime.utcnow()


def approve_tokens(
        web3: Web3,
        deployment: UniswapV2Deployment,
        hot_wallet: HotWallet,
        instructions: List[TradeExecution],
    ) -> List[SignedTransaction]:
    """Approve multiple ERC-20 token allowances for the trades needed.

    Each token is approved only once. E.g. if you have 4 trades using USDC,
    you will get 1 USDC approval.
    """

    signed = []

    approvals = Counter()

    for t in instructions:
        base_token_details = fetch_erc20_details(web3, t.pair.base.checksum_address)
        quote_token_details = fetch_erc20_details(web3, t.pair.quote.checksum_address)

        # Update approval counters for the whole batch
        if t.is_buy():
            approvals[quote_token_details.address] += int(t.planned_reserve * 10**quote_token_details.decimals)
        else:
            approvals[base_token_details.address] += int(-t.planned_quantity * 10**base_token_details.decimals)

    for tpl in approvals.items():
        token_address, amount = tpl

        assert amount > 0, f"Got a non-positive approval {token_address}: {amount}"

        token = get_deployed_contract(web3, "IERC20.json", token_address)
        tx = token.functions.approve(
            deployment.router.address,
            amount,
        ).build_transaction({
            'chainId': web3.eth.chain_id,
            'gas': 100_000,  # Estimate max 100k per approval
            'from': hot_wallet.address,
        })
        signed.append(hot_wallet.sign_transaction_with_new_nonce(tx))

    return signed


def approve_infinity(
        web3: Web3,
        deployment: UniswapV2Deployment,
        hot_wallet: HotWallet,
        instructions: List[TradeExecution],
    ) -> List[SignedTransaction]:
    """Approve multiple ERC-20 token allowances for the trades needed.

    Each token is approved only once. E.g. if you have 4 trades using USDC,
    you will get 1 USDC approval.
    """

    signed = []

    approvals = Counter()

    for t in instructions:
        base_token_details = fetch_erc20_details(web3, t.pair.base.checksum_address)
        quote_token_details = fetch_erc20_details(web3, t.pair.quote.checksum_address)

        # Update approval counters for the whole batch
        if t.is_buy():
            approvals[quote_token_details.address] += int(t.planned_reserve * 10**quote_token_details.decimals)
        else:
            approvals[base_token_details.address] += int(-t.planned_quantity * 10**base_token_details.decimals)

    for tpl in approvals.items():
        token_address, amount = tpl

        assert amount > 0, f"Got a non-positive approval {token_address}: {amount}"

        token = get_deployed_contract(web3, "IERC20.json", token_address)
        tx = token.functions.approve(
            deployment.router.address,
            amount,
        ).build_transaction({
            'chainId': web3.eth.chain_id,
            'gas': 100_000,  # Estimate max 100k per approval
            'from': hot_wallet.address,
        })
        signed.append(hot_wallet.sign_transaction_with_new_nonce(tx))

    return signed


def confirm_approvals(
        web3: Web3,
        txs: List[SignedTransaction],
        confirmation_block_count=0,
        max_timeout=datetime.timedelta(minutes=5),
    ):
    """Wait until all transactions are confirmed.

    :param confirmation_block_count: How many blocks to wait for the transaction to settle

    :raise: If any of the transactions fail
    """
    logger.info("Confirming %d approvals, confirmation_block_count is %d", len(txs), confirmation_block_count)
    receipts = broadcast_and_wait_transactions_to_complete(
        web3,
        txs,
        confirmation_block_count=confirmation_block_count,
        max_timeout=max_timeout,
    )
    return receipts


def broadcast(
        web3: Web3,
        ts: datetime.datetime,
        instructions: List[TradeExecution],
        confirmation_block_count: int=0,
        ganache_sleep=0.5,
) -> Dict[HexBytes, Tuple[TradeExecution, BlockchainTransaction]]:
    """Broadcast multiple transactions and manage the trade executor state for them.

    .. note ::

        The node provider may or may not support broadcasting multiple transactions without confirming existing ones.
        For example, LlamaNode will give nonce too low error. We will try to deal with this in the middleware.

    :return: Map of transaction hashes to watch
    """

    warnings.warn('This function is deprecated. Use multi-node broadcasting instead', DeprecationWarning, stacklevel=2)

    logger.info("Broadcasting %d trades", len(instructions))

    res = {}
    # Another nonce guard
    nonces: Set[int] = set()

    broadcast_batch: List[SignedTransactionWithNonce] = []

    for t in instructions:
        assert len(t.blockchain_transactions) > 0, f"Trade {t} does not have any blockchain transactions prepared"
        for tx in t.blockchain_transactions:
            assert isinstance(tx.signed_bytes, str), f"Got signed transaction: {t.tx_info.signed_bytes}"
            assert tx.nonce not in nonces, "Nonce already used"
            nonces.add(tx.nonce)
            tx.broadcasted_at = ts
            res[tx.tx_hash] = (t, tx)
            # Only SignedTransaction.rawTransaction attribute is intresting in this point
            signed_tx = tx.get_tx_object()
            broadcast_batch.append(signed_tx)
            logger.info("Broadcasting transaction for trade %s:\n %s", t, tx)
        t.mark_broadcasted(datetime.datetime.utcnow())

    try:
        hashes = broadcast_transactions(web3, broadcast_batch, confirmation_block_count=confirmation_block_count)
    except Exception as e:
        # Node error:
        # This happens when Polygon chain is busy.
        # We want to add more error information here
        # ValueError: {'code': -32000, 'message': 'tx fee (6.23 ether) exceeds the configured cap (1.00 ether)'}
        for t in instructions:
            logger.error("Could not transactions for broadcast trade: %s", t)
            for tx in t.blockchain_transactions:
                logger.error("Transaction: %s, planned gas price: %s, gas limit: %s", tx, tx.get_planned_gas_price(), tx.get_gas_limit())
        raise e

    assert len(hashes) >= len(instructions), f"We got {len(hashes)} hashes for {len(instructions)} trades"
    return res


def wait_trades_to_complete(
        web3: Web3,
        trades: List[TradeExecution],
        confirmation_block_count=0,
        max_timeout=datetime.timedelta(minutes=5),
        poll_delay=datetime.timedelta(seconds=1)) -> Dict[HexBytes, dict]:
    """Watch multiple transactions executed at parallel.

    :return: Map of transaction hashes -> receipt
    """
    logger.info("Waiting %d trades to confirm, confirm block count %d, timeout %s", len(trades), confirmation_block_count, max_timeout)
    assert isinstance(confirmation_block_count, int)
    tx_hashes = []
    for t in trades:
        tx_hashes.extend(tx.tx_hash for tx in t.blockchain_transactions)
    receipts = wait_transactions_to_complete(web3, tx_hashes, confirmation_block_count, max_timeout, poll_delay)
    return receipts


def get_held_assets(web3: Web3, address: HexAddress, assets: List[AssetIdentifier]) -> Dict[str, Decimal]:
    """Get list of assets hold by the a wallet  ."""

    result = {}
    for asset in assets:
        token_details = fetch_erc20_details(web3, asset.checksum_address)
        balance = token_details.contract.functions.balanceOf(address).call()
        result[token_details.address.lower()] = Decimal(balance) / Decimal(10 ** token_details.decimals)
    return result