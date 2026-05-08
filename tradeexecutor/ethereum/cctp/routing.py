"""CCTP bridge routing model.

Handles routing of CCTP bridge trades — creates burn transactions
on the source chain and receive transactions on the destination chain.
"""

import logging
from typing import Callable

from eth_defi.chain import fetch_block_timestamp
from eth_defi.compat import native_datetime_utc_now
from hexbytes import HexBytes

from tradeexecutor.ethereum.tx import TransactionBuilder
from tradeexecutor.ethereum.web3config import Web3Config
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.routing import RoutingModel, RoutingState

logger = logging.getLogger(__name__)


class CctpBridgeRoutingState(RoutingState):
    """Tracks CCTP bridge transactions for the current cycle."""

    def __init__(self, tx_builder: TransactionBuilder | None = None):
        self.tx_builder = tx_builder


class CctpBridgeRouting(RoutingModel):
    """Routes CCTP bridge trades.

    Creates burn transactions on the source chain via Circle's CCTP V2
    ``depositForBurn`` contract call. The receive/mint on the destination
    chain is handled in a separate phase after attestation.

    :param web3config:
        Web3Config with connections to all chains involved in bridging.

    :param custody_address_resolver:
        Optional callable that maps a chain_id to the custody address on
        that chain.  For Lagoon/Safe multichain setups each chain has its
        own Safe; the resolver returns the correct one so that
        ``depositForBurn`` mints USDC to the right destination.
        When ``None``, the source chain ``tx_builder`` balance address is
        used as a fallback (correct for hot-wallet deployments where the
        same address exists on every chain).

    :param attestation_timeout:
        Maximum seconds to wait for Circle's Iris API attestation during
        settlement.  Defaults to 1800 (30 minutes).
    """

    def __init__(
        self,
        web3config: Web3Config,
        custody_address_resolver: Callable[[int], str] | None = None,
        attestation_timeout: float = 1800.0,
        skip_attestation: bool = False,
    ):
        self.web3config = web3config
        self.custody_address_resolver = custody_address_resolver
        self.attestation_timeout = attestation_timeout
        #: When True, settle_trade() marks success after burn confirmation
        #: without polling attestation or broadcasting receiveMessage.
        #: Used for Anvil fork tests where Circle's Iris API is unavailable
        #: and the test spoofs the receive via replace_attester_on_fork().
        self.skip_attestation = skip_attestation
        self._hot_wallet = None

    def create_routing_state(
        self,
        universe,
        execution_details: object,
    ) -> CctpBridgeRoutingState:
        tx_builder = execution_details.get("tx_builder") if isinstance(execution_details, dict) else None
        return CctpBridgeRoutingState(tx_builder=tx_builder)

    def setup_trades(
        self,
        state: State,
        routing_state: RoutingState,
        trades: list[TradeExecution],
        check_balances=False,
        rebroadcast=False,
    ):
        """Set up CCTP bridge trades.

        For each bridge trade:

        1. Get the source chain web3 connection
        2. Create approve transaction for CCTP TokenMessenger
        3. Create depositForBurn transaction
        4. Sign both and attach as BlockchainTransactions on the trade

        The actual blockchain transaction creation uses ``eth_defi.cctp.transfer``
        functions.
        """
        from eth_defi.abi import get_deployed_contract
        from eth_defi.cctp.transfer import (
            get_token_messenger_v2,
            prepare_approve_for_burn,
            prepare_deposit_for_burn,
        )
        from eth_defi.token import USDC_NATIVE_TOKEN
        from tradingstrategy.chain import ChainId

        assert isinstance(routing_state, CctpBridgeRoutingState)
        tx_builder = routing_state.tx_builder
        assert tx_builder is not None, "CctpBridgeRoutingState missing tx_builder"

        self._hot_wallet = tx_builder.hot_wallet

        for trade in trades:
            assert trade.pair.is_cctp_bridge(), \
                f"CctpBridgeRouting received non-bridge trade: {trade}"

            pair = trade.pair

            # Bridge direction depends on buy vs sell:
            # - Buy (forward): burn on quote chain, mint on base chain
            # - Sell (reverse): burn on base chain, mint on quote chain
            if trade.is_buy():
                source_chain_id = pair.quote.chain_id
                dest_chain_id = pair.base.chain_id
            else:
                source_chain_id = pair.base.chain_id
                dest_chain_id = pair.quote.chain_id

            source_web3 = self.web3config.get_connection(ChainId(source_chain_id))
            assert source_web3 is not None, \
                f"No web3 connection for source chain {source_chain_id}"

            amount_raw = int(trade.planned_reserve * (10 ** pair.quote.decimals))

            # Resolve mint recipient based on trade direction.
            # The destination depends on buy vs sell:
            # - Buy (forward): mint on base chain (satellite)
            # - Sell (reverse): mint on quote chain (primary)
            if trade.is_buy():
                mint_dest_chain_id = pair.base.chain_id
            else:
                mint_dest_chain_id = pair.quote.chain_id

            if self.custody_address_resolver is not None:
                token_storage = self.custody_address_resolver(mint_dest_chain_id)
            else:
                # Fallback: hot wallet address (same on all chains)
                token_storage = tx_builder.get_erc_20_balance_address()

            # Resolve burn token address (source chain USDC)
            burn_token_address = USDC_NATIVE_TOKEN.get(source_chain_id)
            assert burn_token_address is not None, \
                f"No USDC address known for source chain {source_chain_id}"

            # Load contract objects needed for signing
            usdc_contract = get_deployed_contract(
                source_web3, "ERC20MockDecimals.json", burn_token_address,
            )
            token_messenger = get_token_messenger_v2(source_web3)

            # Create bound function calls
            approve_fn = prepare_approve_for_burn(
                source_web3, amount=amount_raw, burn_token=burn_token_address,
            )
            burn_fn = prepare_deposit_for_burn(
                source_web3,
                amount=amount_raw,
                destination_chain_id=dest_chain_id,
                mint_recipient=token_storage,
                burn_token=burn_token_address,
            )

            # Sign transactions
            # asset_deltas=[] is required by vault transaction builders
            # (Lagoon, Enzyme) even though CCTP bridges don't have traditional
            # asset deltas — the guard doesn't enforce slippage on bridging.
            approve_tx = tx_builder.sign_transaction(
                usdc_contract,
                approve_fn,
                gas_limit=100_000,
                asset_deltas=[],
                notes=f"CCTP approve {trade.planned_reserve} USDC",
            )
            burn_tx = tx_builder.sign_transaction(
                token_messenger,
                burn_fn,
                gas_limit=300_000,
                asset_deltas=[],
                notes=f"CCTP depositForBurn {trade.planned_reserve} USDC chain {source_chain_id} -> {dest_chain_id}",
            )

            trade.set_blockchain_transactions([approve_tx, burn_tx])

            logger.info(
                "CCTP bridge trade %s: burn %s USDC on chain %d -> chain %d",
                trade.get_short_label(),
                trade.planned_reserve,
                source_chain_id,
                dest_chain_id,
            )

    def needs_sequential_trade_execution(self, trades: list[TradeExecution]) -> bool:
        """CCTP bridges must settle before the next trade starts.

        The attestation wait and ``receiveMessage`` broadcast happen during
        settlement — later trades may depend on the minted USDC.
        """
        return len(trades) > 0

    def settle_trade(
        self,
        web3,
        state: State,
        trade: TradeExecution,
        receipts: dict,
        stop_on_execution_failure=False,
    ):
        """Settle a CCTP bridge trade after broadcast.

        Performs a multi-phase settlement:

        1. Verify the burn receipt succeeded on the source chain.
        2. Poll Circle's Iris API for attestation (Phase 2).
        3. Broadcast ``receiveMessage`` on the destination chain (Phase 3).
        4. Mark the trade as success only when both burn and receive are confirmed.

        If the attestation times out or ``receiveMessage`` reverts, the trade
        is marked as in-transit so it can be retried later.

        :param web3:
            Web3 connection to the **source** chain (where the burn happened).

        :param state:
            Current strategy state.

        :param trade:
            The CCTP bridge trade to settle.

        :param receipts:
            Mapping of tx hash → receipt for broadcasted transactions.

        :param stop_on_execution_failure:
            If ``True``, raise on failure instead of marking trade failed.
        """
        from eth_defi.cctp.attestation import fetch_attestation
        from eth_defi.cctp.receive import prepare_receive_message
        from eth_defi.cctp.transfer import _resolve_cctp_domain, get_message_transmitter_v2
        from eth_defi.hotwallet import HotWallet as EthHotWallet
        from tradingstrategy.chain import ChainId

        from tradeexecutor.ethereum.swap import get_swap_transactions, report_failure
        from tradeexecutor.ethereum.tx import HotWalletTransactionBuilder

        # Phase 1: verify burn
        swap_tx = get_swap_transactions(trade)
        receipt = receipts[HexBytes(swap_tx.tx_hash)]

        if receipt["status"] != 1:
            report_failure(native_datetime_utc_now(), state, trade, stop_on_execution_failure)
            return

        ts = fetch_block_timestamp(web3, receipt["blockNumber"])

        # Determine chains
        pair = trade.pair
        if trade.is_buy():
            source_chain_id = pair.quote.chain_id
            dest_chain_id = pair.base.chain_id
        else:
            source_chain_id = pair.base.chain_id
            dest_chain_id = pair.quote.chain_id

        # Fork/simulation mode: skip attestation and receiveMessage entirely.
        # The test environment spoofs the receive via replace_attester_on_fork().
        if self.skip_attestation:
            state.mark_trade_success(
                ts,
                trade,
                executed_price=1.0,
                executed_amount=trade.planned_quantity,
                executed_reserve=trade.planned_reserve,
                lp_fees=0,
                native_token_price=0,
            )
            return

        # Phase 2: attestation
        source_domain = _resolve_cctp_domain(source_chain_id)
        assert source_domain is not None, f"No CCTP domain for chain {source_chain_id}"

        try:
            attestation = fetch_attestation(
                source_domain=source_domain,
                transaction_hash=swap_tx.tx_hash,
                timeout=self.attestation_timeout,
            )
        except TimeoutError:
            logger.warning("CCTP attestation timed out for trade %s, marking in-transit", trade.trade_id)
            state.mark_bridge_in_transit(ts, trade)
            return

        # Phase 3: receiveMessage on destination chain
        dest_web3 = self.web3config.get_connection(ChainId(dest_chain_id))
        dest_wallet = EthHotWallet(self._hot_wallet.account)
        dest_wallet.sync_nonce(dest_web3)

        message_transmitter = get_message_transmitter_v2(dest_web3)
        receive_fn = prepare_receive_message(dest_web3, attestation.message, attestation.attestation)

        dest_tx_builder = HotWalletTransactionBuilder(dest_web3, dest_wallet)
        receive_tx = dest_tx_builder.sign_transaction(
            message_transmitter,
            receive_fn,
            gas_limit=200_000,
            asset_deltas=[],
            notes=f"CCTP receiveMessage chain {source_chain_id} -> {dest_chain_id}",
        )

        trade.blockchain_transactions.append(receive_tx)

        try:
            dest_web3.eth.send_raw_transaction(HexBytes(receive_tx.signed_bytes))
            receive_tx.broadcasted_at = native_datetime_utc_now()
            receive_receipt = dest_web3.eth.wait_for_transaction_receipt(
                HexBytes(receive_tx.tx_hash), timeout=120
            )
        except Exception as e:
            logger.warning("CCTP receiveMessage failed for trade %s: %s", trade.trade_id, e)
            state.mark_bridge_in_transit(ts, trade)
            return

        # Populate confirmation metadata on the receive tx
        receive_ts = fetch_block_timestamp(dest_web3, receive_receipt["blockNumber"])
        receive_tx.set_confirmation_information(
            ts=receive_ts,
            block_number=receive_receipt["blockNumber"],
            block_hash=receive_receipt["blockHash"].hex() if isinstance(receive_receipt["blockHash"], bytes) else str(receive_receipt["blockHash"]),
            realised_gas_units_consumed=receive_receipt["gasUsed"],
            realised_gas_price=receive_receipt.get("effectiveGasPrice", 0),
            status=receive_receipt["status"] == 1,
            revert_reason=None,
        )

        if receive_receipt["status"] != 1:
            logger.warning("CCTP receiveMessage reverted for trade %s", trade.trade_id)
            state.mark_bridge_in_transit(ts, trade)
            return

        # Success: both burn and receive confirmed
        state.mark_trade_success(
            ts,
            trade,
            executed_price=1.0,
            executed_amount=trade.planned_quantity,
            executed_reserve=trade.planned_reserve,
            lp_fees=0,
            native_token_price=0,
        )
