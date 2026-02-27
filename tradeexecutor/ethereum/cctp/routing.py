"""CCTP bridge routing model.

Handles routing of CCTP bridge trades — creates burn transactions
on the source chain and receive transactions on the destination chain.
"""

import logging

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
    """

    def __init__(self, web3config: Web3Config):
        self.web3config = web3config

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

        for trade in trades:
            assert trade.pair.is_cctp_bridge(), \
                f"CctpBridgeRouting received non-bridge trade: {trade}"

            pair = trade.pair
            source_chain_id = pair.quote.chain_id
            dest_chain_id = pair.base.chain_id

            source_web3 = self.web3config.get_connection(ChainId(source_chain_id))
            assert source_web3 is not None, \
                f"No web3 connection for source chain {source_chain_id}"

            amount_raw = int(trade.planned_reserve * (10 ** pair.quote.decimals))

            # Get the token storage address (wallet/safe) that holds USDC
            token_storage = state.sync.deployment.address

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
            approve_tx = tx_builder.sign_transaction(
                usdc_contract,
                approve_fn,
                gas_limit=100_000,
                notes=f"CCTP approve {trade.planned_reserve} USDC",
            )
            burn_tx = tx_builder.sign_transaction(
                token_messenger,
                burn_fn,
                gas_limit=300_000,
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

    def settle_trade(
        self,
        web3,
        state: State,
        trade: TradeExecution,
        receipts: dict,
        stop_on_execution_failure=False,
    ):
        """Settle a CCTP bridge trade after broadcast.

        CCTP bridge is always 1:1, so we just check the receipt status
        and mark the trade at face value.
        """
        from tradeexecutor.ethereum.execution import report_failure
        from tradeexecutor.ethereum.swap import get_swap_transactions

        swap_tx = get_swap_transactions(trade)
        receipt = receipts[HexBytes(swap_tx.tx_hash)]

        if receipt["status"] == 1:
            ts = fetch_block_timestamp(web3, receipt["blockNumber"])
            state.mark_trade_success(
                ts,
                trade,
                executed_price=1.0,
                executed_amount=trade.planned_quantity,
                executed_reserve=trade.planned_reserve,
                lp_fees=0,
                native_token_price=0,
            )
        else:
            report_failure(
                native_datetime_utc_now(),
                state,
                trade,
                stop_on_execution_failure,
            )
