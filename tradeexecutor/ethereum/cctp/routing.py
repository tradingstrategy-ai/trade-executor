"""CCTP bridge routing model.

Handles routing of CCTP bridge trades — creates burn transactions
on the source chain and receive transactions on the destination chain.
"""

import logging
from typing import List

from tradeexecutor.ethereum.web3config import Web3Config
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.routing import RoutingModel, RoutingState

logger = logging.getLogger(__name__)


class CctpBridgeRoutingState(RoutingState):
    """Tracks CCTP bridge transactions for the current cycle."""

    def __init__(self, universe=None):
        self.universe = universe


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
        return CctpBridgeRoutingState(universe)

    def setup_trades(
        self,
        state: State,
        routing_state: RoutingState,
        trades: List[TradeExecution],
        check_balances=False,
        rebroadcast=False,
    ):
        """Set up CCTP bridge trades.

        For each bridge trade:

        1. Get the source chain web3 connection
        2. Create approve transaction for CCTP TokenMessenger
        3. Create depositForBurn transaction
        4. Attach both as BlockchainTransactions on the trade

        The actual blockchain transaction creation uses ``eth_defi.cctp.transfer``
        functions.
        """
        from eth_defi.cctp.transfer import (
            prepare_approve_for_burn,
            prepare_deposit_for_burn,
        )
        from tradingstrategy.chain import ChainId

        for trade in trades:
            assert trade.pair.is_cctp_bridge(), \
                f"CctpBridgeRouting received non-bridge trade: {trade}"

            pair = trade.pair
            source_chain_id = pair.source_chain_id
            dest_chain_id = pair.destination_chain_id

            source_web3 = self.web3config.get_connection(ChainId(source_chain_id))
            assert source_web3 is not None, \
                f"No web3 connection for source chain {source_chain_id}"

            amount_raw = int(trade.planned_quantity * (10 ** pair.quote.decimals))

            # Get the token storage address (wallet/safe) that holds USDC
            token_storage = state.sync.deployment.address

            # Create approve tx
            approve_fn = prepare_approve_for_burn(source_web3, amount=amount_raw)

            # Create burn tx
            burn_fn = prepare_deposit_for_burn(
                source_web3,
                amount=amount_raw,
                destination_chain_id=dest_chain_id,
                mint_recipient=token_storage,
            )

            trade.set_blockchain_transactions([])

            logger.info(
                "CCTP bridge trade %s: burn %s on chain %d -> chain %d",
                trade.get_short_label(),
                trade.planned_quantity,
                source_chain_id,
                dest_chain_id,
            )
