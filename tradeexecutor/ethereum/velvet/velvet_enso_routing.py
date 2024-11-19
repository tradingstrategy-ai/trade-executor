"""Perform spot token swaps for Velvet vault using Enso's intent engine."""

import logging

from eth_defi.velvet import VelvetVault
from tradeexecutor.ethereum.velvet.tx import VelvetTransactionBuilder
from tradeexecutor.state.blockhain_transaction import BlockchainTransaction
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.routing import RoutingState, RoutingModel
from tradeexecutor.strategy.universe_model import StrategyExecutionUniverse
from tradingstrategy.pair import PandasPairUniverse

logger = logging.getLogger(__name__)


class VelvetEnsoRoutingState(RoutingState):

    def __init__(
        self,
        vault: VelvetVault,
        tx_builder: VelvetTransactionBuilder,
    ):
        self.vault = vault
        self.tx_builder = tx_builder()


class VelvetEnsoRouting(RoutingModel):
    """Use Velvet's Enso integration for performing trades.
    """

    def __init__(
        self,
    ):
        pass

    def create_routing_state(
        self,
        universe: StrategyExecutionUniverse,
        execution_details: dict
    ) -> VelvetEnsoRoutingState:
        return VelvetEnsoRoutingState(
            vault=execution_details["vault"],
            tx_builder=execution_details["tx_builder"],
        )

    def perform_preflight_checks_and_logging(
        self,
        pair_universe: PandasPairUniverse
    ):
        logger.info("Routing details")
        logger.info("  - No information displayed yet about Enso routing")

    def swap(
        self,
        state: VelvetEnsoRoutingState,
        trade: TradeExecution,
    ) -> BlockchainTransaction:
        assert trade.is_spot(), "Velvet only supports spot trades"

        assert trade.slippage_tolerance, "TradeExecution.slippage_tolerance must be set with Velvet"

        if trade.is_buy():
            token_in = trade.pair.quote
            token_out = trade.pair.base
        else:
            token_in = trade.pair.base
            token_out = trade.pair.quote

        vault = state.vault
        tx_builder = state.tx_builder
        tx_data = vault.prepare_swap_with_enso(
            token_in=token_in.address,
            token_out=token_out.address,
            slippage=trade.slippage_tolerance,
            remaining_tokens=set(),
            swap_all=trade.closing,
        )
        blockchain_transaction = tx_builder.sign_transaction(
            tx_data=tx_data,
            notes=trade.notes,
        )
        return blockchain_transaction

    def setup_trades(
        self,
        state: VelvetEnsoRoutingState,
        trades: list[TradeExecution],
        check_balances=False,
        rebroadcast=False,
    ):
        for t in trades:
            t.blockchain_transactions = [self.swap(t)]

