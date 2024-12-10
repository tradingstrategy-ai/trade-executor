"""Perform spot token swaps for Velvet vault using Enso's intent engine."""

import logging

from eth_defi.velvet import VelvetVault
from tradeexecutor.ethereum.velvet.tx import VelvetTransactionBuilder
from tradeexecutor.state.blockhain_transaction import BlockchainTransaction
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.types import JSONHexAddress
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
        self.tx_builder = tx_builder


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
        remaining_tokens: set[JSONHexAddress],
    ) -> BlockchainTransaction:
        """Prepare swap payload from Velvet centralised API."""
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
            swap_amount=trade.get_raw_planned_quantity(),
            slippage=trade.slippage_tolerance,
            remaining_tokens=remaining_tokens,
            swap_all=trade.closing,
        )
        blockchain_transaction = tx_builder.sign_transaction_data(
            tx_data,
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

        for trade in trades:
            assert trade.is_spot(), "Velvet only supports spot trades"

        # Calculate what tokens we will have after this trade batch is complete
        remaining_tokens = {t.pair.base.address for t in trades if not t.closing}

        logger.info(
            "Preparing %s trades for Enso execution, we will have %d tokens remaining",
            len(trades),
            len(remaining_tokens),
        )

        for t in trades:
            t.blockchain_transactions = [self.swap(state, t, remaining_tokens)]
