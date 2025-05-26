"""Perform spot token swaps for Velvet vault using Enso's intent engine."""

import logging
from decimal import Decimal
from typing import cast, Dict

from hexbytes import HexBytes

from eth_defi.timestamp import get_block_timestamp
from eth_defi.trade import TradeSuccess
from eth_defi.velvet import VelvetVault
from eth_defi.velvet.analysis import analyse_trade_by_receipt_generic
from eth_defi.velvet.enso import VelvetSwapError
from tradeexecutor.ethereum.swap import report_failure
from tradeexecutor.ethereum.velvet.tx import VelvetTransactionBuilder
from tradeexecutor.state.blockhain_transaction import BlockchainTransaction
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.types import JSONHexAddress
from tradeexecutor.strategy.routing import RoutingState, RoutingModel, PositionAvailabilityResponse
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.strategy.universe_model import StrategyExecutionUniverse
from tradingstrategy.pair import PandasPairUniverse

logger = logging.getLogger(__name__)


class SwapCheckFailed(Exception):
    """Enso check for tradeabiltiy failed"""


class VelvetEnsoRoutingState(RoutingState):
    """Capture trade executor state what we need for one strategy cycle of Enso routing.

    - Not much to do here - Enso swaps are stateless (no approves needed)
    """
    def __init__(
        self,
        vault: VelvetVault,
        tx_builder: VelvetTransactionBuilder,
        strategy_universe: TradingStrategyUniverse,
    ):
        self.vault = vault
        self.tx_builder = tx_builder
        self.strategy_universe = strategy_universe

    def get_reserve_asset(self) -> AssetIdentifier:
        return self.strategy_universe.get_reserve_asset()


class VelvetEnsoRouting(RoutingModel):
    """Use Velvet's Enso integration for performing trades.

    .. note ::

        We use routing parameters ``reserve_token_address`` and ``allowed_intermediary_pairs``
        only in safety trip asserts, as Enso takes care of all routing.

    """

    def create_routing_state(
        self,
        universe: StrategyExecutionUniverse,
        execution_details: dict
    ) -> VelvetEnsoRoutingState:
        return VelvetEnsoRoutingState(
            vault=execution_details["vault"],
            tx_builder=execution_details["tx_builder"],
            strategy_universe=cast(TradingStrategyUniverse, universe),
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
        remaining_tokens: list[JSONHexAddress],
    ) -> BlockchainTransaction:
        """Prepare swap payload from Velvet centralised API."""

        assert trade.is_spot(), "Velvet only supports spot trades"
        assert trade.slippage_tolerance, "TradeExecution.slippage_tolerance must be set with Velvet"
        assert trade.pair.quote.address in self.allowed_intermediary_pairs or trade.pair.quote.address == self.reserve_token_address, f"Unsupported quote token: {trade.pair}"

        # Enso does routing for as, we only care about USDC and the target token
        reserve_asset = state.strategy_universe.get_reserve_asset()
        if trade.is_buy():
            token_in = reserve_asset
            token_out = trade.pair.base
            swap_amount = trade.get_raw_planned_reserve()
        else:
            token_in = trade.pair.base
            token_out = reserve_asset
            swap_amount = -trade.get_raw_planned_quantity()

        vault = state.vault
        tx_builder = state.tx_builder

        logger.info(
            "Preparing Velvet swap %s -> %s, amount %s (%s), slippage tolerance %f",
            token_in.token_symbol,
            token_out.token_symbol,
            swap_amount,
            token_in.convert_to_decimal(swap_amount),
            trade.slippage_tolerance,
        )

        try:
            tx_data = vault.prepare_swap_with_intent(
                token_in=token_in.address,
                token_out=token_out.address,
                swap_amount=swap_amount,
                slippage=trade.slippage_tolerance,
                remaining_tokens=remaining_tokens,
                swap_all=trade.closing,
                manage_token_list=False,
            )
        except Exception as e:
            raise RuntimeError(f"Could not perform trade {trade} on Enso") from e
        blockchain_transaction = tx_builder.sign_transaction_data(
            tx_data,
            notes=trade.notes,
        )
        return blockchain_transaction

    def setup_trades(
        self,
        state: State,
        routing_state: VelvetEnsoRoutingState,
        trades: list[TradeExecution],
        check_balances=False,
        rebroadcast=False,
    ):
        """
        See test_velvet_e2e for tests.

        Error codes:

        - Revert reason: execution reverted: custom error 0xe2f23246

        - 2Po: Enso slippage error, or out of funds
        """

        logger.info(
            "Preparing %s trades for Enso execution",
            len(trades),
        )

        for trade in trades:
            assert trade.is_spot(), f"Velvet only supports spot trades, got {trade}"

        # Calculate what tokens we will have after this trade batch is complete
        remaining_tokens = {p.pair.base for p in state.portfolio.get_open_and_frozen_positions() if p.get_quantity(planned=False) > 0}
        remaining_tokens.add(routing_state.get_reserve_asset())

        for t in trades:
            if t.closing:
                remaining_tokens.discard(t.pair.base)
            elif t.is_buy():
                remaining_tokens.add(t.pair.base)

            logger.info(
                "Preparing trade %s, remaining tokens %s",
                t,
                ", ".join(t.token_symbol for t in remaining_tokens),

            )
            remaining_token_addresses = [t.address for t in remaining_tokens]
            t.blockchain_transactions = [self.swap(routing_state, t, remaining_token_addresses)]

    def settle_trade(
        self,
        web3,
        state: State,
        trade: TradeExecution,
        receipts: Dict[HexBytes, dict],
        stop_on_execution_failure=False,
    ):
        """Post-trade executed price analysis.

        - Read on-chain data about the tx receipt of Enso swap

        :param state:
            Strategy state

        :param web3:
            Web3 connection.

        :param trade:
            Trade executed in this execution batch

        :param receipts:
            Blockchain receipts we received in this execution batch.

            Hash -> receipt mapping.

        :param stop_on_execution_failure:
            Raise an error if the trade failed.

            Used in unit testing.

        """

        assert trade.is_spot()
        assert len(trade.blockchain_transactions) == 1, "Enso trade can have only a single physical tx per trade"

        tx = trade.blockchain_transactions[0]
        tx_hash = tx.tx_hash
        receipt = receipts[HexBytes(tx_hash)]

        result = analyse_trade_by_receipt_generic(
            web3,
            tx_hash=tx_hash,
            tx_receipt=receipt,
            intent_based=True,
        )

        ts = get_block_timestamp(web3, receipt["blockNumber"])

        base = trade.pair.base
        quote = trade.pair.quote
        reserve, rate = state.portfolio.get_default_reserve_asset()

        if isinstance(result, TradeSuccess):
            if trade.is_buy():
                executed_reserve = reserve.convert_to_decimal(result.amount_in)
                executed_amount = base.convert_to_decimal(result.amount_out)
                price = result.get_human_price(reverse_token_order=True)
            else:
                executed_amount = -base.convert_to_decimal(result.amount_in)
                executed_reserve = reserve.convert_to_decimal(result.amount_out)
                price = result.get_human_price(reverse_token_order=False)

            logger.info(f"Executed: {executed_amount} {trade.pair.base.token_symbol} - {trade.pair.quote.token_symbol}, using reserve {executed_reserve} {reserve.token_symbol}")

            lp_fee_paid = result.lp_fee_paid  # Always set to None

            # Mark as success
            state.mark_trade_success(
                ts,
                trade,
                executed_price=float(price),
                executed_amount=executed_amount,
                executed_reserve=executed_reserve,
                lp_fees=lp_fee_paid,
                native_token_price=0,
                cost_of_gas=float(result.get_cost_of_gas()),
            )
        else:
            # Trade failed
            report_failure(ts, state, trade, stop_on_execution_failure)

    def check_enter_position(
        self,
        routing_state: RoutingState,
        trading_pair: TradingPairIdentifier,
        reserve_amount_to_test=Decimal(1),
    ) -> PositionAvailabilityResponse:

        assert isinstance(routing_state, VelvetEnsoRoutingState)

        strategy_universe = routing_state.strategy_universe
        token_in = strategy_universe.get_reserve_asset()
        token_out = trading_pair.base
        swap_amount = token_in.convert_to_raw_amount(reserve_amount_to_test)
        slippage_tolerance = 0.01

        buy_tax =  trading_pair.get_buy_tax()
        sell_tax = trading_pair.get_sell_tax()
        risk_score = trading_pair.get_risk_score()
        metadata = trading_pair.get_token_metadata()
        tokensniffer_data  = trading_pair.get_minimal_tokesniffer_data()

        logger.info(
            "Velvet tradeability check %s -> %s, amount %s (%s), slippage tolerance: %f, buy tax: %s, sell tax: %s, risk score: %s, metadata is %s, tokensniffer data is %s",
            token_in.token_symbol,
            token_out.token_symbol,
            swap_amount,
            slippage_tolerance,
            token_in.convert_to_decimal(swap_amount),
            buy_tax,
            sell_tax,
            risk_score,
            metadata is not None,
            tokensniffer_data,
        )

        vault = routing_state.vault

        try:
            tx_data = vault.prepare_swap_with_intent(
                token_in=token_in.address,
                token_out=token_out.address,
                swap_amount=swap_amount,
                slippage=slippage_tolerance,
                remaining_tokens=[token_in.address],
                swap_all=False,
                manage_token_list=False,
                # Do not retry because we want to handle HTTP 500
                retries=1,
            )
            logger.info(
                "Velvet trade allowed %s: check success, %s",
                token_out.token_symbol,
                tx_data,
            )
            return PositionAvailabilityResponse(
                pair=trading_pair,
                supported=True,
                error_message=None
            )
        except VelvetSwapError as e:
            # "https://intents.velvet.capital/api/v1, code 500: {"message":"Could not quote shortcuts for route 0x55d398326f99059ff775485246999027b3197955 -> 0x75da2849aab9a7970be603449a5b4b47efe61178 on network 56, please make sure your amountIn (8180838235143463421) is within an acceptable range","description":"failed enso request"}"
            marker_string = "Could not quote shortcuts for route"
            if marker_string in str(e):
                logger.warning(
                    "Velvet trade not allowed %s: failed",
                    token_out.token_symbol,
                )
                return PositionAvailabilityResponse(
                    pair=trading_pair,
                    supported=False,
                    error_message=str(e)
                )
            else:
                raise SwapCheckFailed(f"Unknown error when checking tradebility for pair: {trading_pair}") from e
