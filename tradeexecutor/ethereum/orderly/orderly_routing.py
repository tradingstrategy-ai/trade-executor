"""Route trades for Orderly vault deposits and withdrawals."""

import logging
from decimal import Decimal
from typing import Dict, cast

from eth_typing import HexAddress
from web3 import Web3

from eth_defi.orderly.vault import OrderlyVault, deposit, withdraw
from eth_defi.token import fetch_erc20_details

from tradeexecutor.ethereum.swap import get_swap_transactions, report_failure
from tradeexecutor.state.blockhain_transaction import BlockchainTransaction
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.types import JSONHexAddress
from tradeexecutor.strategy.routing import RoutingState, RoutingModel
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.utils.blockchain import get_block_timestamp

from tradeexecutor.ethereum.tx import TransactionBuilder
from tradeexecutor.state.identifier import AssetIdentifier
from tradingstrategy.pair import PandasPairUniverse

from tradeexecutor.strategy.universe_model import StrategyExecutionUniverse


logger = logging.getLogger(__name__)


class OrderlyRoutingState(RoutingState):
    """Capture trade executor state what we need for one strategy cycle of Orderly vault deposits and withdrawals.

    - Manages vault operations for entering and exiting Orderly positions
    - Trading happens externally to trade-executor on Orderly
    """

    def __init__(
        self,
        tx_builder: TransactionBuilder,
        strategy_universe: TradingStrategyUniverse,
        vault: OrderlyVault,
        broker_id: str,
        orderly_account_id: str,
    ):
        self.tx_builder = tx_builder
        self.strategy_universe = strategy_universe
        self.vault = vault
        self.broker_id = broker_id
        self.orderly_account_id = orderly_account_id

    def get_reserve_asset(self) -> AssetIdentifier:
        return self.strategy_universe.get_reserve_asset()


class OrderlyRouting(RoutingModel):
    """Orderly vault routing.

    - Handle deposits to and withdrawals from Orderly vault
    - Actual trading happens externally on Orderly
    """

    def __init__(
        self,
        reserve_token_address: JSONHexAddress,
        vault: OrderlyVault,
        broker_id: str,
        orderly_account_id: str,
        epsilon=Decimal(1e-6),
    ):
        super().__init__(
            allowed_intermediary_pairs={},
            reserve_token_address=reserve_token_address.lower(),
        )
        self.vault = vault
        self.broker_id = broker_id
        self.orderly_account_id = orderly_account_id
        self.epsilon = epsilon

        # Orderly vault operations typically need more gas
        self.vault_interaction_gas_limit = 2_000_000

    def create_routing_state(
        self,
        universe: StrategyExecutionUniverse,
        execution_details: dict
    ) -> OrderlyRoutingState:
        return OrderlyRoutingState(
            tx_builder=execution_details["tx_builder"],
            strategy_universe=cast(TradingStrategyUniverse, universe),
            vault=self.vault,
            broker_id=self.broker_id,
            orderly_account_id=self.orderly_account_id,
        )

    def perform_preflight_checks_and_logging(self,
        pair_universe: PandasPairUniverse):
        """"Checks the integrity of the routing.

        - Called from check-wallet to see our routing and balances are good
        """
        logger.info("Orderly routing details")
        logger.info("  Vault address: %s", self.vault.address)
        logger.info("  Broker ID: %s", self.broker_id)
        logger.info("  Account ID: %s", self.orderly_account_id)
        # TODO: Add reserve asset logging

    def deposit_or_withdraw(
        self,
        state: State,
        routing_state: OrderlyRoutingState,
        trade: TradeExecution,
    ) -> list[BlockchainTransaction]:
        """Prepare Orderly vault flow transactions."""

        assert isinstance(state, State)
        assert isinstance(routing_state, OrderlyRoutingState)

        assert trade.is_vault(), "Orderly routing only supports vault trades"
        assert trade.slippage_tolerance, "TradeExecution.slippage_tolerance must be set"

        reserve_asset = routing_state.strategy_universe.get_reserve_asset()

        tx_builder = routing_state.tx_builder
        web3 = tx_builder.web3
        address = tx_builder.get_token_delivery_address()

        if trade.is_buy():
            # Deposit to Orderly vault
            token_in = reserve_asset
            token_out = trade.pair.base
            amount = int(trade.get_planned_reserve())

            # Get USDC token contract for deposit
            token_contract = fetch_erc20_details(web3, reserve_asset.checksum_address).contract

            approve_fn, get_deposit_fee_fn, deposit_fn = deposit(
                vault=routing_state.vault,
                token=token_contract,
                amount=amount,
                depositor_address=address,
                orderly_account_id=Web3.to_checksum_address(routing_state.orderly_account_id),
                broker_id=routing_state.broker_id,
                token_id="USDC",  # TODO: Make this configurable based on the token
            )

            logger.info(
                "Preparing Orderly vault deposit %s -> %s, amount %s (%s)",
                token_in.token_symbol,
                token_out.token_symbol,
                amount,
                token_in.convert_to_decimal(amount),
            )

            # Get deposit fee for the transaction
            # deposit_fee = get_deposit_fee_fn.call()  # TODO: Use for transaction value

            approve_gas_limit = 200_000
            deposit_gas_limit = self.vault_interaction_gas_limit

            # Create approve transaction
            tx_1 = tx_builder.sign_transaction(
                contract=token_contract,
                args_bound_func=approve_fn,
                gas_limit=approve_gas_limit,
                asset_deltas=[],
                notes=trade.notes or "",
            )

            # Create deposit transaction with native token fee
            tx_2 = tx_builder.sign_transaction(
                contract=routing_state.vault.contract,
                args_bound_func=deposit_fn,
                gas_limit=deposit_gas_limit,
                asset_deltas=[],
                notes=trade.notes or "",
            )

            return [tx_1, tx_2]

        else:
            # Withdraw from Orderly vault
            token_in = trade.pair.base
            token_out = reserve_asset
            amount = int(-trade.planned_quantity)

            # Get USDC token contract for withdrawal
            token_contract = fetch_erc20_details(web3, reserve_asset.checksum_address).contract

            approve_fn, get_withdraw_fee_fn, withdraw_fn = withdraw(
                vault=routing_state.vault,
                token=token_contract,
                amount=amount,
                wallet_address=address,
                orderly_account_id=Web3.to_checksum_address(routing_state.orderly_account_id),
                broker_id=routing_state.broker_id,
                token_id="USDC",  # TODO: Make this configurable based on the token
            )

            logger.info(
                "Preparing Orderly vault withdraw %s -> %s, amount %s",
                token_in.token_symbol,
                token_out.token_symbol,
                amount,
            )

            # Get withdraw fee for the transaction
            # withdraw_fee = get_withdraw_fee_fn.call()  # TODO: Use for transaction value

            approve_gas_limit = 200_000
            withdraw_gas_limit = self.vault_interaction_gas_limit

            # Create approve transaction (if needed)
            tx_1 = tx_builder.sign_transaction(
                contract=token_contract,
                args_bound_func=approve_fn,
                gas_limit=approve_gas_limit,
                asset_deltas=[],
                notes=trade.notes or "",
            )

            # Create withdraw transaction with native token fee
            tx_2 = tx_builder.sign_transaction(
                contract=routing_state.vault.contract,
                args_bound_func=withdraw_fn,
                gas_limit=withdraw_gas_limit,
                asset_deltas=[],
                notes=trade.notes or "",
            )

            return [tx_1, tx_2]

    def setup_trades(
        self,
        state: State,
        routing_state: OrderlyRoutingState,
        trades: list[TradeExecution],
        check_balances=False,
        rebroadcast=False,
    ):
        """Setup Orderly vault trades for deposit/withdraw operations."""

        logger.info(
            "Preparing %d trades for Orderly vault execution",
            len(trades),
        )

        for trade in trades:
            assert trade.is_vault(), f"Not a vault trade: {trade}"
            trade.blockchain_transactions = self.deposit_or_withdraw(state, routing_state, trade)

    def settle_trade(
        self,
        web3: Web3,
        state: State,
        trade: TradeExecution,
        receipts: Dict[str, dict],
        stop_on_execution_failure=False,
    ):
        """Settle Orderly vault trade by analyzing transaction receipts."""

        logger.info(f"Settling Orderly vault trade: #{trade.trade_id}")

        base_token_details = fetch_erc20_details(web3, trade.pair.base.checksum_address)
        reserve = trade.reserve_currency

        swap_tx = get_swap_transactions(trade)

        try:
            receipt = receipts[swap_tx.tx_hash or ""]
        except KeyError as e:
            raise KeyError(f"Could not find hash: {swap_tx.tx_hash} in {receipts}") from e

        ts = get_block_timestamp(web3, receipt["blockNumber"])

        # For Orderly vault operations, we need to analyze the transaction logs
        # to determine the actual amounts deposited/withdrawn
        try:
            if trade.is_buy():
                # Deposit operation - analyze deposit logs
                # TODO: Implement proper Orderly vault deposit analysis
                # For now, assume the planned amounts were executed
                executed_reserve = trade.planned_reserve
                executed_amount = trade.planned_quantity
                price = executed_reserve / executed_amount if executed_amount != 0 else 1.0

            else:
                # Withdraw operation - analyze withdraw logs
                # TODO: Implement proper Orderly vault withdraw analysis
                # For now, assume the planned amounts were executed
                executed_amount = trade.planned_quantity
                executed_reserve = trade.planned_reserve
                price = -executed_reserve / executed_amount if executed_amount != 0 else 1.0

            logger.info("Executed amount: %s, executed reserve: %s, price: %s", executed_amount, executed_reserve, price)

            # Mark as success
            state.mark_trade_success(
                ts,
                trade,
                executed_price=float(price),
                executed_amount=executed_amount,
                executed_reserve=executed_reserve,
                lp_fees=0,
                native_token_price=0,  # Not applicable for vault operations
                cost_of_gas=0,  # TODO: Calculate from receipt
            )

            slippage = trade.get_slippage()
            logger.info(f"Executed: {executed_amount} {trade.pair.base.token_symbol}, {executed_reserve} {trade.pair.quote.token_symbol}, price: {trade.executed_price}, expected reserve: {trade.planned_reserve} {trade.pair.quote.token_symbol}, slippage {slippage:.2%}")

        except Exception as e:
            logger.error(f"Failed to settle Orderly vault trade {trade.trade_id}: {e}")
            report_failure(ts, state, trade, stop_on_execution_failure)