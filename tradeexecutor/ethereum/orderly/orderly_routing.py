"""Route trades for Orderly vault deposits and withdrawals."""

import logging
from decimal import Decimal
from typing import Dict, cast

from eth_typing import HexAddress
from web3 import Web3

from eth_defi.orderly.vault import deposit, withdraw
from eth_defi.token import fetch_erc20_details
from eth_defi.trade import TradeSuccess

from tradeexecutor.ethereum.orderly.orderly_vault import OrderlyVault
from tradeexecutor.ethereum.orderly.orderly_analysis import analyse_orderly_flow_transaction
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


def get_orderly_vault_for_pair(
    web3: Web3,
    target_pair: "TradingPairIdentifier",
    routing_vault: OrderlyVault,
) -> OrderlyVault:
    """Get OrderlyVault instance for a trading pair.

    Since Orderly vault is already configured in routing, just return it.
    This maintains consistency with the ERC-4626 vault pattern.

    :param web3:
        Web3 connection

    :param target_pair:
        Trading pair

    :param routing_vault:
        Vault from routing model

    :return:
        OrderlyVault instance
    """
    assert target_pair.is_vault(), f"Not a vault pair: {target_pair}"
    return routing_vault


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
        token_id_mapping: dict[str, str] | None = None,
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

        # Token symbol to Orderly token_id mapping
        # Example: {"USDC": "USDC", "WETH": "WETH"}
        self.token_id_mapping = token_id_mapping or {}

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

    def _get_orderly_token_id(self, token_symbol: str) -> str:
        """Get Orderly token_id for a given token symbol.

        :param token_symbol:
            Token symbol like "USDC", "WETH"

        :return:
            Orderly token_id, defaults to the token_symbol if not in mapping
        """
        return self.token_id_mapping.get(token_symbol, token_symbol)

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

            # Get token contract for deposit
            token_contract = fetch_erc20_details(web3, reserve_asset.checksum_address).contract

            # Get Orderly token_id for this token
            token_id = self._get_orderly_token_id(token_in.token_symbol)

            approve_fn, get_deposit_fee_fn, deposit_fn = deposit(
                vault=routing_state.vault,
                token=token_contract,
                amount=amount,
                depositor_address=address,
                orderly_account_id=Web3.to_checksum_address(routing_state.orderly_account_id),
                broker_id=routing_state.broker_id,
                token_id=token_id,
            )

            logger.info(
                "Preparing Orderly vault deposit %s -> %s, amount %s (%s), token_id %s",
                token_in.token_symbol,
                token_out.token_symbol,
                amount,
                token_in.convert_to_decimal(amount),
                token_id,
            )

            # Query deposit fee from contract
            try:
                deposit_fee = get_deposit_fee_fn.call()
                logger.info("Orderly deposit fee: %s wei", deposit_fee)
            except Exception as e:
                logger.warning("Could not query deposit fee: %s", e)
                deposit_fee = 0

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

            # Get token contract for withdrawal
            token_contract = fetch_erc20_details(web3, reserve_asset.checksum_address).contract

            # Get Orderly token_id for this token
            token_id = self._get_orderly_token_id(token_out.token_symbol)

            approve_fn, get_withdraw_fee_fn, withdraw_fn = withdraw(
                vault=routing_state.vault,
                token=token_contract,
                amount=amount,
                wallet_address=address,
                orderly_account_id=Web3.to_checksum_address(routing_state.orderly_account_id),
                broker_id=routing_state.broker_id,
                token_id=token_id,
            )

            logger.info(
                "Preparing Orderly vault withdraw %s -> %s, amount %s, token_id %s",
                token_in.token_symbol,
                token_out.token_symbol,
                amount,
                token_id,
            )

            # Query withdraw fee from contract
            try:
                withdraw_fee = get_withdraw_fee_fn.call()
                logger.info("Orderly withdraw fee: %s wei", withdraw_fee)
            except Exception as e:
                logger.warning("Could not query withdraw fee: %s", e)
                withdraw_fee = 0

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

        # Get vault instance for analysis
        vault = get_orderly_vault_for_pair(web3, trade.pair, self.vault)

        swap_tx = get_swap_transactions(trade)

        try:
            receipt = receipts[swap_tx.tx_hash or ""]
        except KeyError as e:
            raise KeyError(f"Could not find hash: {swap_tx.tx_hash} in {receipts}") from e

        ts = get_block_timestamp(web3, receipt["blockNumber"])

        # Determine direction
        direction = "deposit" if trade.is_buy() else "withdraw"

        # Analyze transaction logs to get actual amounts
        try:
            result = analyse_orderly_flow_transaction(
                vault=vault,
                tx_hash=swap_tx.tx_hash,
                tx_receipt=receipt,
                direction=direction,
                hot_wallet=False,
            )
        except Exception as e:
            # If analysis fails, log warning and fall back to planned amounts
            logger.warning(
                f"Could not analyze Orderly vault transaction {swap_tx.tx_hash}: {e}. "
                f"Falling back to planned amounts."
            )
            # Use planned amounts as fallback
            if trade.is_buy():
                executed_reserve = trade.planned_reserve
                executed_amount = trade.planned_quantity
                price = executed_reserve / executed_amount if executed_amount != 0 else 1.0
            else:
                executed_amount = trade.planned_quantity
                executed_reserve = trade.planned_reserve
                price = -executed_reserve / executed_amount if executed_amount != 0 else 1.0

            state.mark_trade_success(
                ts,
                trade,
                executed_price=float(price),
                executed_amount=executed_amount,
                executed_reserve=executed_reserve,
                lp_fees=0,
                native_token_price=0,
                cost_of_gas=0,
            )
            return

        # Analysis succeeded - extract actual amounts
        if isinstance(result, TradeSuccess):
            reserve_asset = trade.reserve_currency
            base_token_details = fetch_erc20_details(web3, trade.pair.base.checksum_address)

            if trade.is_buy():
                # Deposit: USDC in -> vault shares out
                executed_reserve = result.amount_in / Decimal(10 ** reserve_asset.decimals)
                executed_amount = result.amount_out / Decimal(10 ** base_token_details.decimals)
                price = executed_reserve / executed_amount if executed_amount != 0 else 1.0
            else:
                # Withdraw: vault shares in -> USDC out
                executed_amount = -result.amount_in / Decimal(10 ** base_token_details.decimals)
                executed_reserve = result.amount_out / Decimal(10 ** reserve_asset.decimals)
                price = -executed_reserve / executed_amount if executed_amount != 0 else 1.0

            logger.info(
                "Executed amount: %s, executed reserve: %s, price: %s",
                executed_amount,
                executed_reserve,
                price
            )

            # Mark as success with actual amounts
            state.mark_trade_success(
                ts,
                trade,
                executed_price=float(price),
                executed_amount=executed_amount,
                executed_reserve=executed_reserve,
                lp_fees=result.lp_fee_paid,
                native_token_price=0,  # Not applicable for vault operations
                cost_of_gas=float(result.get_cost_of_gas()),
            )

            slippage = trade.get_slippage()
            logger.info(
                f"Executed: {executed_amount} {trade.pair.base.token_symbol}, "
                f"{executed_reserve} {trade.pair.quote.token_symbol}, "
                f"price: {trade.executed_price}, "
                f"expected reserve: {trade.planned_reserve} {trade.pair.quote.token_symbol}, "
                f"slippage {slippage:.2%}"
            )

        else:
            # Trade failed - report failure
            report_failure(ts, state, trade, stop_on_execution_failure)
