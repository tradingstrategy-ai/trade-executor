"""Route trades for Hypercore native vaults on HyperEVM.

Hypercore vault deposits/withdrawals use a multi-phase flow:

**Deposit (buy)**:

0. Activation (if needed): performed synchronously via
   :py:func:`~eth_defi.hyperliquid.evm_escrow.activate_account` in ``setup_trades()``
1. Phase 1: ``approve`` + ``CDW.deposit`` — bridge USDC to HyperCore spot
2. Escrow wait: poll ``spotClearinghouseState`` until USDC clears
3. Phase 2: ``transferUsdClass`` + ``vaultTransfer`` — move to perp, deposit into vault

If the Safe is already activated, step 0 is skipped.

**Withdrawal (sell)**:

A single multicall transaction (``vaultTransfer`` + ``transferUsdClass`` + ``spotSend``)
is created in ``setup_trades`` and broadcast normally.

The build functions from :py:mod:`eth_defi.hyperliquid.core_writer` return
``module.functions.multicall(calls)`` already wrapped through
``TradingStrategyModuleV0.performCall()``.  Using
:py:class:`~tradeexecutor.ethereum.lagoon.tx.LagoonTransactionBuilder`
would double-wrap them, so we sign directly via :py:class:`~eth_defi.hotwallet.HotWallet`.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Dict, TYPE_CHECKING, cast

from hexbytes import HexBytes
from web3 import Web3
from web3.contract.contract import ContractFunction

from eth_defi.gas import apply_gas
from eth_defi.hotwallet import HotWallet
from eth_defi.hyperliquid.api import fetch_user_vault_equity
from eth_defi.hyperliquid.core_writer import (
    build_hypercore_deposit_phase1,
    build_hypercore_deposit_phase2,
    build_hypercore_withdraw_multicall,
)
from eth_defi.hyperliquid.evm_escrow import (
    DEFAULT_ACTIVATION_AMOUNT,
    activate_account,
    is_account_activated,
    wait_for_evm_escrow_clear,
)
from eth_defi.hyperliquid.session import (
    HYPERLIQUID_API_URL,
    HYPERLIQUID_TESTNET_API_URL,
    HyperliquidSession,
    create_hyperliquid_session,
)

from tradeexecutor.ethereum.swap import report_failure
from tradeexecutor.state.blockhain_transaction import (
    BlockchainTransaction,
    BlockchainTransactionType,
    JSONAssetDelta,
)
from tradeexecutor.state.pickle_over_json import encode_pickle_over_json
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.types import JSONHexAddress
from tradeexecutor.strategy.routing import RoutingState, RoutingModel
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.strategy.universe_model import StrategyExecutionUniverse
from tradeexecutor.utils.blockchain import get_block_timestamp
from tradeexecutor.utils.hex import hexbytes_to_hex_str

from tradeexecutor.ethereum.tx import TransactionBuilder
from tradeexecutor.state.identifier import TradingPairIdentifier, AssetIdentifier
from tradingstrategy.pair import PandasPairUniverse

if TYPE_CHECKING:
    from eth_defi.erc_4626.vault_protocol.lagoon.vault import LagoonVault

logger = logging.getLogger(__name__)

#: Gas limit for Hypercore multicall transactions (approve + CDW.deposit or CoreWriter actions).
HYPERCORE_MULTICALL_GAS = 500_000


class HypercoreVaultRoutingState(RoutingState):
    """Routing state for Hypercore vault trades.

    Captures the transaction builder and strategy universe
    needed during a single strategy cycle.
    """

    def __init__(
        self,
        tx_builder: TransactionBuilder,
        strategy_universe: TradingStrategyUniverse,
    ):
        self.tx_builder = tx_builder
        self.strategy_universe = strategy_universe

    def get_reserve_asset(self) -> AssetIdentifier:
        return self.strategy_universe.get_reserve_asset()


class HypercoreVaultRouting(RoutingModel):
    """Route Hypercore vault deposit/withdrawal trades.

    Creates :py:class:`~tradeexecutor.state.blockhain_transaction.BlockchainTransaction`
    objects for Hypercore native vault operations via the Lagoon vault's
    ``TradingStrategyModuleV0`` multicall interface.

    The build functions from :py:mod:`eth_defi.hyperliquid.core_writer` return
    ready-to-sign ``ContractFunction`` objects (``module.functions.multicall(...)``).
    These are signed directly with the deployer :py:class:`~eth_defi.hotwallet.HotWallet`
    rather than going through ``LagoonTransactionBuilder.sign_transaction()``
    which would double-wrap them.
    """

    def __init__(
        self,
        web3: Web3,
        lagoon_vault: LagoonVault,
        deployer: HotWallet,
        reserve_token_address: str,
    ):
        """
        :param web3:
            HyperEVM Web3 connection.

        :param lagoon_vault:
            Lagoon vault instance with ``trading_strategy_module_address`` configured.

        :param deployer:
            Hot wallet for the asset manager / deployer EOA.

        :param reserve_token_address:
            USDC token address (lowercase).
        """
        super().__init__(
            allowed_intermediary_pairs={},
            reserve_token_address=reserve_token_address,
        )
        self.web3 = web3
        self.lagoon_vault = lagoon_vault
        self.deployer = deployer
        self.chain_id = web3.eth.chain_id
        self.is_testnet = self.chain_id == 998
        self._session: HyperliquidSession | None = None
        self._activation_cost_raw: int = 0

    @property
    def safe_address(self) -> str:
        return self.lagoon_vault.safe_address

    def _get_session(self) -> HyperliquidSession:
        """Lazily create a Hyperliquid API session."""
        if self._session is None:
            api_url = HYPERLIQUID_TESTNET_API_URL if self.is_testnet else HYPERLIQUID_API_URL
            self._session = create_hyperliquid_session(api_url=api_url)
        return self._session

    def create_routing_state(
        self,
        universe: StrategyExecutionUniverse,
        execution_details: dict,
    ) -> HypercoreVaultRoutingState:
        return HypercoreVaultRoutingState(
            tx_builder=execution_details["tx_builder"],
            strategy_universe=cast(TradingStrategyUniverse, universe),
        )

    def perform_preflight_checks_and_logging(
        self,
        pair_universe: PandasPairUniverse,
    ):
        logger.info("Hypercore vault routing details")
        self.reserve_asset_logging(pair_universe)

    # ------------------------------------------------------------------
    # Transaction building helpers
    # ------------------------------------------------------------------

    def _sign_multicall(
        self,
        fn: ContractFunction,
        gas_limit: int = HYPERCORE_MULTICALL_GAS,
        notes: str = "",
    ) -> BlockchainTransaction:
        """Sign a TradingStrategyModuleV0 multicall and create a BlockchainTransaction.

        The build functions from ``eth_defi.hyperliquid.core_writer`` return
        ``module.functions.multicall(calls)`` which is already addressed to the
        module contract.  We sign it directly with the deployer hot wallet.

        :param fn:
            Bound ``ContractFunction`` (``module.functions.multicall(...)``).

        :param gas_limit:
            Gas limit for the transaction.

        :param notes:
            Human-readable notes for diagnostics.

        :return:
            Signed :py:class:`BlockchainTransaction`.
        """
        tx_data = fn.build_transaction({
            "chainId": self.chain_id,
            "from": self.deployer.address,
            "gas": gas_limit,
        })

        gas_price = self.web3.eth.gas_price
        tx_data["gasPrice"] = gas_price

        signed_tx = self.deployer.sign_transaction_with_new_nonce(tx_data)
        signed_bytes = hexbytes_to_hex_str(signed_tx.rawTransaction)

        # Needed for get_swap_transactions() compatibility
        tx_data["function"] = "multicall"

        return BlockchainTransaction(
            type=BlockchainTransactionType.lagoon_vault,
            chain_id=self.chain_id,
            from_address=self.deployer.address,
            contract_address=fn.address,
            function_selector=fn.fn_name,
            transaction_args=None,
            args=None,
            wrapped_args=None,
            signed_bytes=signed_bytes,
            signed_tx_object=encode_pickle_over_json(signed_tx),
            tx_hash=hexbytes_to_hex_str(signed_tx.hash),
            nonce=signed_tx.nonce,
            details=tx_data,
            asset_deltas=[],
            notes=notes,
        )

    def _get_vault_address(self, trade: TradeExecution) -> str:
        """Extract the Hypercore vault address from the trade pair."""
        vault_addr = trade.pair.other_data.get("hypercore_vault_address")
        assert vault_addr, f"No hypercore_vault_address in pair other_data: {trade.pair}"
        return vault_addr

    def _get_raw_usdc_amount(self, trade: TradeExecution) -> int:
        """Get the raw USDC amount (6 decimals) for the trade."""
        planned_reserve = trade.get_planned_reserve()
        return int(planned_reserve * 10**6)

    def _create_deposit_or_withdraw_txs(
        self,
        trade: TradeExecution,
        activation_cost_raw: int = 0,
    ) -> list[BlockchainTransaction]:
        """Create blockchain transactions for a vault deposit or withdrawal.

        - **Buy (deposit)**: creates phase 1 tx only.
          Phase 2 is handled in :py:meth:`_settle_deposit`.
        - **Sell (withdrawal)**: creates the full withdraw multicall.

        Activation (if needed) is handled synchronously in
        :py:meth:`setup_trades` before this method is called.

        :param activation_cost_raw:
            Raw USDC consumed by activation (deducted from deposit amount).
        """
        vault_address = self._get_vault_address(trade)
        raw_amount = self._get_raw_usdc_amount(trade)

        if trade.is_buy() and activation_cost_raw > 0:
            raw_amount -= activation_cost_raw
            assert raw_amount > 0, \
                f"Deposit amount {self._get_raw_usdc_amount(trade)} raw USDC is less than " \
                f"activation cost {activation_cost_raw} raw USDC"
            logger.info(
                "Reduced deposit by %d raw USDC activation cost → %d raw USDC",
                activation_cost_raw,
                raw_amount,
            )

        if trade.is_buy():
            logger.info(
                "Building Hypercore vault deposit phase 1: %d raw USDC to vault %s",
                raw_amount,
                vault_address,
            )
            fn = build_hypercore_deposit_phase1(
                self.lagoon_vault,
                evm_usdc_amount=raw_amount,
            )
            tx = self._sign_multicall(fn, notes=f"Hypercore deposit phase 1: {raw_amount} raw USDC")
            return [tx]

        else:
            logger.info(
                "Building Hypercore vault withdrawal: %d raw USDC from vault %s",
                raw_amount,
                vault_address,
            )
            fn = build_hypercore_withdraw_multicall(
                self.lagoon_vault,
                hypercore_usdc_amount=raw_amount,
                vault_address=vault_address,
            )
            tx = self._sign_multicall(fn, notes=f"Hypercore withdrawal: {raw_amount} raw USDC")
            return [tx]

    # ------------------------------------------------------------------
    # setup_trades
    # ------------------------------------------------------------------

    def setup_trades(
        self,
        state: State,
        routing_state: HypercoreVaultRoutingState,
        trades: list[TradeExecution],
        check_balances=False,
        rebroadcast=False,
    ):
        """Create blockchain transactions for Hypercore vault trades.

        If the Safe is not yet activated on HyperCore, activation is performed
        synchronously via :py:func:`~eth_defi.hyperliquid.evm_escrow.activate_account`
        before building the deposit transaction.  After activation, phase 1
        is created here and phase 2 is deferred to :py:meth:`settle_trade`.
        """
        logger.info(
            "Preparing %d trades for Hypercore vault execution",
            len(trades),
        )

        activated = is_account_activated(self.web3, self.safe_address)
        activation_cost_raw = 0

        if activated:
            logger.info("Safe %s is activated on HyperCore", self.safe_address)
        else:
            logger.info("Safe %s is NOT activated on HyperCore — will activate first", self.safe_address)

        has_buys = any(t.is_buy() for t in trades)

        if not activated and has_buys:
            logger.info("Activating Safe %s on HyperCore before deposit...", self.safe_address)
            session = self._get_session()
            activate_account(
                web3=self.web3,
                lagoon_vault=self.lagoon_vault,
                deployer=self.deployer,
                session=session,
            )
            self.deployer.sync_nonce(self.web3)
            activation_cost_raw = DEFAULT_ACTIVATION_AMOUNT
            self._activation_cost_raw = activation_cost_raw
            logger.info(
                "Safe %s activated on HyperCore (cost %d raw USDC from Safe)",
                self.safe_address,
                activation_cost_raw,
            )

        for trade in trades:
            assert trade.is_vault(), f"Not a vault trade: {trade}"

            if not trade.is_buy() and not activated:
                raise AssertionError(
                    f"Cannot withdraw from Hypercore vault: Safe {self.safe_address} "
                    f"is not activated on HyperCore."
                )

            trade.blockchain_transactions = self._create_deposit_or_withdraw_txs(
                trade,
                activation_cost_raw=activation_cost_raw,
            )

    # ------------------------------------------------------------------
    # Settlement helpers
    # ------------------------------------------------------------------

    def _broadcast_phase1(
        self,
        trade: TradeExecution,
    ) -> tuple[BlockchainTransaction, dict]:
        """Build, sign, and broadcast phase 1 (approve + CDW.deposit).

        :return:
            Tuple of (BlockchainTransaction, receipt dict).
        """
        raw_amount = self._get_raw_usdc_amount(trade)

        fn = build_hypercore_deposit_phase1(
            self.lagoon_vault,
            evm_usdc_amount=raw_amount,
        )
        tx = self._sign_multicall(
            fn,
            notes=f"Hypercore deposit phase 1: {raw_amount} raw USDC",
        )

        tx_hash = self.web3.eth.send_raw_transaction(HexBytes(tx.signed_bytes))
        receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        return tx, receipt

    def _broadcast_phase2(
        self,
        trade: TradeExecution,
        vault_address: str,
        raw_amount: int,
    ) -> tuple[BlockchainTransaction, dict]:
        """Build, sign, and broadcast phase 2 (transferUsdClass + vaultTransfer).

        :return:
            Tuple of (BlockchainTransaction, receipt dict).
        """
        fn = build_hypercore_deposit_phase2(
            self.lagoon_vault,
            hypercore_usdc_amount=raw_amount,
            vault_address=vault_address,
        )
        tx = self._sign_multicall(
            fn,
            notes=f"Hypercore deposit phase 2: {raw_amount} raw USDC to vault {vault_address}",
        )

        tx_hash = self.web3.eth.send_raw_transaction(HexBytes(tx.signed_bytes))
        receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        return tx, receipt

    def _settle_deposit(
        self,
        web3: Web3,
        state: State,
        trade: TradeExecution,
        receipts: Dict[str, dict],
        stop_on_execution_failure: bool,
    ):
        """Settle a Hypercore vault deposit (buy).

        Phase 1 tx was broadcast by the execution model. This method:

        1. Verifies the phase 1 receipt.
        2. Waits for EVM escrow to clear.
        3. Builds and broadcasts phase 2 (transferUsdClass + vaultTransfer).
        4. Queries vault equity and marks trade success.

        Activation (if needed) was already handled in :py:meth:`setup_trades`.
        """
        broadcast_tx = trade.blockchain_transactions[-1]
        try:
            receipt = receipts[HexBytes(broadcast_tx.tx_hash)]
        except KeyError as e:
            raise KeyError(
                f"Could not find tx hash {broadcast_tx.tx_hash} in receipts: "
                f"{list(receipts.keys())}"
            ) from e

        ts = get_block_timestamp(web3, receipt["blockNumber"])

        if receipt["status"] != 1:
            logger.error(
                "Hypercore vault deposit tx %s reverted (trade %s)",
                broadcast_tx.tx_hash,
                trade.trade_id,
            )
            report_failure(ts, state, trade, stop_on_execution_failure)
            return

        vault_address = self._get_vault_address(trade)
        deposit_raw = self._get_raw_usdc_amount(trade) - self._activation_cost_raw
        session = self._get_session()
        planned_reserve = trade.get_planned_reserve()

        logger.info(
            "Hypercore deposit phase 1 succeeded (tx %s). Waiting for escrow to clear...",
            broadcast_tx.tx_hash,
        )

        # --- Escrow wait ---
        try:
            wait_for_evm_escrow_clear(
                session,
                user=self.safe_address,
                timeout=60.0,
                poll_interval=2.0,
            )
        except TimeoutError as e:
            logger.error("EVM escrow did not clear: %s", e)
            report_failure(ts, state, trade, stop_on_execution_failure)
            return

        logger.info("Escrow cleared. Building and broadcasting phase 2...")

        # --- Phase 2 ---
        self.deployer.sync_nonce(web3)

        try:
            phase2_tx, phase2_receipt = self._broadcast_phase2(trade, vault_address, deposit_raw)
        except Exception as e:
            logger.error("Phase 2 broadcast failed: %s", e)
            report_failure(ts, state, trade, stop_on_execution_failure)
            return

        trade.blockchain_transactions.append(phase2_tx)

        if phase2_receipt["status"] != 1:
            logger.error("Hypercore deposit phase 2 reverted: tx %s", phase2_tx.tx_hash)
            report_failure(ts, state, trade, stop_on_execution_failure)
            return

        ts = get_block_timestamp(web3, phase2_receipt["blockNumber"])
        logger.info("Hypercore deposit phase 2 succeeded (tx %s)", phase2_tx.tx_hash)

        # --- Query vault equity and settle ---
        # executed_reserve = total USDC removed from Safe (activation + deposit).
        # executed_amount = vault equity received (queried from API).
        # The activation cost is accounted as part of the trade's reserve spend.
        executed_reserve = planned_reserve
        actual_deposit_human = Decimal(deposit_raw) / Decimal(10**6)
        executed_amount = actual_deposit_human

        try:
            eq = fetch_user_vault_equity(
                session,
                user=self.safe_address,
                vault_address=vault_address,
                bypass_cache=True,
            )
            if eq is not None:
                executed_amount = eq.equity
                logger.info(
                    "Vault equity after deposit: %s (deposited %s USDC, activation cost %s USDC)",
                    executed_amount,
                    actual_deposit_human,
                    Decimal(self._activation_cost_raw) / Decimal(10**6),
                )
        except Exception as e:
            logger.warning(
                "Could not query vault equity after deposit (using planned values): %s",
                e,
            )

        price = float(executed_reserve / executed_amount) if executed_amount else 1.0

        state.mark_trade_success(
            ts,
            trade,
            executed_price=price,
            executed_amount=executed_amount,
            executed_reserve=executed_reserve,
            lp_fees=0,
            native_token_price=0,
        )

        logger.info(
            "Hypercore vault deposit settled: %s USDC deposited, equity %s",
            executed_reserve,
            executed_amount,
        )

    def _settle_withdrawal(
        self,
        web3: Web3,
        state: State,
        trade: TradeExecution,
        receipts: Dict[str, dict],
        stop_on_execution_failure: bool,
    ):
        """Settle a Hypercore vault withdrawal (sell)."""
        withdraw_tx = trade.blockchain_transactions[-1]
        try:
            receipt = receipts[HexBytes(withdraw_tx.tx_hash)]
        except KeyError as e:
            raise KeyError(
                f"Could not find tx hash {withdraw_tx.tx_hash} in receipts: "
                f"{list(receipts.keys())}"
            ) from e

        ts = get_block_timestamp(web3, receipt["blockNumber"])

        if receipt["status"] != 1:
            logger.error(
                "Hypercore vault withdrawal tx %s reverted (trade %s)",
                withdraw_tx.tx_hash,
                trade.trade_id,
            )
            report_failure(ts, state, trade, stop_on_execution_failure)
            return

        logger.info(
            "Hypercore vault withdrawal succeeded (tx %s)",
            withdraw_tx.tx_hash,
        )

        planned_reserve = trade.get_planned_reserve()
        executed_reserve = planned_reserve
        executed_amount = -planned_reserve  # Negative for sells

        state.mark_trade_success(
            ts,
            trade,
            executed_price=1.0,
            executed_amount=executed_amount,
            executed_reserve=executed_reserve,
            lp_fees=0,
            native_token_price=0,
        )

        logger.info(
            "Hypercore vault withdrawal settled: %s USDC withdrawn",
            executed_reserve,
        )

    # ------------------------------------------------------------------
    # settle_trade (dispatcher)
    # ------------------------------------------------------------------

    def settle_trade(
        self,
        web3: Web3,
        state: State,
        trade: TradeExecution,
        receipts: Dict[str, dict],
        stop_on_execution_failure=False,
    ):
        """Settle a Hypercore vault trade.

        Dispatches to :py:meth:`_settle_deposit` or :py:meth:`_settle_withdrawal`
        based on the trade direction.
        """
        if trade.is_buy():
            self._settle_deposit(web3, state, trade, receipts, stop_on_execution_failure)
        else:
            self._settle_withdrawal(web3, state, trade, receipts, stop_on_execution_failure)
