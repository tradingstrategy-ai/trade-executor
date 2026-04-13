"""Route trades for Hypercore native vaults on HyperEVM.

Hypercore vault deposits/withdrawals use a multi-phase flow:

**Deposit (buy)**:

0. Activation (if needed): performed synchronously via
   :py:func:`~eth_defi.hyperliquid.evm_escrow.activate_account` in ``setup_trades()``
1. Phase 1: separate ``approve`` and ``CDW.deposit`` Safe calls — bridge USDC to HyperCore spot
2. Escrow wait: poll ``spotClearinghouseState`` until USDC clears
3. Phase 2: ``transferUsdClass`` + ``vaultTransfer`` — move to perp, deposit into vault

If the Safe is already activated, step 0 is skipped.

**Withdrawal (sell)**:

1. Phase 1: ``vaultTransfer`` — withdraw from vault to HyperCore perp
2. Perp wait: poll ``clearinghouseState`` until withdrawable USDC appears
3. Phase 2: ``transferUsdClass`` — move from perp to spot
4. Spot wait: poll ``spotClearinghouseState`` until free USDC appears
5. Phase 3: ``sendAsset`` — bridge USDC from HyperCore spot back to HyperEVM

The build functions from :py:mod:`eth_defi.hyperliquid.core_writer` return
either ``TradingStrategyModuleV0.performCall()`` or ``multicall()`` functions
that are already wrapped for the Safe. Using
:py:class:`~tradeexecutor.ethereum.lagoon.tx.LagoonTransactionBuilder`
would double-wrap them, so we sign directly via :py:class:`~eth_defi.hotwallet.HotWallet`.

For the upstream guard-side and operator-side background on these Hypercore
deposit/withdrawal legs, see also:

- ``deps/web3-ethereum-defi/eth_defi/hyperliquid/core_writer.py``
- ``deps/web3-ethereum-defi/docs/README-Hypercore-guard.md``
- ``deps/web3-ethereum-defi/docs/source/tutorials/lagoon-hyperliquid.rst``
"""

from __future__ import annotations

import logging
import time
from decimal import Decimal
from typing import Dict, TYPE_CHECKING, cast

from hexbytes import HexBytes
from web3 import Web3
from web3.contract.contract import ContractFunction

from eth_defi.gas import apply_gas, estimate_gas_price
from eth_defi.hotwallet import HotWallet
from eth_defi.hyperliquid.api import (
    HypercoreDepositVerificationError,
    fetch_perp_clearinghouse_state,
    fetch_spot_clearinghouse_state,
    fetch_user_abstraction_mode,
    fetch_user_vault_equity,
    wait_for_vault_deposit_confirmation,
)
from eth_defi.hyperliquid.core_writer import (
    MINIMUM_VAULT_DEPOSIT,
    build_hypercore_approve_deposit_wallet_call,
    build_hypercore_deposit_multicall,
    build_hypercore_deposit_to_spot_call,
    build_hypercore_deposit_phase2,
    compute_spot_to_evm_withdrawal_amount,
    build_hypercore_withdraw_from_vault_call,
    build_hypercore_send_asset_to_evm_call,
    build_hypercore_transfer_usd_class_call,
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
from tradeexecutor.state.trade import TradeExecution, TradeFlag
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

#: Gas limit for Hypercore TradingStrategyModuleV0 transactions.
#:
#: The batched Hypercore deposit on a HyperEVM Anvil fork currently needs a bit
#: over 520k gas once routed through TradingStrategyModuleV0. Keep a modest
#: buffer here so the live split tests do not fail due to an avoidable gas cap.
HYPERCORE_MULTICALL_GAS = 650_000

#: USDC uses 6 decimals on HyperEVM.
USDC_DECIMALS = 6

#: When closing a Hypercore vault position, the planned withdrawal amount
#: (from trade creation) and the live vault equity (at execution time) must
#: agree within this tolerance.  If live equity is below this fraction of
#: the planned amount, something is seriously wrong (wrong vault, corrupted
#: state, major vault event) and we abort rather than risk a bad withdrawal.
HYPERCORE_LIKELY_CLOSE_TOLERANCE = 0.975

#: Safety margin (raw USDC, 6 decimals) subtracted from live vault equity
#: when building full-close withdrawals.
#:
#: HyperCore's ``vaultTransfer`` silently rejects withdrawals that exceed
#: actual equity — the EVM tx succeeds but zero USDC moves.  Between the
#: API read (``userVaultEquities``) and HyperCore processing the queued
#: action, the vault NAV can fluctuate (fees, PnL, mark-to-market).  The
#: API may also round the equity string upward vs. the on-chain value.
#:
#: $1.00 (1 000 000 raw) covers larger observed drift for volatile vaults.
#: The original $0.01 margin was too tight, and even the later $0.10 margin
#: proved insufficient for some live withdrawals when vault share price moved
#: between the API read and HyperCore processing the queued action.
#:
#: This will leave up to ~$1.00 unclaimable dust in the vault (below the $5
#: minimum vault withdrawal threshold) that must be cleaned up in
#: accounting later.
HYPERCORE_WITHDRAWAL_SAFETY_MARGIN_RAW = 1_000_000

# Temporary stop-gap for follow-up withdrawal verification phases.
# Proper fix: carry the actually observed amount from one phase to the next.
# Remove this extra slack once settlement becomes amount-adaptive across phases.
HYPERCORE_FOLLOW_UP_PHASE_TOLERANCE_RAW = 200_000

def usdc_to_raw(amount: Decimal) -> int:
    """Convert a human-readable USDC amount to raw 6-decimal integer."""
    return int(amount * 10**USDC_DECIMALS)


def raw_to_usdc(raw: int) -> Decimal:
    """Convert a raw 6-decimal USDC integer to human-readable Decimal."""
    return Decimal(raw) / Decimal(10**USDC_DECIMALS)


class HypercoreWithdrawalVerificationError(Exception):
    """Raised when a Hypercore vault withdrawal cannot be verified on EVM.

    The withdrawal multicall succeeded on HyperEVM but the expected USDC
    did not arrive in the Safe's EVM balance within the timeout period.
    This can happen when any of the three CoreWriter actions
    (``vaultTransfer``, ``transferUsdClass``, ``spotSend``) fail silently
    on HyperCore, or when the HyperCore-to-EVM bridge is delayed or dry.
    """


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
    ``TradingStrategyModuleV0`` interface.

    The build functions from :py:mod:`eth_defi.hyperliquid.core_writer` return
    ready-to-sign ``ContractFunction`` objects already wrapped for the Safe.
    These are signed directly with the deployer
    :py:class:`~eth_defi.hotwallet.HotWallet` rather than going through
    ``LagoonTransactionBuilder.sign_transaction()`` which would double-wrap them.
    """

    def __init__(
        self,
        web3: Web3,
        lagoon_vault: LagoonVault,
        deployer: HotWallet,
        reserve_token_address: str,
        simulate: bool = False,
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

        :param simulate:
            If True, use batched multicall (all 4 steps in one tx) instead of
            two-phase deposit with escrow wait. For Anvil fork testing with
            mock Hypercore Writer contracts.
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
        self.simulate = simulate
        self._session: HyperliquidSession | None = None

        # Verify deployer is not in big blocks mode.  If left enabled
        # from a previous deployment, all transactions would go to the
        # large block mempool (~1 min confirmation instead of ~1s).
        if not simulate:
            try:
                from eth_defi.hyperliquid.block import fetch_using_big_blocks
                if fetch_using_big_blocks(web3, deployer.address):
                    raise AssertionError(
                        f"Deployer {deployer.address} has big blocks enabled on HyperEVM. "
                        f"This causes ~1 minute confirmation times instead of ~1 second. "
                        f"Disable big blocks before running the strategy."
                    )
            except Exception as e:
                if isinstance(e, AssertionError):
                    raise
                # RPC method may not be available on Anvil forks or
                # non-HyperEVM chains — log and continue.
                logger.debug("Could not check big blocks mode: %s", e)

    @property
    def safe_address(self) -> str:
        return self.lagoon_vault.safe_address

    def _get_session(self) -> HyperliquidSession:
        """Lazily create a Hyperliquid API session."""
        if self._session is None:
            api_url = HYPERLIQUID_TESTNET_API_URL if self.is_testnet else HYPERLIQUID_API_URL
            self._session = create_hyperliquid_session(api_url=api_url)
        return self._session

    def _log_account_mode(self, context: str) -> str | None:
        """Log the Safe's Hyperliquid account mode for diagnostics."""
        session = self._get_session()
        try:
            mode = fetch_user_abstraction_mode(session, user=self.safe_address)
        except Exception as e:
            logger.warning(
                "Could not read Hyperliquid account mode for Safe %s during %s: %s",
                self.safe_address,
                context,
                e,
            )
            return None

        logger.info(
            "Safe %s Hyperliquid account mode during %s: %s",
            self.safe_address,
            context,
            mode,
        )
        return mode

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

    def needs_sequential_trade_execution(
        self,
        trades: list[TradeExecution],
    ) -> bool:
        """Hypercore trades must settle before the next trade is prepared.

        Hypercore deposits and withdrawals can:

        - release spendable capital only during settlement, not at phase 1 receipt
        - append extra follow-up transactions during settlement

        Because of this, batching multiple Hypercore trades before settling the
        earlier ones can mis-sequence capital reuse and nonce allocation.
        """
        return len(trades) > 0

    def get_sequential_trade_execution_reason(
        self,
        trades: list[TradeExecution],
    ) -> str | None:
        if not trades:
            return None
        return (
            "Hypercore vault trades release spendable capital only after settlement "
            "and may create follow-up settlement transactions"
        )

    # ------------------------------------------------------------------
    # Transaction building helpers
    # ------------------------------------------------------------------

    def _sign_module_call(
        self,
        fn: ContractFunction,
        gas_limit: int = HYPERCORE_MULTICALL_GAS,
        notes: str = "",
        logical_function_name: str | None = None,
    ) -> BlockchainTransaction:
        """Sign a TradingStrategyModuleV0 call and create a BlockchainTransaction.

        The build functions from ``eth_defi.hyperliquid.core_writer`` return
        ``module.functions.performCall(...)`` or ``module.functions.multicall(...)``
        objects already addressed to the module contract. We sign them directly
        with the deployer hot wallet.

        :param fn:
            Bound ``ContractFunction`` already wrapped through the trading strategy module.

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

        gas_price_suggestion = estimate_gas_price(self.web3)
        apply_gas(tx_data, gas_price_suggestion)

        signed_tx = self.deployer.sign_transaction_with_new_nonce(tx_data)
        signed_bytes = hexbytes_to_hex_str(signed_tx.rawTransaction)

        # Needed for get_swap_transactions() compatibility.
        tx_data["function"] = logical_function_name or fn.fn_name

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
        vault_addr = trade.pair.pool_address
        assert vault_addr, f"No pool_address set for Hypercore vault pair: {trade.pair}"
        return vault_addr

    def _get_raw_usdc_amount(self, trade: TradeExecution) -> int:
        """Get the raw USDC amount (6 decimals) for the trade."""
        planned_reserve = trade.get_planned_reserve()
        return usdc_to_raw(planned_reserve)

    def _get_live_withdrawal_amount_for_close(
        self,
        trade: TradeExecution,
        planned_raw: int,
        vault_address: str,
    ) -> int:
        """Query live vault equity and return the authoritative withdrawal amount.

        Hyperliquid has no "withdraw all" API — ``vaultTransfer`` requires an
        exact USD amount.  If we pass a value that exceeds the vault's current
        equity (even by a fraction), HyperCore silently rejects the withdrawal:
        the EVM precompile tx succeeds but zero USDC moves.

        For full position closes (``TradeFlag.close``), the planned reserve set
        at trade-creation time can drift from actual vault equity due to fees,
        PnL, or NAV changes.  This method fetches the **live** equity from the
        ``userVaultEquities`` API and uses it directly as the withdrawal amount.
        The planned value serves only as a sanity reference — if the two
        disagree by more than :py:data:`HYPERCORE_LIKELY_CLOSE_TOLERANCE`, we
        abort because something is seriously wrong.

        The live amount is stored in ``trade.other_data`` so that settlement
        phases 2-3 use the same value that phase 1's ``vaultTransfer`` was
        built with.

        :param trade:
            The closing trade.  Must have ``TradeFlag.close`` in its flags.
        :param planned_raw:
            Raw USDC amount (6 decimals) from ``trade.get_planned_reserve()``.
        :param vault_address:
            Hypercore vault address to query equity for.
        :return:
            Raw USDC amount to withdraw (the live vault equity).
        :raises AssertionError:
            If the vault has no position or the live equity diverges from
            the planned amount by more than the tolerance.
        """
        session = self._get_session()

        # 1. Fetch the live vault equity — bypass cache to get a fresh read
        #    right before we build the withdrawal tx.
        equity = fetch_user_vault_equity(
            session,
            user=self.safe_address,
            vault_address=vault_address,
            bypass_cache=True,
        )

        # 2. A close trade must have an existing vault position.
        #    If the API returns None the position has already been withdrawn
        #    or was never opened — something is very wrong.
        assert equity is not None, (
            f"Cannot close vault position: fetch_user_vault_equity returned None "
            f"for Safe {self.safe_address} in vault {vault_address}. "
            f"The position may already be withdrawn."
        )

        live_raw = usdc_to_raw(equity.equity)

        # 3. Sanity check: the planned and live values must agree within
        #    HYPERCORE_LIKELY_CLOSE_TOLERANCE (default 97.5%).  A larger
        #    divergence signals a wrong vault address, corrupted state, or a
        #    major vault event — abort rather than withdraw an unexpected amount.
        assert live_raw >= planned_raw * HYPERCORE_LIKELY_CLOSE_TOLERANCE, (
            f"Live vault equity ({equity.equity} USDC, {live_raw} raw) is too far "
            f"below planned withdrawal ({raw_to_usdc(planned_raw)} USDC, {planned_raw} raw) "
            f"for Safe {self.safe_address} in vault {vault_address}. "
            f"Ratio: {live_raw / planned_raw:.4f}, "
            f"tolerance: {HYPERCORE_LIKELY_CLOSE_TOLERANCE}. "
            f"Aborting withdrawal to avoid unexpected behaviour."
        )

        # 4. Subtract a safety margin from the live equity to account for
        #    NAV drift between the API read and HyperCore action execution.
        #    HyperCore silently rejects vaultTransfer withdrawals that exceed
        #    actual equity — even by 1 raw USDC.  The vault NAV fluctuates
        #    due to fees, PnL, and mark-to-market during the ~2-5 s between
        #    reading the API and HyperCore processing the queued action.
        #    This leaves ~$0.01 unclaimable dust in the vault that must be
        #    cleaned up in accounting later.
        safe_raw = live_raw - HYPERCORE_WITHDRAWAL_SAFETY_MARGIN_RAW
        assert safe_raw > 0, (
            f"Live vault equity {live_raw} raw ({equity.equity} USDC) is too small to "
            f"withdraw after subtracting safety margin {HYPERCORE_WITHDRAWAL_SAFETY_MARGIN_RAW} raw"
        )

        logger.info(
            "Full close: live vault equity %s USDC (%d raw), "
            "safety margin %d raw, withdrawal amount %d raw (%s USDC), "
            "planned %s USDC (%d raw) for Safe %s, vault %s",
            equity.equity, live_raw,
            HYPERCORE_WITHDRAWAL_SAFETY_MARGIN_RAW, safe_raw, raw_to_usdc(safe_raw),
            raw_to_usdc(planned_raw), planned_raw,
            self.safe_address, vault_address,
        )

        # 5. Store the safe amount in trade.other_data so settlement can read
        #    it back.  Phases 2-3 (transferUsdClass, sendAsset) must use the
        #    same amount that phase 1's vaultTransfer was built with.
        trade.other_data["hypercore_capped_withdrawal_raw"] = safe_raw

        return safe_raw

    def _fetch_safe_evm_usdc_balance(self) -> int:
        """Read the Safe's EVM USDC balance (raw, 6 decimals).

        :return:
            Raw USDC balance in the Safe on HyperEVM.
        """
        from eth_defi.abi import get_deployed_contract
        usdc_contract = get_deployed_contract(
            self.web3,
            "centre/ERC20.json",
            Web3.to_checksum_address(self.reserve_token_address),
        )
        return usdc_contract.functions.balanceOf(
            Web3.to_checksum_address(self.safe_address)
        ).call()

    def _fetch_safe_spot_free_usdc_balance(self) -> Decimal:
        """Read the Safe's free HyperCore spot USDC balance."""
        state = fetch_spot_clearinghouse_state(
            self._get_session(),
            user=self.safe_address,
        )
        for balance in state.balances:
            if balance.coin == "USDC":
                return balance.total - balance.hold
        return Decimal(0)

    def _fetch_safe_perp_withdrawable_balance(self) -> Decimal:
        """Read the Safe's HyperCore perp withdrawable USDC balance."""
        state = fetch_perp_clearinghouse_state(
            self._get_session(),
            user=self.safe_address,
        )
        return state.withdrawable

    def _wait_for_usdc_arrival(
        self,
        baseline_balance_raw: int,
        expected_increase_raw: int,
        timeout: float = 30.0,
        poll_interval: float = 2.0,
    ) -> int:
        """Poll the Safe's EVM USDC balance until withdrawn USDC arrives.

        The HyperCore-to-EVM bridge (``spotSend``) has async latency of
        typically 2-10 seconds.  This method polls until the Safe's USDC
        balance has increased by at least *expected_increase_raw*.

        :param baseline_balance_raw:
            Safe's EVM USDC balance (raw) captured before the withdrawal.
        :param expected_increase_raw:
            Expected USDC increase in raw units (6 decimals).
        :param timeout:
            Maximum seconds to wait.
        :param poll_interval:
            Seconds between balance polls.
        :return:
            Actual USDC increase in raw units.
        :raises HypercoreWithdrawalVerificationError:
            If USDC does not arrive within the timeout.
        """
        # Temporary stop-gap: accept a slightly smaller bridged amount here.
        # This covers minor drift between the requested phase 3 bridge amount
        # and the final EVM arrival, while still logging both numbers so we can
        # see whether the gap looks like ordinary fee/rounding behaviour or a
        # genuine settlement failure.
        #
        # Cross-reference in upstream eth_defi package:
        # - `eth_defi.hyperliquid.core_writer.compute_spot_to_evm_withdrawal_amount()`
        # - `eth_defi.hyperliquid.core_writer.build_hypercore_send_asset_to_evm_call()`
        # - `deps/web3-ethereum-defi/docs/README-Hypercore-guard.md`
        expected_increase_threshold_raw = max(
            expected_increase_raw - HYPERCORE_FOLLOW_UP_PHASE_TOLERANCE_RAW,
            0,
        )
        deadline = time.time() + timeout
        attempt = 0

        logger.info(
            "Waiting for Hypercore phase 3 USDC arrival in Safe %s: "
            "baseline %d raw, expected %d raw, threshold %d raw, tolerance slack %d raw",
            self.safe_address,
            baseline_balance_raw,
            expected_increase_raw,
            expected_increase_threshold_raw,
            expected_increase_raw - expected_increase_threshold_raw,
        )

        # Initial delay: give the bridge time to process
        time.sleep(poll_interval)

        while True:
            attempt += 1
            current_balance_raw = self._fetch_safe_evm_usdc_balance()
            increase = current_balance_raw - baseline_balance_raw

            if increase >= expected_increase_threshold_raw:
                logger.info(
                    "USDC arrived in Safe %s after %d poll(s): "
                    "+%d raw (expected at least %d raw, planned %d raw)",
                    self.safe_address, attempt,
                    increase, expected_increase_threshold_raw, expected_increase_raw,
                )
                return increase

            remaining = deadline - time.time()
            if remaining <= 0:
                raise HypercoreWithdrawalVerificationError(
                    f"USDC did not arrive in Safe {self.safe_address} within {timeout}s. "
                    f"Expected increase at threshold: {expected_increase_threshold_raw} raw "
                    f"(planned {expected_increase_raw} raw), "
                    f"actual increase: {increase} raw, "
                    f"baseline: {baseline_balance_raw} raw, "
                    f"current: {current_balance_raw} raw, "
                    f"after {attempt} poll(s). "
                    f"The HyperCore-to-EVM bridge may be dry or one of the "
                    f"CoreWriter actions (vaultTransfer/transferUsdClass/spotSend) "
                    f"failed silently on HyperCore."
                )

            logger.info(
                "Waiting for USDC in Safe %s: baseline %d raw, current %d raw, "
                "increase %d/%d raw (planned %d raw, %.0fs remaining, poll #%d)",
                self.safe_address,
                baseline_balance_raw,
                current_balance_raw,
                increase,
                expected_increase_threshold_raw,
                expected_increase_raw,
                remaining,
                attempt,
            )
            time.sleep(min(poll_interval, remaining))

    def _wait_for_perp_withdrawable_balance(
        self,
        baseline_balance: Decimal,
        expected_increase_raw: int,
        timeout: float = 30.0,
        poll_interval: float = 2.0,
    ) -> Decimal:
        """Poll HyperCore perp withdrawable USDC until the withdrawal reaches perp."""
        # Allow $0.10 tolerance for Hypercore API rounding and vault NAV drift
        expected_balance = baseline_balance + raw_to_usdc(expected_increase_raw) - Decimal("0.10")
        deadline = time.time() + timeout
        attempt = 0

        while True:
            attempt += 1
            current_balance = self._fetch_safe_perp_withdrawable_balance()

            if current_balance >= expected_balance:
                logger.info(
                    "Perp withdrawable balance is ready for Safe %s after %d poll(s): "
                    "%s USDC (expected at least %s USDC)",
                    self.safe_address,
                    attempt,
                    current_balance,
                    expected_balance,
                )
                return current_balance

            remaining = deadline - time.time()
            if remaining <= 0:
                raise HypercoreWithdrawalVerificationError(
                    f"Perp withdrawable USDC did not reach {expected_balance} for Safe {self.safe_address} "
                    f"within {timeout}s. Current balance: {current_balance} after {attempt} poll(s). "
                    f"The vaultTransfer action may have failed silently on HyperCore."
                )

            logger.info(
                "Waiting for perp withdrawable USDC in Safe %s: %s/%s USDC "
                "(%.0fs remaining, poll #%d)",
                self.safe_address,
                current_balance,
                expected_balance,
                remaining,
                attempt,
            )
            time.sleep(min(poll_interval, remaining))

    def _wait_for_spot_free_usdc_balance(
        self,
        baseline_balance: Decimal,
        expected_increase_raw: int,
        timeout: float = 30.0,
        poll_interval: float = 2.0,
    ) -> Decimal:
        """Poll HyperCore spot free USDC until the perp-to-spot move is visible."""
        # Temporary stop-gap: accept a slightly smaller spot balance here.
        # Proper fix: propagate the observed perp balance into phase 2.
        expected_increase_threshold_raw = max(
            expected_increase_raw - HYPERCORE_FOLLOW_UP_PHASE_TOLERANCE_RAW,
            0,
        )
        expected_balance = baseline_balance + raw_to_usdc(expected_increase_threshold_raw)
        deadline = time.time() + timeout
        attempt = 0

        while True:
            attempt += 1
            current_balance = self._fetch_safe_spot_free_usdc_balance()

            if current_balance >= expected_balance:
                logger.info(
                    "Spot free USDC is ready for Safe %s after %d poll(s): "
                    "%s USDC (expected at least %s USDC)",
                    self.safe_address,
                    attempt,
                    current_balance,
                    expected_balance,
                )
                return current_balance

            remaining = deadline - time.time()
            if remaining <= 0:
                raise HypercoreWithdrawalVerificationError(
                    f"Spot free USDC did not reach {expected_balance} for Safe {self.safe_address} "
                    f"within {timeout}s. Current balance: {current_balance} after {attempt} poll(s). "
                    f"The transferUsdClass(perp->spot) action may have failed silently on HyperCore."
                )

            logger.info(
                "Waiting for spot free USDC in Safe %s: %s/%s USDC "
                "(%.0fs remaining, poll #%d)",
                self.safe_address,
                current_balance,
                expected_balance,
                remaining,
                attempt,
            )
            time.sleep(min(poll_interval, remaining))

    def _mark_stranded_usdc(
        self,
        trade: TradeExecution,
        raw_amount: int,
        location: str,
    ):
        """Record stranded USDC info on a failed trade for operator recovery.

        When phase 2 fails after phase 1 bridged USDC to HyperCore,
        or when the vault deposit is silently rejected, this method
        stores recovery info in ``trade.other_data``.

        :param trade:
            The failed trade.
        :param raw_amount:
            Stranded USDC amount in raw units (6 decimals).
        :param location:
            Where the USDC is stranded (e.g. ``"hypercore_spot"``).
        """
        human_amount = raw_to_usdc(raw_amount)
        stranded_info = {
            "amount_raw": raw_amount,
            "amount_human": str(human_amount),
            "location": location,
            "safe_address": self.safe_address,
            "recovery": (
                f"USDC ({human_amount}) is stranded in {location} for "
                f"Safe {self.safe_address}. Use check-hypercore-user.py to "
                f"verify, then manually execute spotSend to bridge back to EVM "
                f"or complete the vault deposit."
            ),
        }
        if not hasattr(trade, "other_data") or trade.other_data is None:
            trade.other_data = {}
        trade.other_data["hypercore_stranded_usdc"] = stranded_info
        trade.add_note(
            f"USDC stranded on HyperCore ({human_amount} USDC in {location})"
        )
        logger.error(
            "STRANDED USDC: %s USDC in %s for Safe %s (trade %s). "
            "Use check-hypercore-user.py to verify and recover.",
            human_amount, location, self.safe_address, trade.trade_id,
        )

    def _create_deposit_or_withdraw_txs(
        self,
        trade: TradeExecution,
        activation_cost_raw: int = 0,
    ) -> list[BlockchainTransaction]:
        """Create blockchain transactions for a vault deposit or withdrawal.

        - **Buy (deposit)**: creates phase 1 approve + deposit txs.
          Phase 2 is handled in :py:meth:`_settle_deposit`.
        - **Sell (withdrawal)**: creates phase 1 only (vault -> perp).

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

        # Enforce Hyperliquid minimum vault deposit/withdrawal amounts.
        # Hyperliquid silently rejects vault transfers below the minimum
        # threshold — no error, no event, the USDC just stays in the
        # escrow. The minimum is 5 USDC (MINIMUM_VAULT_DEPOSIT = 5_000_000
        # raw, 6 decimals), determined by reverse-engineering the
        # Hyperliquid web UI. See eth_defi.hyperliquid.core_writer.
        if trade.is_buy():
            activation_hint = (
                f" Note: {activation_cost_raw / 1e6:.0f} USDC was deducted for "
                f"HyperCore account activation on first deposit — "
                f"minimum initial deposit is "
                f"{(MINIMUM_VAULT_DEPOSIT + activation_cost_raw) / 1e6:.0f} USDC."
                if activation_cost_raw > 0 else ""
            )
            assert raw_amount >= MINIMUM_VAULT_DEPOSIT, (
                f"Vault deposit amount {raw_amount / 1e6:.2f} USDC "
                f"({raw_amount} raw) is below the Hyperliquid minimum "
                f"of {MINIMUM_VAULT_DEPOSIT / 1e6:.0f} USDC. "
                f"Hyperliquid silently rejects deposits below this threshold."
                f"{activation_hint}"
            )

        if trade.is_buy():
            if self.simulate:
                logger.info(
                    "Building Hypercore vault deposit (simulate, batched): %d raw USDC to vault %s",
                    raw_amount,
                    vault_address,
                )
                fn = build_hypercore_deposit_multicall(
                    lagoon_vault=self.lagoon_vault,
                    evm_usdc_amount=raw_amount,
                    hypercore_usdc_amount=raw_amount,
                    vault_address=vault_address,
                    check_activation=False,
                    chain_id=self.web3.eth.chain_id,
                    asset_address=self.reserve_token_address,
                )
                tx = self._sign_module_call(fn, notes=f"Hypercore deposit (simulate): {raw_amount} raw USDC")
                return [tx]
            else:
                logger.info(
                    "Building Hypercore vault deposit phase 1 as separate approve + deposit calls: %d raw USDC to vault %s",
                    raw_amount,
                    vault_address,
                )
                approve_fn = build_hypercore_approve_deposit_wallet_call(
                    self.lagoon_vault,
                    evm_usdc_amount=raw_amount,
                )
                deposit_fn = build_hypercore_deposit_to_spot_call(
                    self.lagoon_vault,
                    evm_usdc_amount=raw_amount,
                )
                approve_tx = self._sign_module_call(
                    approve_fn,
                    notes=f"Hypercore deposit phase 1 approval: {raw_amount} raw USDC",
                    logical_function_name="approve",
                )
                deposit_tx = self._sign_module_call(
                    deposit_fn,
                    notes=f"Hypercore deposit phase 1 deposit: {raw_amount} raw USDC",
                    logical_function_name="deposit",
                )
                return [approve_tx, deposit_tx]

        else:
            # For full position close, query live vault equity and use that as
            # the withdrawal amount.  HyperCore's vaultTransfer precompile
            # silently rejects withdrawals that exceed actual equity — the EVM
            # tx succeeds but zero USDC moves.  The live equity is the
            # authoritative amount; planned_reserve is only a sanity reference.
            if TradeFlag.close in trade.flags and not self.simulate:
                raw_amount = self._get_live_withdrawal_amount_for_close(
                    trade, raw_amount, vault_address,
                )

            logger.info(
                "Building Hypercore vault withdrawal phase 1: %d raw USDC from vault %s",
                raw_amount,
                vault_address,
            )
            logger.info(
                "Queueing Hypercore withdrawal steps for Safe %s: "
                "1) vaultTransfer(vault->perp), "
                "2) transferUsdClass(perp->spot), "
                "3) spotSend(spot->EVM Safe) "
                "for %d raw USDC",
                self.safe_address,
                raw_amount,
            )
            fn = build_hypercore_withdraw_from_vault_call(
                self.lagoon_vault,
                vault_address=vault_address,
                hypercore_usdc_amount=raw_amount,
            )
            tx = self._sign_module_call(
                fn,
                notes=f"Hypercore withdrawal phase 1: {raw_amount} raw USDC from vault {vault_address}",
                logical_function_name="sendRawAction",
            )
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
            "Preparing %d trades for Hypercore vault execution (simulate=%s)",
            len(trades),
            self.simulate,
        )

        # Activation cost is tracked per-trade in trade.other_data, not
        # on the instance, so there is no stale state to reset between cycles.

        # In multichain setups, GenericRouting creates a satellite wallet
        # with the correct nonce for this chain and wraps it in a
        # LagoonTransactionBuilder with the satellite vault.  Use these
        # instead of the originals (which target the primary chain).
        #
        # Note: the primary-chain gas check does NOT cover satellite chains.
        # If the deployer runs out of HYPE on HyperEVM, multicall transactions
        # fail with out-of-gas.  Consider adding a HYPE balance check here
        # if gas management becomes a recurring issue.
        if hasattr(routing_state, 'tx_builder') and hasattr(routing_state.tx_builder, 'hot_wallet'):
            self.deployer = routing_state.tx_builder.hot_wallet
        if hasattr(routing_state, 'tx_builder') and hasattr(routing_state.tx_builder, 'vault'):
            self.lagoon_vault = routing_state.tx_builder.vault

        # Hypercore transactions are signed directly with the deployer hot wallet
        # instead of the generic transaction builder.  Refresh the nonce here so
        # earlier Lagoon setup calls done by fixtures or previous phases do not
        # leave us signing a stale nonce and getting a misleading revert.
        self.deployer.sync_nonce(self.web3)

        activation_cost_raw = 0

        if self.simulate:
            logger.info("Simulate mode — skipping activation check")
        else:
            activated = is_account_activated(self.web3, self.safe_address)

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
                activated = True
                activation_cost_raw = DEFAULT_ACTIVATION_AMOUNT
                logger.info(
                    "Safe %s activated on HyperCore (cost %d raw USDC from Safe)",
                    self.safe_address,
                    activation_cost_raw,
                )

            if activated:
                self._log_account_mode("setup_trades")

        # Only the first buy trade in the cycle bears the activation cost.
        activation_cost_applied = False

        for trade in trades:
            assert trade.is_vault(), f"Not a vault trade: {trade}"

            if not self.simulate and not trade.is_buy() and not activated:
                raise AssertionError(
                    f"Cannot withdraw from Hypercore vault: Safe {self.safe_address} "
                    f"is not activated on HyperCore."
                )

            cost_for_this_trade = activation_cost_raw if (trade.is_buy() and not activation_cost_applied) else 0

            trade.blockchain_transactions = self._create_deposit_or_withdraw_txs(
                trade,
                activation_cost_raw=cost_for_this_trade,
            )

            # Persist per-trade activation cost for settlement to read.
            if trade.is_buy() and cost_for_this_trade > 0:
                activation_cost_applied = True
                trade.add_note(f"Activation cost: {cost_for_this_trade} raw USDC")
                if not hasattr(trade, "other_data") or trade.other_data is None:
                    trade.other_data = {}
                trade.other_data["hypercore_activation_cost_raw"] = cost_for_this_trade

    # ------------------------------------------------------------------
    # Settlement helpers
    # ------------------------------------------------------------------

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
        tx = self._sign_module_call(
            fn,
            notes=f"Hypercore deposit phase 2: {raw_amount} raw USDC to vault {vault_address}",
        )

        tx_hash = self.web3.eth.send_raw_transaction(HexBytes(tx.signed_bytes))
        receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        return tx, receipt

    def _broadcast_withdrawal_phase2(
        self,
        raw_amount: int,
    ) -> tuple[BlockchainTransaction, dict]:
        """Build, sign, and broadcast withdrawal phase 2 (perp -> spot)."""
        fn = build_hypercore_transfer_usd_class_call(
            self.lagoon_vault,
            hypercore_usdc_amount=raw_amount,
            to_perp=False,
        )
        tx = self._sign_module_call(
            fn,
            notes=f"Hypercore withdrawal phase 2: {raw_amount} raw USDC perp->spot",
            logical_function_name="sendRawAction",
        )

        tx_hash = self.web3.eth.send_raw_transaction(HexBytes(tx.signed_bytes))
        receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        return tx, receipt

    def _broadcast_withdrawal_phase3(
        self,
        raw_amount: int,
    ) -> tuple[BlockchainTransaction, dict]:
        """Build, sign, and broadcast withdrawal phase 3 (spot -> EVM)."""
        fn = build_hypercore_send_asset_to_evm_call(
            self.lagoon_vault,
            evm_usdc_amount=raw_amount,
        )
        tx = self._sign_module_call(
            fn,
            notes=f"Hypercore withdrawal phase 3: {raw_amount} raw USDC spot->EVM",
            logical_function_name="sendRawAction",
        )

        tx_hash = self.web3.eth.send_raw_transaction(HexBytes(tx.signed_bytes))
        receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        return tx, receipt

    def _settle_deposit_simulate(
        self,
        web3: Web3,
        state: State,
        trade: TradeExecution,
        receipts: Dict[str, dict],
        stop_on_execution_failure: bool,
    ):
        """Settle a Hypercore vault deposit in simulate mode.

        The batched multicall already executed all 4 steps in a single tx.
        No escrow wait or phase 2 needed — just verify receipt and mark success.

        :param web3:
            HyperEVM Web3 connection (Anvil fork).

        :param state:
            Strategy state to update.

        :param trade:
            The vault deposit trade to settle.

        :param receipts:
            Transaction receipts from the broadcast step, keyed by tx hash.

        :param stop_on_execution_failure:
            If ``True``, raise on revert instead of marking trade as failed.
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
                "Hypercore vault deposit (simulate) tx %s reverted (trade %s)",
                broadcast_tx.tx_hash,
                trade.trade_id,
            )
            report_failure(ts, state, trade, stop_on_execution_failure)
            return

        planned_reserve = trade.get_planned_reserve()
        executed_reserve = planned_reserve
        executed_amount = planned_reserve  # In simulate mode, 1:1 USDC deposit

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
            "Hypercore vault deposit (simulate) settled: %s USDC deposited",
            executed_reserve,
        )

    def _settle_deposit(
        self,
        web3: Web3,
        state: State,
        trade: TradeExecution,
        receipts: Dict[str, dict],
        stop_on_execution_failure: bool,
    ):
        """Settle a Hypercore vault deposit (buy).

        Phase 1 approve + deposit txs were broadcast by the execution model. This method:

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
        # Read per-trade activation cost (only set on the first buy).
        activation_cost = 0
        if hasattr(trade, "other_data") and trade.other_data:
            activation_cost = trade.other_data.get("hypercore_activation_cost_raw", 0)
        deposit_raw = self._get_raw_usdc_amount(trade) - activation_cost
        session = self._get_session()
        planned_reserve = trade.get_planned_reserve()

        logger.info(
            "Hypercore deposit phase 1 succeeded (tx %s). Waiting for escrow to clear...",
            broadcast_tx.tx_hash,
        )

        # --- Escrow wait ---
        # Pass expected USDC so the escrow wait also verifies that USDC
        # appeared in the HyperCore spot balance, not just that escrows cleared.
        expected_usdc_human = raw_to_usdc(deposit_raw)
        try:
            wait_for_evm_escrow_clear(
                session,
                user=self.safe_address,
                timeout=60.0,
                poll_interval=2.0,
                expected_usdc=expected_usdc_human,
            )
        except TimeoutError as e:
            logger.error("EVM escrow did not clear: %s", e)
            report_failure(ts, state, trade, stop_on_execution_failure)
            return

        logger.info("Escrow cleared. Building and broadcasting phase 2...")

        # Snapshot existing vault equity before phase 2 so we can detect
        # the increase after deposit.  If the snapshot fails, abort: without
        # a baseline the verification step cannot distinguish pre-existing
        # equity from a fresh deposit and would falsely mark a silent
        # HyperCore rejection as success.
        existing_equity: Decimal | None = None
        try:
            eq_before = fetch_user_vault_equity(
                session,
                user=self.safe_address,
                vault_address=vault_address,
                bypass_cache=True,
            )
            if eq_before is not None:
                existing_equity = eq_before.equity
                logger.info("Existing vault equity before phase 2: %s", existing_equity)
        except Exception as e:
            logger.error(
                "Cannot snapshot vault equity before phase 2: %s. "
                "Aborting deposit to prevent false verification.",
                e,
            )
            self._mark_stranded_usdc(trade, deposit_raw, "hypercore_spot")
            report_failure(ts, state, trade, stop_on_execution_failure)
            return

        # --- Phase 2 ---
        # Note: RPC failure during nonce sync between phases could strand
        # USDC in HyperCore spot.  Manual recovery via check-hypercore-user.py.
        self.deployer.sync_nonce(web3)

        try:
            phase2_tx, phase2_receipt = self._broadcast_phase2(trade, vault_address, deposit_raw)
        except Exception as e:
            logger.error("Phase 2 broadcast failed: %s", e)
            self._mark_stranded_usdc(trade, deposit_raw, "hypercore_spot")
            report_failure(ts, state, trade, stop_on_execution_failure)
            return

        trade.blockchain_transactions.append(phase2_tx)

        if phase2_receipt["status"] != 1:
            logger.error("Hypercore deposit phase 2 reverted: tx %s", phase2_tx.tx_hash)
            self._mark_stranded_usdc(trade, deposit_raw, "hypercore_spot_or_perp")
            report_failure(ts, state, trade, stop_on_execution_failure)
            return

        ts = get_block_timestamp(web3, phase2_receipt["blockNumber"])
        logger.info("Hypercore deposit phase 2 succeeded (tx %s)", phase2_tx.tx_hash)

        # --- Verify deposit on HyperCore via poll loop ---
        # CoreWriter actions are NOT atomic: the EVM tx can succeed but the
        # deposit may be silently rejected by HyperCore.  We poll the API
        # until vault equity appears/increases, or fail the trade.
        executed_reserve = planned_reserve
        actual_deposit_human = raw_to_usdc(deposit_raw)

        try:
            confirmed_eq = wait_for_vault_deposit_confirmation(
                session,
                user=self.safe_address,
                vault_address=vault_address,
                expected_deposit=actual_deposit_human,
                existing_equity=existing_equity,
                timeout=60.0,
                poll_interval=2.0,
            )
            # Use the deposited amount (delta) as executed_amount, NOT the
            # total vault equity. Position quantity tracks cumulative USDC
            # deposited; the valuation model then computes per-unit price
            # as equity/quantity so that value = equity.
            #
            # For the first HyperCore deposit, the activation fee is deducted
            # from deposit_raw above. This means the first position absorbs
            # the activation fee through a smaller executed_amount and thus a
            # higher executed_price (executed_reserve / executed_amount),
            # instead of treating the fee as a separate neutral transfer.
            executed_amount = actual_deposit_human
            logger.info(
                "Vault equity after deposit: %s (deposited %s USDC, activation cost %s USDC)",
                confirmed_eq.equity,
                actual_deposit_human,
                raw_to_usdc(activation_cost),
            )
        except HypercoreDepositVerificationError as e:
            logger.error(
                "Vault deposit verification failed for trade %s: %s",
                trade.trade_id, e,
            )
            # Deposit silently rejected — USDC stranded in spot or perp
            self._mark_stranded_usdc(trade, deposit_raw, "hypercore_spot_or_perp")
            report_failure(ts, state, trade, stop_on_execution_failure)
            return

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
        """Settle a Hypercore vault withdrawal (sell).

        After the phase 1 vault-withdraw tx lands on HyperEVM:

        1. Verifies the phase 1 EVM receipt.
        2. Waits for the withdrawn USDC to appear in HyperCore perp.
        3. Broadcasts phase 2 (``transferUsdClass(perp->spot)``).
        4. Waits for the USDC to appear in HyperCore spot.
        5. Broadcasts phase 3 (``sendAsset`` / spot -> EVM).
        6. Polls the Safe's EVM USDC balance until the bridged USDC arrives.
        7. Compares actual received USDC against planned amount.
        8. Marks the trade as success or failure.

        In simulate mode, the balance verification is skipped because
        the mock CoreWriter does not actually bridge USDC.
        """
        # Capture baseline USDC balance before processing receipt.
        # The withdrawal multicall has already been mined but spotSend's
        # bridge delivery takes 2-10 seconds, so USDC hasn't arrived yet.
        # Capture baseline EVM USDC balance and vault equity before the
        # bridge delivers the withdrawal.  These are used for dual-chain
        # verification after the USDC arrives on EVM.
        if not self.simulate:
            baseline_balance_raw = self._fetch_safe_evm_usdc_balance()
            expected_raw = self._get_raw_usdc_amount(trade)

            # If the withdrawal amount was adjusted during phase 1 build
            # (full close trades that use live vault equity), use that amount
            # for all settlement phases.  Phase 1's vaultTransfer was built
            # with this amount, so phases 2-3 must transfer exactly the same
            # amount through the pipeline: perp → spot → EVM.
            if "hypercore_capped_withdrawal_raw" in trade.other_data:
                capped_raw = trade.other_data["hypercore_capped_withdrawal_raw"]
                logger.info(
                    "Using live equity withdrawal amount for settlement: %d raw USDC (%s USDC) "
                    "(planned was %d raw USDC, safety margin was %d raw)",
                    capped_raw, raw_to_usdc(capped_raw),
                    expected_raw, HYPERCORE_WITHDRAWAL_SAFETY_MARGIN_RAW,
                )
                expected_raw = capped_raw

            logger.info(
                "Withdrawal settlement: Safe %s baseline EVM USDC = %d raw, "
                "expected increase = %d raw",
                self.safe_address, baseline_balance_raw, expected_raw,
            )
            equity_before: Decimal | None = None
            vault_address = self._get_vault_address(trade)
            session = self._get_session()
            baseline_perp_withdrawable = self._fetch_safe_perp_withdrawable_balance()
            baseline_spot_free = self._fetch_safe_spot_free_usdc_balance()
            logger.info(
                "Withdrawal settlement: Safe %s baseline HyperCore perp withdrawable = %s USDC, "
                "baseline spot free USDC = %s",
                self.safe_address,
                baseline_perp_withdrawable,
                baseline_spot_free,
            )
            try:
                eq_before = fetch_user_vault_equity(
                    session,
                    user=self.safe_address,
                    vault_address=vault_address,
                    bypass_cache=True,
                )
                if eq_before is not None:
                    equity_before = eq_before.equity
                    logger.info(
                        "Vault equity before withdrawal: %s", equity_before,
                    )
            except Exception as e:
                logger.warning(
                    "Could not snapshot vault equity before withdrawal: %s", e,
                )

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
            "Hypercore vault withdrawal phase 1 EVM tx succeeded (%s)",
            withdraw_tx.tx_hash,
        )

        planned_reserve = trade.get_planned_reserve()

        if self.simulate:
            # Simulate mode: mock CoreWriter does not bridge USDC,
            # so skip EVM balance verification and trust planned values.
            executed_reserve = planned_reserve
        else:
            try:
                perp_balance = self._wait_for_perp_withdrawable_balance(
                    baseline_balance=baseline_perp_withdrawable,
                    expected_increase_raw=expected_raw,
                    timeout=30.0,
                    poll_interval=2.0,
                )
            except HypercoreWithdrawalVerificationError as e:
                logger.error(
                    "Withdrawal phase 1 verification failed for trade %s: %s",
                    trade.trade_id, e,
                )
                self._mark_stranded_usdc(trade, expected_raw, "hypercore_perp")
                report_failure(ts, state, trade, stop_on_execution_failure)
                return

            logger.info(
                "Before transferUsdClass(perp->spot), Safe %s perp withdrawable balance is %s USDC",
                self.safe_address,
                perp_balance,
            )

            self.deployer.sync_nonce(web3)

            try:
                phase2_tx, phase2_receipt = self._broadcast_withdrawal_phase2(expected_raw)
            except Exception as e:
                logger.error("Withdrawal phase 2 broadcast failed: %s", e)
                self._mark_stranded_usdc(trade, expected_raw, "hypercore_perp")
                report_failure(ts, state, trade, stop_on_execution_failure)
                return

            trade.blockchain_transactions.append(phase2_tx)

            if phase2_receipt["status"] != 1:
                logger.error("Hypercore withdrawal phase 2 reverted: tx %s", phase2_tx.tx_hash)
                self._mark_stranded_usdc(trade, expected_raw, "hypercore_perp")
                report_failure(ts, state, trade, stop_on_execution_failure)
                return

            ts = get_block_timestamp(web3, phase2_receipt["blockNumber"])

            try:
                spot_balance = self._wait_for_spot_free_usdc_balance(
                    baseline_balance=baseline_spot_free,
                    expected_increase_raw=expected_raw,
                    timeout=30.0,
                    poll_interval=2.0,
                )
            except HypercoreWithdrawalVerificationError as e:
                logger.error(
                    "Withdrawal phase 2 verification failed for trade %s: %s",
                    trade.trade_id, e,
                )
                self._mark_stranded_usdc(trade, expected_raw, "hypercore_spot")
                report_failure(ts, state, trade, stop_on_execution_failure)
                return

            logger.info(
                "Before sendAsset(spot->EVM), Safe %s spot free USDC balance is %s USDC",
                self.safe_address,
                spot_balance,
            )

            # `spotSend` pays the HyperCore -> HyperEVM bridge fee out of the
            # Safe's spot balance before the linked USDC lands on HyperEVM. If
            # we try to bridge the entire visible spot balance, the phase 3 call
            # can succeed on HyperEVM while silently doing nothing on HyperCore.
            #
            # Cross-reference in upstream eth_defi package:
            # - `eth_defi.hyperliquid.core_writer.build_hypercore_send_asset_to_evm_call()`
            # - `eth_defi.hyperliquid.core_writer.compute_spot_to_evm_withdrawal_amount()`
            # - `eth_defi.hyperliquid.core_writer.build_hypercore_withdraw_multicall()`
            # - `deps/web3-ethereum-defi/docs/README-Hypercore-guard.md`
            # - `deps/web3-ethereum-defi/docs/source/tutorials/lagoon-hyperliquid.rst`
            phase3_withdraw_amount = compute_spot_to_evm_withdrawal_amount(
                spot_balance=spot_balance,
                desired_amount=raw_to_usdc(expected_raw),
            )
            phase3_raw = usdc_to_raw(phase3_withdraw_amount)
            reserved_fee_headroom_raw = max(expected_raw - phase3_raw, 0)

            logger.info(
                "Hypercore withdrawal phase 3 planning for trade %s: "
                "requested %d raw (%s USDC), observed spot free %s USDC, "
                "bridging %d raw (%s USDC), reserved headroom %d raw (%s USDC)",
                trade.trade_id,
                expected_raw,
                raw_to_usdc(expected_raw),
                spot_balance,
                phase3_raw,
                phase3_withdraw_amount,
                reserved_fee_headroom_raw,
                raw_to_usdc(reserved_fee_headroom_raw),
            )

            if phase3_raw <= 0:
                logger.error(
                    "Hypercore withdrawal phase 3 cannot bridge any USDC for trade %s: "
                    "spot free balance %s USDC is too small after reserving bridge-fee headroom. "
                    "Requested %d raw (%s USDC), reserved headroom %d raw (%s USDC)",
                    trade.trade_id,
                    spot_balance,
                    expected_raw,
                    raw_to_usdc(expected_raw),
                    reserved_fee_headroom_raw,
                    raw_to_usdc(reserved_fee_headroom_raw),
                )
                self._mark_stranded_usdc(trade, expected_raw, "hypercore_spot")
                report_failure(ts, state, trade, stop_on_execution_failure)
                return

            if phase3_raw != expected_raw:
                logger.info(
                    "Adjusting Hypercore withdrawal phase 3 amount for bridge fee headroom: "
                    "spot free %s USDC, requested %d raw (%s USDC), bridging %d raw (%s USDC)",
                    spot_balance,
                    expected_raw,
                    raw_to_usdc(expected_raw),
                    phase3_raw,
                    phase3_withdraw_amount,
                )

            self.deployer.sync_nonce(web3)

            try:
                phase3_tx, phase3_receipt = self._broadcast_withdrawal_phase3(phase3_raw)
            except Exception as e:
                logger.error(
                    "Withdrawal phase 3 broadcast failed for trade %s: %s. "
                    "Requested %d raw, attempted %d raw after bridge-fee headroom",
                    trade.trade_id,
                    e,
                    expected_raw,
                    phase3_raw,
                )
                self._mark_stranded_usdc(trade, expected_raw, "hypercore_spot")
                report_failure(ts, state, trade, stop_on_execution_failure)
                return

            trade.blockchain_transactions.append(phase3_tx)

            if phase3_receipt["status"] != 1:
                logger.error(
                    "Hypercore withdrawal phase 3 reverted for trade %s: tx %s, "
                    "attempted bridge amount %d raw (%s USDC)",
                    trade.trade_id,
                    phase3_tx.tx_hash,
                    phase3_raw,
                    phase3_withdraw_amount,
                )
                self._mark_stranded_usdc(trade, expected_raw, "hypercore_spot")
                report_failure(ts, state, trade, stop_on_execution_failure)
                return

            ts = get_block_timestamp(web3, phase3_receipt["blockNumber"])

            logger.info(
                "Hypercore withdrawal phase 3 tx succeeded for trade %s: tx %s, "
                "bridge amount %d raw (%s USDC)",
                trade.trade_id,
                phase3_tx.tx_hash,
                phase3_raw,
                phase3_withdraw_amount,
            )

            # Poll for USDC arrival on EVM
            try:
                actual_increase_raw = self._wait_for_usdc_arrival(
                    baseline_balance_raw=baseline_balance_raw,
                    expected_increase_raw=phase3_raw,
                    timeout=30.0,
                    poll_interval=2.0,
                )
            except HypercoreWithdrawalVerificationError as e:
                logger.error(
                    "Withdrawal verification failed for trade %s after phase 3 tx %s: %s. "
                    "Requested %d raw, phase 3 attempted %d raw, baseline EVM balance %d raw",
                    trade.trade_id,
                    phase3_tx.tx_hash,
                    e,
                    expected_raw,
                    phase3_raw,
                    baseline_balance_raw,
                )
                self._mark_stranded_usdc(trade, expected_raw, "hypercore_spot")
                report_failure(ts, state, trade, stop_on_execution_failure)
                return

            executed_reserve = raw_to_usdc(actual_increase_raw)
            # Compare against the effective withdrawal target (live equity if
            # capped, else planned) to avoid spurious "partial" warnings on
            # normal capped-close trades where live equity < planned_reserve.
            effective_reserve = raw_to_usdc(phase3_raw)
            if executed_reserve < effective_reserve:
                logger.warning(
                    "Withdrawal partial: received %s USDC but expected %s USDC (trade %s)",
                    executed_reserve, effective_reserve, trade.trade_id,
                )

            # Dual-chain check: verify vault equity decreased on HyperCore
            # to match the USDC that arrived on EVM.  Non-fatal — EVM
            # verification already passed; this is an extra consistency check.
            try:
                eq_after = fetch_user_vault_equity(
                    session,
                    user=self.safe_address,
                    vault_address=vault_address,
                    bypass_cache=True,
                )
                remaining_equity = eq_after.equity if eq_after else Decimal(0)

                if equity_before is not None:
                    equity_decrease = equity_before - remaining_equity
                    expected_decrease = executed_reserve
                    tolerance = expected_decrease * Decimal("0.01")
                    if equity_decrease < expected_decrease - tolerance:
                        logger.warning(
                            "Withdrawal dual-chain mismatch: EVM USDC arrived (+%s), "
                            "but HyperCore equity decreased by only %s (expected ~%s). "
                            "Before: %s, after: %s",
                            executed_reserve, equity_decrease, expected_decrease,
                            equity_before, remaining_equity,
                        )
                    else:
                        logger.info(
                            "Withdrawal dual-chain verified: EVM USDC arrived (+%s), "
                            "HyperCore equity decreased by %s. Before: %s, after: %s",
                            executed_reserve, equity_decrease,
                            equity_before, remaining_equity,
                        )
                else:
                    logger.info(
                        "Withdrawal dual-chain check (no baseline): "
                        "EVM USDC arrived (+%s), HyperCore equity remaining: %s",
                        executed_reserve, remaining_equity,
                    )
            except Exception as e:
                logger.warning(
                    "Could not verify vault equity after withdrawal: %s", e,
                )

        executed_amount = -executed_reserve  # Negative for sells

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
            "Hypercore vault withdrawal settled: %s USDC withdrawn (planned %s)",
            executed_reserve, planned_reserve,
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
            if self.simulate:
                self._settle_deposit_simulate(web3, state, trade, receipts, stop_on_execution_failure)
            else:
                self._settle_deposit(web3, state, trade, receipts, stop_on_execution_failure)
        else:
            self._settle_withdrawal(web3, state, trade, receipts, stop_on_execution_failure)
