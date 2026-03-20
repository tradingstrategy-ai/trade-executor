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
    fetch_user_vault_equity,
    wait_for_vault_deposit_confirmation,
)
from eth_defi.hyperliquid.core_writer import (
    MINIMUM_VAULT_DEPOSIT,
    build_hypercore_deposit_multicall,
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
#:
#: The batched Hypercore deposit on a HyperEVM Anvil fork currently needs a bit
#: over 520k gas once routed through TradingStrategyModuleV0.  Keep a modest
#: buffer here so the live split tests do not fail due to an avoidable gas cap.
HYPERCORE_MULTICALL_GAS = 650_000

#: USDC uses 6 decimals on HyperEVM.
USDC_DECIMALS = 6


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

        gas_price_suggestion = estimate_gas_price(self.web3)
        apply_gas(tx_data, gas_price_suggestion)

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
        vault_addr = trade.pair.pool_address
        assert vault_addr, f"No pool_address set for Hypercore vault pair: {trade.pair}"
        return vault_addr

    def _get_raw_usdc_amount(self, trade: TradeExecution) -> int:
        """Get the raw USDC amount (6 decimals) for the trade."""
        planned_reserve = trade.get_planned_reserve()
        return usdc_to_raw(planned_reserve)

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
        deadline = time.time() + timeout
        attempt = 0

        # Initial delay: give the bridge time to process
        time.sleep(poll_interval)

        while True:
            attempt += 1
            current_balance_raw = self._fetch_safe_evm_usdc_balance()
            increase = current_balance_raw - baseline_balance_raw

            if increase >= expected_increase_raw:
                logger.info(
                    "USDC arrived in Safe %s after %d poll(s): "
                    "+%d raw (expected %d raw)",
                    self.safe_address, attempt,
                    increase, expected_increase_raw,
                )
                return increase

            remaining = deadline - time.time()
            if remaining <= 0:
                raise HypercoreWithdrawalVerificationError(
                    f"USDC did not arrive in Safe {self.safe_address} within {timeout}s. "
                    f"Expected increase: {expected_increase_raw} raw, "
                    f"actual increase: {increase} raw, "
                    f"baseline: {baseline_balance_raw} raw, "
                    f"current: {current_balance_raw} raw, "
                    f"after {attempt} poll(s). "
                    f"The HyperCore-to-EVM bridge may be dry or one of the "
                    f"CoreWriter actions (vaultTransfer/transferUsdClass/spotSend) "
                    f"failed silently on HyperCore."
                )

            logger.info(
                "Waiting for USDC in Safe %s: increase %d/%d raw "
                "(%.0fs remaining, poll #%d)",
                self.safe_address, increase, expected_increase_raw,
                remaining, attempt,
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

        # Enforce Hyperliquid minimum vault deposit/withdrawal amounts.
        # Hyperliquid silently rejects vault transfers below the minimum
        # threshold — no error, no event, the USDC just stays in the
        # escrow. The minimum is 5 USDC (MINIMUM_VAULT_DEPOSIT = 5_000_000
        # raw, 6 decimals), determined by reverse-engineering the
        # Hyperliquid web UI. See eth_defi.hyperliquid.core_writer.
        if trade.is_buy():
            assert raw_amount >= MINIMUM_VAULT_DEPOSIT, (
                f"Vault deposit amount {raw_amount / 1e6:.2f} USDC "
                f"({raw_amount} raw) is below the Hyperliquid minimum "
                f"of {MINIMUM_VAULT_DEPOSIT / 1e6:.0f} USDC. "
                f"Hyperliquid silently rejects deposits below this threshold."
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
                tx = self._sign_multicall(fn, notes=f"Hypercore deposit (simulate): {raw_amount} raw USDC")
                return [tx]
            else:
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
                activation_cost_raw = DEFAULT_ACTIVATION_AMOUNT
                logger.info(
                    "Safe %s activated on HyperCore (cost %d raw USDC from Safe)",
                    self.safe_address,
                    activation_cost_raw,
                )

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
            # total vault equity.  Position quantity tracks cumulative USDC
            # deposited; the valuation model then computes per-unit price
            # as equity/quantity so that value = equity.
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

        After the withdrawal multicall tx lands on HyperEVM:

        1. Verifies the EVM receipt.
        2. Polls the Safe's EVM USDC balance until the bridged USDC arrives
           (the ``spotSend`` action bridges USDC from HyperCore to HyperEVM
           with ~2-10 s latency).
        3. Compares actual received USDC against planned amount.
        4. Marks the trade as success or failure.

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
            logger.info(
                "Withdrawal settlement: Safe %s baseline EVM USDC = %d raw, "
                "expected increase = %d raw",
                self.safe_address, baseline_balance_raw, expected_raw,
            )
            equity_before: Decimal | None = None
            vault_address = self._get_vault_address(trade)
            session = self._get_session()
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
            "Hypercore vault withdrawal EVM tx succeeded (%s)",
            withdraw_tx.tx_hash,
        )

        planned_reserve = trade.get_planned_reserve()

        if self.simulate:
            # Simulate mode: mock CoreWriter does not bridge USDC,
            # so skip EVM balance verification and trust planned values.
            executed_reserve = planned_reserve
        else:
            # Poll for USDC arrival on EVM
            try:
                actual_increase_raw = self._wait_for_usdc_arrival(
                    baseline_balance_raw=baseline_balance_raw,
                    expected_increase_raw=expected_raw,
                    timeout=30.0,
                    poll_interval=2.0,
                )
            except HypercoreWithdrawalVerificationError as e:
                logger.error(
                    "Withdrawal verification failed for trade %s: %s",
                    trade.trade_id, e,
                )
                report_failure(ts, state, trade, stop_on_execution_failure)
                return

            executed_reserve = raw_to_usdc(actual_increase_raw)
            if executed_reserve < planned_reserve:
                logger.warning(
                    "Withdrawal partial: received %s USDC but planned %s USDC (trade %s)",
                    executed_reserve, planned_reserve, trade.trade_id,
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
