"""Lagoon vault integration."""

import datetime
import logging
from dataclasses import dataclass
from decimal import Decimal
from pprint import pformat
from types import NoneType
from typing import Callable, Iterable, Optional

from eth_abi import decode
from web3 import Web3
from web3.contract.contract import ContractFunction
from web3.exceptions import ContractLogicError

from eth_defi.abi import get_abi_by_filename
from eth_defi.compat import native_datetime_utc_fromtimestamp, native_datetime_utc_now
from eth_defi.confirmation import wait_and_broadcast_multiple_nodes_mev_blocker
from eth_defi.erc_4626.vault_protocol.lagoon.analysis import \
    analyse_vault_flow_in_settlement
from eth_defi.erc_4626.vault_protocol.lagoon.vault import (
    DEFAULT_LAGOON_POST_VALUATION_GAS, DEFAULT_LAGOON_SETTLE_GAS, LagoonVault)
from eth_defi.hotwallet import HotWallet
from eth_defi.provider.anvil import is_anvil
from eth_defi.provider.broken_provider import get_almost_latest_block_number
from eth_defi.provider.mev_blocker import MEVBlockerProvider
from eth_defi.provider.receipt import wait_for_transaction_receipt_robust
from eth_defi.revert_reason import extract_revert_data
from eth_defi.token import fetch_erc20_details
from eth_defi.trace import assert_transaction_success_with_explanation
from eth_typing import BlockIdentifier, HexAddress
from tradingstrategy.chain import ChainId

from tradeexecutor.ethereum.address_sync_model import AddressSyncModel
from tradeexecutor.ethereum.lagoon.tx import LagoonTransactionBuilder
from tradeexecutor.state.balance_update import (BalanceUpdate,
                                                BalanceUpdateCause,
                                                BalanceUpdatePositionType)
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.state.sync import BalanceEventRef
from tradeexecutor.state.types import (BlockNumber, JSONHexAddress, Percent,
                                       USDollarPrice)
from tradeexecutor.strategy.interest import sync_interests
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.sync_model import OnChainBalance
from tradeexecutor.strategy.trading_strategy_universe import \
    TradingStrategyUniverse

logger = logging.getLogger(__name__)


def _load_lagoon_guard_v0_error_selectors() -> dict[str, bytes]:
    """Load GuardV0 custom-error selectors from the packaged Lagoon ABI.

    Keep the error parameter definitions in ``eth_defi/abi/guard/LagoonLib``:
    they are part of the deployed GuardV0 contract, not trade-executor's local
    protocol definition.
    """
    guard_abi = get_abi_by_filename("guard/LagoonLib.json")["abi"]
    return {
        entry["name"]: Web3.keccak(
            text=f'{entry["name"]}({",".join(input_["type"] for input_ in entry["inputs"])})'
        )[:4]
        for entry in guard_abi
        if entry["type"] == "error"
    }


# GuardV0 custom-error selectors derived from the packaged LagoonLib ABI.
LAGOON_GUARD_V0_ERROR_SELECTORS = _load_lagoon_guard_v0_error_selectors()

# Selector for a queue whose GuardV0 gross settlement flow exceeds its cap.
LAGOON_SETTLEMENT_LIMIT_EXCEEDED_SELECTOR = LAGOON_GUARD_V0_ERROR_SELECTORS[
    "LagoonSettlementLimitExceeded"
]

# Selector for an otherwise eligible queue during the GuardV0 cooldown.
LAGOON_SETTLEMENT_COOLDOWN_ACTIVE_SELECTOR = LAGOON_GUARD_V0_ERROR_SELECTORS[
    "LagoonSettlementCooldownActive"
]

# TradingStrategyModuleV0 releases predating the GuardV0 safety configuration getter.
LAGOON_UNLIMITED_LEGACY_MODULE_VERSIONS = frozenset({
    "v0.1.0",
    "v0.1.1",
    "v0.2",
    "v0.3",
    "v0.4",
})


class LagoonSettlementSafetyError(Exception):
    """Base class for executor-side GuardV0 settlement-safety failures."""


class LagoonUnsupportedTradingStrategyModuleVersion(LagoonSettlementSafetyError):
    """Raised when a module version has no explicit GuardV0 compatibility path."""


class LagoonSettlementSafetyConfigurationError(LagoonSettlementSafetyError):
    """Raised when an enabled GuardV0 policy cannot safely be validated."""


class LagoonSettlementPreflightError(LagoonSettlementSafetyError):
    """Raised when settlement simulation cannot obtain a valid on-chain input."""


class LagoonFrozenPositionSettlementError(LagoonSettlementSafetyError):
    """Raised when frozen strategy positions make a Lagoon NAV unsafe to post."""


@dataclass(frozen=True)
class LagoonSettlementPreflight:
    """GuardV0 simulation result used to decide whether to broadcast settlement.

    ``should_settle`` permits the asset-manager transaction. A limit breach is
    represented by ``manual_settlement_required`` with Guard-reported raw
    amounts; a cooldown has ``next_settlement_timestamp`` instead. The raw
    queue balances are retained for diagnostics and state accounting when no
    settlement is sent.
    """

    # True only when the asset manager may broadcast settlement automatically.
    should_settle: bool

    # True for an over-cap queue that requires a direct Safe-governance settlement.
    manual_settlement_required: bool = False

    # GuardV0-reported gross settlement amount that exceeded the configured cap.
    actual_amount_raw: int | None = None

    # GuardV0-reported inclusive maximum gross settlement amount.
    max_amount_raw: int | None = None

    # Unix timestamp at which a cooldown-deferred settlement may be retried.
    next_settlement_timestamp: int | None = None

    # Current pending underlying deposit balance in the Lagoon Silo, in raw units.
    pending_deposit_raw: int = 0

    # Current pending redemption share balance in the Lagoon Silo, in raw units.
    pending_redemption_shares_raw: int = 0


def fetch_lagoon_guard_v0_settlement_metadata(
    vault: LagoonVault,
) -> dict[str, bool | int | Decimal | None] | None:
    """Read the live GuardV0 automatic-settlement policy for frontend metadata.

    The returned ``daily_automatic_settlement_limit`` is GuardV0's maximum
    gross underlying-token movement for one automatic settlement. GuardV0
    applies ``settlement_cooldown_seconds`` after each successful non-empty
    settlement (normally 24 hours), which makes this a practical daily limit.
    It is *not* a net deposit-minus-redemption limit: see GuardV0's gross-flow
    calculation in ``.claude/docs/lagoon-treasury-settlement.md``.

    ``None`` means that this vault has no module, or has an older or unsupported
    module and therefore no GuardV0 policy to display. An enabled policy whose
    vault topology does not match the Lagoon vault is deliberately also omitted:
    displaying a limit for a different vault would mislead the frontend.

    :return:
        Policy fields for ``on_chain_data.smart_contracts`` (serialised by the
        metadata JSON encoder), or ``None`` when GuardV0 is unavailable.
    """
    if vault.trading_strategy_module_address is None:
        # A Lagoon vault can be used without executor-managed Safe automation.
        return None

    module = vault.trading_strategy_module
    try:
        module_version = module.functions.getTradingStrategyModuleVersion().call()
    except (ContractLogicError, ValueError):
        # GuardV0 was introduced after these releases, which do not expose the
        # version getter or the settlement-safety configuration getter.
        return None

    # GuardV0 settlement metadata has the same explicit compatibility boundary
    # as LagoonVaultSyncModel._has_lagoon_settlement_safety(). Do not infer
    # support for a future GuardV0/module release from a similarly shaped ABI.
    if module_version != "v0.5":
        return None

    try:
        (
            allowed,
            limit_enabled,
            asset,
            pending_silo,
            max_settlement_amount_raw,
            settlement_cooldown_seconds,
            _last_settlement_timestamp,
            next_settlement_timestamp,
        ) = module.functions.getLagoonSettlementSafetyConfig(vault.address).call()
    except (ContractLogicError, ValueError) as e:
        logger.warning(
            "Could not read GuardV0 settlement metadata for Lagoon vault %s from module %s: %s",
            vault.address,
            vault.trading_strategy_module_address,
            e,
        )
        return None

    # GuardV0 configures a module globally but enables it per Lagoon vault.
    # Refuse to publish a plausible-looking cap if the getter resolves another
    # asset or Silo; the executor takes the same fail-closed approach before
    # it submits an automated settlement.
    if asset.lower() != vault.underlying_token.address.lower() or pending_silo.lower() != vault.silo_address.lower():
        logger.warning(
            "Not exposing GuardV0 settlement metadata for Lagoon vault %s: "
            "the policy targets asset %s and Silo %s",
            vault.address,
            asset,
            pending_silo,
        )
        return None

    daily_limit_enabled = bool(allowed and limit_enabled)
    return {
        # GuardV0 lets the frontend identify the policy which produced these values.
        "guard_version": "GuardV0",
        # GuardV0 applies a cap only after both its vault permission and limit are enabled.
        "daily_automatic_settlement_limit_enabled": daily_limit_enabled,
        # The cap is meaningful only while GuardV0 actively applies its daily policy.
        "daily_automatic_settlement_limit": (
            vault.underlying_token.convert_to_decimals(max_settlement_amount_raw)
            if daily_limit_enabled
            else None
        ),
        # Raw units avoid any loss of precision in frontend integrations.
        "daily_automatic_settlement_limit_raw": max_settlement_amount_raw if daily_limit_enabled else None,
        # GuardV0's post-settlement wait; together with the cap this defines the daily capacity.
        "settlement_cooldown_seconds": settlement_cooldown_seconds,
        # Unix timestamp. A zero value means that no automatic settlement has started a cooldown.
        "next_automatic_settlement_timestamp": next_settlement_timestamp,
    }


def _get_position_vault_log_suffix(position) -> str:
    """Get optional vault name suffix for position freshness logs."""
    vault_name = position.pair.get_vault_name()
    if not vault_name:
        metadata = position.pair.get_token_metadata()
        if isinstance(metadata, dict):
            vault_name = metadata.get("vault_name")
        else:
            vault_name = getattr(metadata, "vault_name", None)

    if vault_name:
        return f", vault={vault_name}"

    return ""


def _transact_anvil_sequentially(
    web3,
    hot_wallet: HotWallet,
    txs: list[tuple[ContractFunction, int]],
    *,
    timeout: int = 120,
) -> list:
    """Broadcast a small ordered tx batch directly on Anvil.

    Anvil forks are prone to returning transient ``nonce too low`` errors when
    we feed already-signed sequential transactions through the multi-node retry
    broadcast helper that is designed for real RPC infrastructure. In unit
    tests we only have a single local Anvil node, so we re-sync the wallet
    nonce once against the local chain and then sign, submit and confirm the
    transactions one by one against the active provider.
    """
    provider = getattr(web3, "provider", None)
    active_provider = getattr(provider, "get_active_provider", lambda: provider)()
    direct_web3 = Web3(active_provider) if active_provider is not None else web3
    hot_wallet.sync_nonce(direct_web3)

    tx_hashes = []
    for bound_func, gas_limit in txs:
        signed_tx = hot_wallet.sign_bound_call_with_new_nonce(
            bound_func,
            tx_params={"gas": gas_limit},
            web3=direct_web3,
            fill_gas_price=True,
        )
        tx_hash = direct_web3.eth.send_raw_transaction(signed_tx.rawTransaction)
        receipt = direct_web3.eth.wait_for_transaction_receipt(tx_hash, timeout=timeout)
        if receipt["status"] != 1:
            assert_transaction_success_with_explanation(direct_web3, tx_hash)
        tx_hashes.append(tx_hash)

    return tx_hashes


class LagoonVaultSyncModel(AddressSyncModel):
    """Update Lagoon vault balances.

    - We do specific NAV update and settlement cycle to update
    """

    def __init__(
        self,
        vault: LagoonVault,
        hot_wallet: HotWallet | None,
        extra_gnosis_gas: int = 500_000,
        valuation_data_freshness=datetime.timedelta(hours=4),
        min_nav_change_update: Percent=0.005,
        unit_testing=False,
        calculate_valuation_func: Callable[..., USDollarPrice] | None = None,
        abort_lagoon_settlement_on_frozen_positions: bool = False,
    ):
        """
        :param extra_gnosis_gas:
            How much extra gas we need for transactions going through Gnosis machinery.

            Because of estimation problems.

        :param valuation_data_freshness:
            Crash is valuation data is older than this.

            Abort posting new valuations to onchain the valuation is too old.

        :param unit_testing:
            Don't use minor regorg safe latest block protection.

            Needed for tenderly.

        :param min_nav_change_update:
            Deprecated compatibility parameter. It is ignored: post-valuation
            Lagoon treasury syncs always post the current NAV before deciding
            whether the investor queue may settle.

        :param calculate_valuation_func:
            Optional strategy-specific NAV calculation function.

            When set, :py:meth:`calculate_valuation` delegates to this
            callable instead of using the default
            ``portfolio.get_net_asset_value()``.

            This is needed for strategies where external systems (e.g.
            FreqTrade managing GMX positions) move funds in and out of
            the Safe without the trade engine knowing, making the
            portfolio's reserve balance stale.

            See :py:func:`tradeexecutor.exchange_account.gmx.create_gmx_vault_valuation_func`
            for the GMX-specific implementation.

        :param abort_lagoon_settlement_on_frozen_positions:
            Safety feature for live trading.

            When enabled, abort Lagoon settlement before NAV calculation if the
            strategy has frozen positions. This forces operators to resolve the
            frozen positions manually first and avoids miscounting or double
            counting capital in the posted NAV.
        """
        assert isinstance(vault, LagoonVault), f"Got {type(vault)} instead of LagoonVault"
        if hot_wallet is not None:
            # We can do initial setup without hot wallet
            assert isinstance(hot_wallet, HotWallet), f"Got {type(hot_wallet)} instead of HotWallet"
        self.vault = vault
        self.hot_wallet = hot_wallet
        self.extra_gnosis_gas = extra_gnosis_gas
        self.valuation_data_freshness = valuation_data_freshness
        self.min_nav_change_update = min_nav_change_update
        self.anvil = is_anvil(self.web3)  # Running test mode
        self.unit_testing = unit_testing  #
        self.calculate_valuation_func = calculate_valuation_func
        self.abort_lagoon_settlement_on_frozen_positions = abort_lagoon_settlement_on_frozen_positions
        assert vault.trading_strategy_module, "LagoonVault.trading_strategy_module initialisation param not set - needed to run the sync model properly"
        # assert isinstance(self.web3.provider, MEVBlockerProvider), f"This sync model needs MEVBlockerProvider, got {type(self.web3.provider)}"

    def __repr__(self):
        return f"<LagoonVaultSyncModel for vault {self.vault.name} ({self.vault_address})>"

    @property
    def web3(self):
        return self.vault.web3

    @property
    def portfolio_address(self) -> HexAddress:
        return self.vault.spec.vault_address

    @property
    def vault_address(self) -> HexAddress:
        return self.vault.address

    @property
    def chain_id(self) -> ChainId:
        return ChainId(self.vault.spec.chain_id)

    def has_async_deposits(self):
        return True

    def _check_frozen_positions_for_settlement(
        self,
        state: State,
        *,
        post_valuation: bool,
    ) -> None:
        """Abort Lagoon settlement before NAV calculation if frozen positions exist."""
        if not post_valuation:
            return

        if not self.abort_lagoon_settlement_on_frozen_positions:
            return

        frozen_count = len(state.portfolio.frozen_positions)
        if frozen_count == 0:
            return

        raise LagoonFrozenPositionSettlementError(
            "Lagoon settlement safety feature aborted settlement because the strategy has "
            f"{frozen_count} frozen position(s). Resolve frozen positions manually before "
            "calculating NAV to avoid miscounting or double counting capital."
        )

    def get_hot_wallet(self) -> Optional[HotWallet]:
        return self.hot_wallet

    def get_key_address(self) -> Optional[str]:
        return self.vault.vault_address

    def get_main_address(self) -> Optional[JSONHexAddress]:
        return self.vault.vault_address

    def get_token_storage_address(self) -> Optional[str]:
        return self.vault.safe_address

    def get_safe_latest_block(self) -> int:
        if self.anvil or self.unit_testing:
            # On Anvil tests, we need to always follow the latest block
            # Set self.unit_testing when using Tenderly
            return self.web3.eth.block_number
        else:
            # Leave room for minor reorg of 1-2 blocks
            return get_almost_latest_block_number(self.web3)

    def create_transaction_builder(self) -> LagoonTransactionBuilder:
        return LagoonTransactionBuilder(self.vault, self.hot_wallet, self.extra_gnosis_gas)

    def sync_initial(
        self,
        state: State,
        reserve_asset: AssetIdentifier | None = None,
        reserve_token_price: USDollarPrice | None = None,
        **kwargs,
    ):
        """Set ups sync starting point"""
        super().sync_initial(
            state=state,
            reserve_asset=reserve_asset,
            reserve_token_price=reserve_token_price,
        )

        deployment = state.sync.deployment
        deployment.vault_token_name = self.vault.name
        deployment.vault_token_symbol = self.vault.symbol

    def sync_interests(
        self,
        timestamp: datetime.datetime,
        state: State,
        universe: TradingStrategyUniverse,
        pricing_model: PricingModel,
    ) -> list[BalanceUpdate]:
        """Sync interests events.

        - Read interest gained onchain

        - Apply it to your state

        :return:
            The list of applied interest change events
        """

        return sync_interests(
            web3=self.web3,
            wallet_address=self.get_token_storage_address(),
            timestamp=timestamp,
            state=state,
            universe=universe,
            pricing_model=pricing_model,
        )

    def fetch_onchain_balances(
        self,
        assets: list[AssetIdentifier],
        filter_zero=True,
        block_identifier: BlockIdentifier = None,
    ) -> Iterable[OnChainBalance]:
        # Use parent's multichain-aware implementation which routes
        # assets to the correct chain's web3 connection (e.g. CCTP bridge
        # positions hold destination-chain USDC that must be queried on
        # that chain, not the vault's home chain).
        if block_identifier is None:
            block_identifier = self.get_safe_latest_block()

        return super().fetch_onchain_balances(
            sorted(assets, key=lambda a: a.address),
            filter_zero=filter_zero,
            block_identifier=block_identifier,
        )

    def calculate_valuation(self, state: State, *, block_number: int | None = None) -> USDollarPrice:
        """Calculate NAV of the vault.

        - Calculate the equity of all assets in the vault
        - Check that we do not use stale data
        - If a strategy-specific ``calculate_valuation_func`` was provided
          (e.g. for GMX), delegate to it; otherwise use the default
          ``portfolio.get_net_asset_value()``

        The freshness check always runs regardless of which valuation
        path is used.

        :param state:
            Current strategy state.
        :param block_number:
            Block number at which to read on-chain state.
            Forwarded to ``calculate_valuation_func`` when set.
        """

        now = native_datetime_utc_now()
        all_positions = list(state.portfolio.get_open_and_frozen_positions())
        logger.info(
            "calculate_valuation() freshness check: %d open/frozen positions, now=%s, threshold=%s",
            len(all_positions),
            now,
            self.valuation_data_freshness,
        )
        for p in all_positions:
            if p.get_quantity() != 0:
                # Frozen positions may have quantity of 0 (failed open trades) and cannot have value
                valued_at = p.get_last_valued_at()
                updated_ago = now - valued_at
                last_event = p.valuation_updates[-1] if p.valuation_updates else None

                logger.info(
                    "Freshness check position #%d %s: valued_at=%s, ago=%s, valuation_updates=%d, last_pricing_at=%s, kind=%s, qty=%s%s",
                    p.position_id,
                    p.pair.base.token_symbol,
                    valued_at,
                    updated_ago,
                    len(p.valuation_updates),
                    p.last_pricing_at,
                    p.pair.kind.value,
                    p.get_quantity(),
                    _get_position_vault_log_suffix(p),
                )

                # Try to dump as much as possible information for diagnostics
                assert updated_ago < self.valuation_data_freshness, f"The last valuation of this position is too old for us to comfortably update the onchain share price. Position {p}. Now: {now}, updated at: {valued_at}, diff: {updated_ago}, threshold: {self.valuation_data_freshness}, last valuation event: {last_event}"
            else:
                logger.info(
                    "Freshness check position #%d %s: skipped (quantity=0)%s",
                    p.position_id,
                    p.pair.base.token_symbol,
                    _get_position_vault_log_suffix(p),
                )

        if self.calculate_valuation_func is not None:
            valuation = self.calculate_valuation_func(state, block_number=block_number)
            return valuation + state.portfolio.get_vault_settlement_pending_value()
        else:
            return state.portfolio.get_net_asset_value(include_interest=True)

    def _mark_treasury_sync_completed(
        self,
        treasury_sync,
        strategy_cycle_ts: datetime.datetime,
        block_number: int,
        pending_redemptions: Decimal | None = None,
        share_count: Decimal | None = None,
    ) -> None:
        """Mark Lagoon treasury as synced even when no settlement was needed."""
        treasury_sync.last_updated_at = native_datetime_utc_now()
        treasury_sync.last_cycle_at = strategy_cycle_ts
        treasury_sync.last_block_scanned = block_number
        if pending_redemptions is not None:
            treasury_sync.pending_redemptions = float(pending_redemptions)
        if share_count is not None:
            treasury_sync.share_count = share_count

    def _has_lagoon_settlement_safety(self) -> bool:
        """Check whether the supported module enables GuardV0 settlement safety."""
        module = self.vault.trading_strategy_module
        try:
            module_version = module.functions.getTradingStrategyModuleVersion().call()
        except (ContractLogicError, ValueError):
            # This is the existing pre-version-getter module behaviour.
            module_version = "v0.1.0"

        # Older modules have no GuardV0 getter, so retain their unlimited
        # settlement behaviour instead of probing an ABI they do not expose.
        if module_version in LAGOON_UNLIMITED_LEGACY_MODULE_VERSIONS:
            return False

        # A new contract release must opt in here explicitly. Guessing support
        # from a version shape or a failed feature probe could bypass GuardV0.
        if module_version != "v0.5":
            raise LagoonUnsupportedTradingStrategyModuleVersion(
                f"Unsupported Lagoon TradingStrategyModuleV0 version {module_version!r}. "
                "Add an explicit settlement-safety implementation before using this version."
            )

        try:
            (
                allowed,
                limit_enabled,
                asset,
                pending_silo,
                _max_settlement_amount_raw,
                _settlement_cooldown,
                _last_settlement_timestamp,
                _next_settlement_timestamp,
            ) = module.functions.getLagoonSettlementSafetyConfig(
                self.vault.address,
            ).call()
        except Exception as e:
            raise LagoonSettlementSafetyConfigurationError(
                f"Could not read Lagoon settlement safety configuration from supported "
                f"TradingStrategyModuleV0 {self.vault.trading_strategy_module_address}"
            ) from e

        # GuardV0 may be deployed but disabled for this vault: this is the
        # explicit unlimited-policy scenario, not a configuration failure.
        if not limit_enabled:
            return False

        # An enabled GuardV0 policy must apply to this exact Lagoon vault,
        # underlying asset and pending Silo. Fail closed on any mismatch.
        if not allowed:
            raise LagoonSettlementSafetyConfigurationError(
                f"Lagoon settlement safety policy is not enabled for vault {self.vault.address}"
            )

        if asset.lower() != self.vault.underlying_token.address.lower():
            raise LagoonSettlementSafetyConfigurationError(
                f"Lagoon settlement safety asset mismatch: GuardV0 measures {asset}, "
                f"but vault underlying is {self.vault.underlying_token.address}"
            )

        if pending_silo.lower() != self.vault.silo_address.lower():
            raise LagoonSettlementSafetyConfigurationError(
                f"Lagoon settlement safety Silo mismatch: GuardV0 measures {pending_silo}, "
                f"but vault Silo is {self.vault.silo_address}"
            )

        return True

    def _preflight_lagoon_settlement(
        self,
        settle_func: ContractFunction,
    ) -> LagoonSettlementPreflight:
        """Simulate a non-empty GuardV0 settlement before spending gas."""
        flow_manager = self.vault.get_flow_manager()
        call_web3 = self.web3

        if self.anvil or self.unit_testing:
            # Consecutive reads on an Anvil fork must use the same provider as
            # the transaction broadcast. A fallback provider can briefly lag
            # locally mined NAV updates.
            provider = getattr(self.web3, "provider", None)
            active_provider = getattr(provider, "get_active_provider", lambda: provider)()
            call_web3 = Web3(active_provider) if active_provider is not None else self.web3

            def fetch_raw_balance(token) -> int:
                data = token.contract.functions.balanceOf(self.vault.silo_address)._encode_transaction_data()
                result = call_web3.eth.call({"to": token.address, "data": data}, block_identifier="latest")
                if len(result) != 32:
                    raise LagoonSettlementPreflightError(
                        f"Invalid ERC-20 balanceOf response from {token.address}: {result.hex()}"
                    )
                return int.from_bytes(result, byteorder="big")

            pending_deposit_raw = fetch_raw_balance(self.vault.underlying_token)
            pending_redemption_shares_raw = fetch_raw_balance(self.vault.share_token)
        else:
            # In production use the normal Lagoon flow manager so reads retain
            # the configured multi-provider retry and failover behaviour.
            pending_deposit_raw = self.vault.underlying_token.convert_to_raw(
                flow_manager.fetch_pending_deposit("latest"),
            )
            pending_redemption_shares_raw = self.vault.share_token.convert_to_raw(
                flow_manager.fetch_pending_redemption("latest"),
            )

        # Empty investor queues never need settleDeposit(). The caller has
        # already posted NAV, so this is deliberately a NAV-only cycle.
        if pending_deposit_raw == 0 and pending_redemption_shares_raw == 0:
            return LagoonSettlementPreflight(
                should_settle=False,
                pending_deposit_raw=pending_deposit_raw,
                pending_redemption_shares_raw=pending_redemption_shares_raw,
            )

        # Legacy modules and explicitly disabled GuardV0 policies preserve the
        # historical automatic settlement path for a non-empty queue.
        if not self._has_lagoon_settlement_safety():
            return LagoonSettlementPreflight(
                should_settle=True,
                pending_deposit_raw=pending_deposit_raw,
                pending_redemption_shares_raw=pending_redemption_shares_raw,
            )

        assert self.hot_wallet is not None
        try:
            # Simulate the exact wrapped asset-manager call after NAV posting.
            # On Anvil, call_web3 is pinned to the provider that observed the
            # locally mined NAV transaction. Production uses the normal
            # multi-provider connection and retains its failover behaviour.
            call_web3.eth.call({
                "from": self.hot_wallet.address,
                "to": settle_func.address,
                "data": settle_func._encode_transaction_data(),
                "gas": DEFAULT_LAGOON_SETTLE_GAS,
            })
        except Exception as e:
            # Provider envelopes are not uniform. Recover the raw custom-error
            # payload before classifying only GuardV0's expected deferrals.
            revert_data = extract_revert_data(e)
            if revert_data is None:
                raise

            selector = revert_data[:4]
            if selector == LAGOON_SETTLEMENT_LIMIT_EXCEEDED_SELECTOR:
                # GuardV0 measures gross movement, not net cash movement. Its
                # amount check precedes the cooldown check, so this remains a
                # manual-Safe scenario even during an active cooldown.
                actual_amount_raw, max_amount_raw = decode(["uint256", "uint256"], revert_data[4:])
                return LagoonSettlementPreflight(
                    should_settle=False,
                    manual_settlement_required=True,
                    actual_amount_raw=actual_amount_raw,
                    max_amount_raw=max_amount_raw,
                    pending_deposit_raw=pending_deposit_raw,
                    pending_redemption_shares_raw=pending_redemption_shares_raw,
                )
            if selector == LAGOON_SETTLEMENT_COOLDOWN_ACTIVE_SELECTOR:
                # The queue is otherwise permitted but must remain pending
                # until GuardV0 allows the next non-zero settlement.
                _current_timestamp, next_settlement_timestamp = decode(["uint256", "uint256"], revert_data[4:])
                return LagoonSettlementPreflight(
                    should_settle=False,
                    next_settlement_timestamp=next_settlement_timestamp,
                    pending_deposit_raw=pending_deposit_raw,
                    pending_redemption_shares_raw=pending_redemption_shares_raw,
                )
            # Liquidity, access-control and unexpected module reverts are not
            # GuardV0 deferrals; retain fail-fast behaviour for those faults.
            raise

        # A successful simulation is the only guarded scenario that may spend
        # gas on the real asset-manager settlement transaction.
        return LagoonSettlementPreflight(
            should_settle=True,
            pending_deposit_raw=pending_deposit_raw,
            pending_redemption_shares_raw=pending_redemption_shares_raw,
        )

    def _log_manual_lagoon_settlement_required(
        self,
        preflight: LagoonSettlementPreflight,
    ) -> None:
        """Tell the operator to settle an oversized GuardV0 queue through Safe governance."""
        assert preflight.actual_amount_raw is not None
        assert preflight.max_amount_raw is not None
        underlying_token = self.vault.underlying_token
        logger.error(
            "Lagoon automated settlement skipped: direct Safe-governance settlement required. "
            "chain=%d vault=%s safe=%s module=%s pending_deposit=%s %s "
            "pending_redemption_shares_raw=%d gross_flow=%s %s (%d raw) cap=%s %s (%d raw). "
            "NAV update succeeded and both queues remain pending.",
            self.chain_id.value,
            self.vault.address,
            self.vault.safe_address,
            self.vault.trading_strategy_module_address,
            underlying_token.convert_to_decimals(preflight.pending_deposit_raw),
            underlying_token.symbol,
            preflight.pending_redemption_shares_raw,
            underlying_token.convert_to_decimals(preflight.actual_amount_raw),
            underlying_token.symbol,
            preflight.actual_amount_raw,
            underlying_token.convert_to_decimals(preflight.max_amount_raw),
            underlying_token.symbol,
            preflight.max_amount_raw,
        )

    def sync_treasury(
        self,
        strategy_cycle_ts: datetime.datetime,
        state: State,
        supported_reserves: Optional[list[AssetIdentifier]] = None,
        end_block: BlockNumber | NoneType = None,
        post_valuation=False,
    ) -> list[BalanceUpdate]:
        """Sync Lagoon treasury.

        - Calcualte NAV
        - Post it onchain if `post_valuation` is true
        - Will crash if the valuation or settle tx broadcast fails

        :param post_valuation:
            Doesn't do anything unless the post valuation is true.

            Because to get deposit events, we need to settle with a new valuation posted onchain.
        """

        web3 = self.web3
        sync = state.sync

        vault = self.vault
        treasury_sync = sync.treasury
        portfolio = state.portfolio

        assert sync.is_initialised(), f"Vault sync not initialised: {sync}\nPlease run trade-executor init command"

        match len(portfolio.reserves):
            case 1:
                # We have already run sync once
                logger.info("Reserve previously synced at %s", treasury_sync.last_updated_at)
                reserve_position = portfolio.get_default_reserve_position()
                reserve_asset = reserve_position.asset
            case 0:
                # Tabula rasa sync, need to create initial reserve position
                logger.info("Creating initial reserve")
                assert supported_reserves is not None
                reserve_asset = supported_reserves[0]
                state.portfolio.initialise_reserves(reserve_asset, reserve_token_price=1.0)
                reserve_position = portfolio.get_default_reserve_position()
            case _:
                raise NotImplementedError("Multireserve not supported")

        assert reserve_asset.is_stablecoin()

        reserve_token = fetch_erc20_details(
            web3,
            reserve_asset.address,
            cache=self.vault.token_cache,
            chain_id=reserve_asset.chain_id,
        )

        self._check_frozen_positions_for_settlement(
            state,
            post_valuation=post_valuation,
        )

        # Reconcile reserves from on-chain before calculating NAV.
        #
        # Exchange account positions (e.g. GMX) transfer USDC from the Safe
        # via sendTokens() in multicall — outside the trade engine.
        # This means reserve_position.quantity can be stale: it still
        # reflects the pre-transfer balance while the USDC has already
        # left the Safe.
        #
        # The exchange account value function (e.g. create_gmx_account_value_func)
        # only returns capital locked in exchange positions, NOT free USDC
        # in the Safe — so the Safe's actual USDC balance is the correct
        # reserve component for NAV.  Without this reconciliation,
        # calculate_valuation() double-counts the transferred USDC
        # (once in stale reserves, once in the exchange account position),
        # inflating the NAV and mispricing deposits.
        #
        # See README-GMX-Lagoon.md for the full token flow.
        block_number = self.get_safe_latest_block()
        onchain_balance = reserve_token.fetch_balance_of(
            self.get_token_storage_address(),
            block_identifier=block_number,
        )
        if reserve_position.quantity != onchain_balance:
            logger.warning(
                "Reserve balance mismatch: portfolio=%s, on-chain=%s. "
                "Updating to on-chain value before NAV calculation.",
                reserve_position.quantity,
                onchain_balance,
            )
            reserve_position.quantity = onchain_balance

        valuation = self.calculate_valuation(state, block_number=block_number)

        if not post_valuation:
            logger.warning("LagoonVaultSyncModel.sync_treasury() called with post_valuation=False")
            return []

        assert self.hot_wallet, "asset_manager HotWallet needed in order to sync Lagoon vault"

        old_balance = reserve_token.fetch_balance_of(self.get_token_storage_address())

        # NAV must be fresh on every requested post-valuation cycle. GuardV0
        # decides investor settlement separately and never suppresses this tx.
        logger.info("Posting new Lagoon valuation: %f USD", valuation)
        valuation_decimal = Decimal(valuation)
        valuation_func = vault.post_new_valuation(valuation_decimal)

        if self.anvil or self.unit_testing:
            logger.info("Broadcasting Lagoon valuation on Anvil")
            nav_tx_hash = _transact_anvil_sequentially(
                web3,
                self.hot_wallet,
                [(valuation_func, DEFAULT_LAGOON_POST_VALUATION_GAS)],
            )[0]
        else:
            signed_nav_tx = self.hot_wallet.sign_bound_call_with_new_nonce(
                valuation_func,
                tx_params={"gas": DEFAULT_LAGOON_POST_VALUATION_GAS},
                web3=web3,
                fill_gas_price=True,
            )
            wait_and_broadcast_multiple_nodes_mev_blocker(
                web3.provider,
                [signed_nav_tx],
            )
            nav_tx_hash = signed_nav_tx.hash

        nav_receipt = wait_for_transaction_receipt_robust(
            web3,
            nav_tx_hash,
            confirmation_block_count=2,
            extra_sleep=2.0,
            allow_partial_visibility_after_timeout=True,
        )
        if nav_receipt["status"] != 1:
            assert_transaction_success_with_explanation(web3, nav_tx_hash, func=valuation_func)

        # Do not pre-sign settlement: GuardV0 is evaluated against the state
        # after this confirmed NAV transaction.
        nav_block_number = nav_receipt["blockNumber"]

        logger.info("Preparing to settle Lagoon")

        # This is an operator liquidity warning only. GuardV0 preflight below
        # remains the authoritative settlement decision.
        block_number = web3.eth.block_number
        pending_shares = vault.get_flow_manager().fetch_pending_redemption(block_number)

        if pending_shares > 0:
            # Calculate how much USDC is needed for redemptions
            total_assets = vault.fetch_total_assets(block_number)
            total_supply = vault.fetch_total_supply(block_number)

            if total_supply > 0:
                share_price = total_assets / total_supply
                required_usdc = pending_shares * share_price

                # Check actual USDC balance in the Safe
                safe_usdc_balance = reserve_token.fetch_balance_of(vault.safe_address, block_number)

                logger.info(
                    "Redemption check: pending shares=%s, share price=%s, required USDC=%s, Safe balance=%s",
                    pending_shares,
                    share_price,
                    required_usdc,
                    safe_usdc_balance,
                )

                if required_usdc > safe_usdc_balance:
                    deficit = required_usdc - safe_usdc_balance
                    logger.warning(
                        "⚠️  INSUFFICIENT LIQUID USDC FOR REDEMPTIONS ⚠️\n"
                        "Pending redemptions: %s shares\n"
                        "Current share price: %s USDC/share\n"
                        "Required USDC: %s\n"
                        "Available in Safe: %s\n"
                        "Deficit: %s USDC\n"
                        "Redemptions will NOT be processed in this settlement cycle.\n"
                        "Consider redeeming from vault positions (IPOR/Morpho) before next settlement.",
                        pending_shares,
                        share_price,
                        required_usdc,
                        safe_usdc_balance,
                        deficit,
                    )

        settle_func = vault.settle_via_trading_strategy_module(valuation_decimal)
        preflight = self._preflight_lagoon_settlement(settle_func)

        # Every non-settlement scenario still updates sync metadata. In
        # particular, pending redemptions must remain reserved so yield logic
        # cannot allocate cash needed for a deferred or manual settlement.
        if not preflight.should_settle:
            metadata_block_number = web3.eth.block_number
            flow_manager = vault.get_flow_manager()
            pending_redemptions = flow_manager.calculate_underlying_needed_for_redemptions(metadata_block_number)
            share_count = vault.fetch_total_supply(metadata_block_number)

            if preflight.manual_settlement_required:
                # Gross flow is above the GuardV0 cap. The Safe, rather than
                # the guarded asset manager, must deliberately settle it.
                self._log_manual_lagoon_settlement_required(preflight)
            elif preflight.next_settlement_timestamp is not None:
                # Gross flow is within the cap, but a previous non-zero
                # settlement started GuardV0's cooldown.
                next_eligible_at = native_datetime_utc_fromtimestamp(preflight.next_settlement_timestamp)
                logger.info(
                    "Lagoon automated settlement deferred by GuardV0 cooldown until %s. "
                    "The queue will be retried automatically. Pending deposit raw=%d, "
                    "pending redemption shares raw=%d.",
                    next_eligible_at,
                    preflight.pending_deposit_raw,
                    preflight.pending_redemption_shares_raw,
                )
            else:
                # The preflight established that both Silo balances are zero.
                logger.info("Lagoon NAV posted without settlement because the queue is empty")

            self._mark_treasury_sync_completed(
                treasury_sync=treasury_sync,
                strategy_cycle_ts=strategy_cycle_ts,
                block_number=nav_block_number,
                pending_redemptions=pending_redemptions,
                share_count=share_count,
            )
            return []

        # Only the successful guarded or unlimited preflight path reaches this
        # point; all cap and cooldown reverts were handled without gas spend.
        if self.anvil or self.unit_testing:
            logger.info("Broadcasting Lagoon settlement on Anvil after GuardV0 preflight")
            settle_tx_hash = _transact_anvil_sequentially(
                web3,
                self.hot_wallet,
                [(settle_func, DEFAULT_LAGOON_SETTLE_GAS)],
            )[0]
        else:
            signed_settle_tx = self.hot_wallet.sign_bound_call_with_new_nonce(
                settle_func,
                tx_params={"gas": DEFAULT_LAGOON_SETTLE_GAS},
                web3=web3,
                fill_gas_price=True,
            )
            wait_and_broadcast_multiple_nodes_mev_blocker(
                web3.provider,
                [signed_settle_tx],
            )
            settle_tx_hash = signed_settle_tx.hash

        # Let all read RPCs see the settlement receipt and state propagate before
        # analysis. The preceding broadcast helper already confirmed the transaction,
        # so a permanently unhealthy secondary reader may fall back after the timeout.
        settle_receipt = wait_for_transaction_receipt_robust(
            web3,
            settle_tx_hash,
            confirmation_block_count=2,
            extra_sleep=2.0,
            allow_partial_visibility_after_timeout=True,
        )
        if settle_receipt["status"] != 1:
            assert_transaction_success_with_explanation(web3, settle_tx_hash, func=settle_func)

        analysis = analyse_vault_flow_in_settlement(
            vault,
            settle_tx_hash,
        )

        logger.info(
            "Lagoon settled. Settle result is:\n%s",
            pformat(analysis.get_serialiable_diagnostics_data())
        )

        # Post-settlement check: warn if redemptions were pending but not processed
        if analysis.pending_redemptions_shares > 0 and analysis.redeem_events == 0:
            logger.warning(
                "⚠️  REDEMPTIONS WERE NOT PROCESSED ⚠️\n"
                "Pending redemptions remain: %s shares (%s USDC)\n"
                "This typically indicates insufficient liquid USDC in the Safe.\n"
                "Redemption requests will remain pending until the next settlement cycle.",
                analysis.pending_redemptions_shares,
                analysis.pending_redemptions_underlying,
            )

        delta = analysis.get_underlying_diff()
        event_id = portfolio.next_balance_update_id
        portfolio.next_balance_update_id += 1

        # Include our valuation in the other_data diangnostics
        other_data = analysis.get_serialiable_diagnostics_data()
        other_data["valuation"] = valuation
        valuation_with_deposits = valuation + float(delta)
        other_data["valuation_with_deposits"] = valuation_with_deposits

        share_count = vault.fetch_total_supply(analysis.block_number)
        other_data["share_count"] = share_count

        evt = BalanceUpdate(
            balance_update_id=event_id,
            position_type=BalanceUpdatePositionType.reserve,
            cause=BalanceUpdateCause.deposit_and_redemption,
            asset=reserve_position.asset,
            block_mined_at=analysis.timestamp,
            strategy_cycle_included_at=strategy_cycle_ts,
            chain_id=reserve_asset.chain_id,
            old_balance=old_balance,
            quantity=delta,
            owner_address=None,
            tx_hash=analysis.tx_hash.hex(),
            log_index=None,
            position_id=None,
            usd_value=float(delta),  # Assume stablecoin
            notes=f"Lagoon reserve update at tx {analysis.tx_hash.hex()}, block {analysis.block_number:,}",
            block_number=analysis.block_number,
            other_data=other_data,
        )

        # Update reserve position mutable value
        reserve_position.reserve_token_price = float(1)
        reserve_position.last_pricing_at = analysis.timestamp
        reserve_position.last_sync_at = analysis.timestamp
        reserve_position.quantity = analysis.underlying_balance
        reserve_position.add_balance_update_event(evt)

        # Add in the event cross reference list
        ref = BalanceEventRef.from_balance_update_event(evt)
        treasury_sync.balance_update_refs.append(ref)
        treasury_sync.last_block_scanned = analysis.block_number
        treasury_sync.last_updated_at = native_datetime_utc_now()
        treasury_sync.last_cycle_at = strategy_cycle_ts
        treasury_sync.pending_redemptions = float(analysis.pending_redemptions_underlying)
        treasury_sync.share_count = share_count

        logger.info(
            f"Lagoon settlements done, the last block is now {treasury_sync.last_block_scanned:,}\n"
            f"Safe address: {vault.safe_address}, vault address: {vault.vault_address}, silo address: {vault.silo_address}\n"
            f"Settled {analysis.get_underlying_diff()} USD\n"
            f"Non-deposit valuation is {valuation:,.2f} USD, with-deposit valuation is {valuation_with_deposits:,.2f} USD\n"
            f"Pending redemptions {analysis.pending_redemptions_underlying} USD\n"
            f"Share count {share_count} {vault.share_token.symbol}"
        )
        return [evt]
