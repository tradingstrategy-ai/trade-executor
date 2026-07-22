"""Standalone vault test-trade controller helpers.

The command intentionally has no strategy module.  It obtains Lagoon topology
from the deployment artefact and builds a small trading universe for one vault
attempt at a time.
"""

import datetime
import hashlib
import json
import logging
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from eth_account import Account
from eth_defi.abi import ZERO_ADDRESS_STR
from eth_defi.cctp.constants import CHAIN_ID_TO_CCTP_DOMAIN
from eth_defi.cctp.whitelist import CCTPDeployment
from eth_defi.erc_4626.vault import ERC4626Vault
from eth_defi.erc_4626.vault_protocol.lagoon.deployment import (
    LagoonConfig,
    LagoonDeploymentParameters,
    deploy_multichain_lagoon_vault,
)
from eth_defi.provider.anvil import fund_erc20_on_anvil, set_balance
from eth_defi.token import USDC_NATIVE_TOKEN, fetch_erc20_details
from eth_defi.vault.base import VaultSpec
from eth_defi.compat import native_datetime_utc_now
from tradingstrategy.chain import ChainId
from tradingstrategy.candle import Candle
from tradingstrategy.timebucket import TimeBucket

from tradeexecutor.state.identifier import (
    AssetIdentifier,
    TradingPairIdentifier,
    TradingPairKind,
)
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeStatus
from tradeexecutor.ethereum.web3config import get_chain_slug, get_rpc_env_var_name
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.trading_strategy_universe import (
    TradingStrategyUniverse,
    load_partial_data,
)
from tradeexecutor.strategy.universe_model import UniverseOptions

logger = logging.getLogger(__name__)


#: Anvil default account #0. Simulated deployments must not need production
#: signing material, and Web3Config-created forks expose this account.
SIMULATED_LAGOON_PRIVATE_KEY = (
    "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
)


class _SimulatedWhitelistVault(ERC4626Vault):
    """Address-only ERC-4626 descriptor used while deploying the test guard.

    Do not probe the adapter here. Incomplete or unsupported vault adapters are
    a diagnostic result of the later per-vault test, not a reason to abort the
    shared simulated Lagoon deployment.
    """

    def __init__(self, web3, spec: VaultSpec, name: str):
        """Initialise an address-only vault descriptor with a display name."""

        super().__init__(web3, spec)
        self._simulated_name = name

    @property
    def name(self) -> str:
        """Return the downloaded vault name without probing the adapter."""

        return self._simulated_name

    @property
    def symbol(self) -> str:
        """Return a stable placeholder symbol for Lagoon guard deployment."""

        return "VTEST"


def filter_rpc_kwargs_for_vault_specs(
    rpc_kwargs: dict, vault_specs: list[VaultSpec]
) -> dict:
    """Keep only JSON-RPC connections needed by an explicit simulated batch."""

    selected_keys = {
        get_rpc_env_var_name(ChainId(spec.chain_id)).lower() for spec in vault_specs
    }
    return {
        key: value if key in selected_keys else None
        for key, value in rpc_kwargs.items()
    }


def _create_simulated_lagoon_chain_config(
    *,
    chain_id: ChainId,
    primary_chain_id: ChainId,
    selected_chain_ids: list[ChainId],
    web3: Any,
    vault_specs: list[VaultSpec],
    vault_universe: Any,
    account_address: str,
    safe_salt_nonce: int,
) -> LagoonConfig:
    """Build one chain's guard whitelist and optional CCTP permissions."""

    # Use address-only wrappers so guard deployment does not probe incomplete
    # adapters before their individual diagnostic attempt.
    whitelist_vaults = []
    for spec in vault_specs:
        vault = vault_universe.get_by_vault_spec(
            (spec.chain_id, spec.vault_address)
        )
        display_name = getattr(vault, "name", None) or spec.vault_address
        whitelist_vaults.append(_SimulatedWhitelistVault(web3, spec, display_name))

    # The hub may send to all supported satellites; a satellite needs only its
    # return route to the hub.  Unsupported chains still get a local Lagoon Safe
    # so their vault adapters can be tested without a cross-chain route.
    cctp_deployment = None
    if (
        chain_id.value in CHAIN_ID_TO_CCTP_DOMAIN
        and primary_chain_id.value in CHAIN_ID_TO_CCTP_DOMAIN
    ):
        if chain_id == primary_chain_id:
            destinations = [
                destination.value
                for destination in selected_chain_ids
                if destination != primary_chain_id
                and destination.value in CHAIN_ID_TO_CCTP_DOMAIN
            ]
        else:
            destinations = [primary_chain_id.value]
        if destinations:
            cctp_deployment = CCTPDeployment.create_for_chain(
                chain_id=chain_id.value,
                allowed_destinations=destinations,
            )

    return LagoonConfig(
        parameters=LagoonDeploymentParameters(
            underlying=USDC_NATIVE_TOKEN[chain_id.value],
            name="Vault test simulated Lagoon",
            symbol="VTS",
            managementRate=0,
            performanceRate=0,
        ),
        asset_managers=[account_address],
        safe_owners=[account_address],
        safe_threshold=1,
        safe_salt_nonce=safe_salt_nonce,
        cctp_deployment=cctp_deployment,
        any_asset=True,
        erc_4626_vaults=whitelist_vaults,
        satellite_chain=chain_id != primary_chain_id,
        between_contracts_delay_seconds=0,
    )


def _serialise_simulated_lagoon_deployment(
    *,
    result: Any,
    chain_web3: dict[str, Any],
    primary_chain_id: ChainId,
    account_address: str,
) -> tuple["LagoonDeployment", dict]:
    """Translate eth-defi deployment output into runtime and JSON forms."""

    primary_slug = get_chain_slug(primary_chain_id)
    primary = result.deployments[primary_slug]
    satellite_modules = {
        ChainId(
            chain_web3[slug].eth.chain_id
        ): deployment.trading_strategy_module.address
        for slug, deployment in result.deployments.items()
        if deployment.is_satellite
    }
    deployment = LagoonDeployment(
        primary_chain_id=primary_chain_id,
        vault_address=primary.vault.address,
        module_address=primary.trading_strategy_module.address,
        satellite_modules=satellite_modules,
    )
    artifact = {
        "multichain": len(result.deployments) > 1,
        "simulated": True,
        "deployments": {
            slug: {
                "vault_address": deployed.vault.address
                if not deployed.is_satellite
                else None,
                "safe_address": deployed.safe_address,
                "module_address": deployed.trading_strategy_module.address,
                "asset_manager": account_address,
                "asset_managers": [account_address],
                "valuation_manager": account_address,
                "is_satellite": deployed.is_satellite,
            }
            for slug, deployed in result.deployments.items()
        },
    }
    return deployment, artifact


def deploy_simulated_lagoon_multichain(
    *,
    web3config,
    vault_specs: list[VaultSpec],
    vault_universe,
    private_key: str,
    amount: Decimal,
) -> tuple["LagoonDeployment", dict]:
    """Deploy a temporary Lagoon topology on the selected Anvil forks.

    Follows the multichain Lagoon integration-test setup: one source vault on
    the first explicitly supplied chain, deterministic satellite Safes on the
    remaining chains, per-chain transaction sequences and CCTP permissions for
    every supported source/destination route.
    """

    # The first explicit id defines the hub chain.  Preserving caller order is
    # important because all cross-chain vaults bridge through this deployment.
    assert vault_specs, "A simulated deployment needs at least one vault"
    account = Account.from_key(private_key)
    primary_chain_id = ChainId(vault_specs[0].chain_id)
    selected_chain_ids = list(
        dict.fromkeys(ChainId(spec.chain_id) for spec in vault_specs)
    )

    # Fail before deploying any contracts if one requested chain could not be
    # forked.  Partial topologies are never useful for the sequential batch.
    missing_connections = [
        chain_id.get_name()
        for chain_id in selected_chain_ids
        if chain_id not in web3config.connections
    ]
    if missing_connections:
        raise RuntimeError(
            f"Missing JSON-RPC connections for simulated Lagoon deployment: {', '.join(missing_connections)}"
        )

    # The Lagoon deployer expects slug-keyed Web3 instances.  Fund the standard
    # Anvil account independently on every fork because native balances are not
    # shared across chains.
    chain_web3 = {
        get_chain_slug(chain_id): web3config.get_connection(chain_id)
        for chain_id in selected_chain_ids
    }
    for web3 in chain_web3.values():
        set_balance(web3, account.address, 100 * 10**18)

    # Build each guard whitelist only from vaults hosted by that chain.
    specs_by_chain: dict[ChainId, list[VaultSpec]] = {}
    for spec in vault_specs:
        specs_by_chain.setdefault(ChainId(spec.chain_id), []).append(spec)

    # Reproduce the production Lagoon topology: the hub gets a real ERC-4626
    # vault and every other selected chain gets a satellite Safe and module.
    safe_salt_nonce = 42
    configs = {
        get_chain_slug(chain_id): _create_simulated_lagoon_chain_config(
            chain_id=chain_id,
            primary_chain_id=primary_chain_id,
            selected_chain_ids=selected_chain_ids,
            web3=chain_web3[get_chain_slug(chain_id)],
            vault_specs=specs_by_chain[chain_id],
            vault_universe=vault_universe,
            account_address=account.address,
            safe_salt_nonce=safe_salt_nonce,
        )
        for chain_id in selected_chain_ids
    }

    # Contract deployment is atomic at the generation level: the caller tears
    # down every fork if any chain fails here.
    result = deploy_multichain_lagoon_vault(
        chain_web3=chain_web3,
        deployer=account,
        chain_configs=configs,
    )

    # Seed the hub Safe with enough USDC for every sequential attempt.  Token
    # decimals come from the live forked contract and are never hardcoded.
    primary_slug = get_chain_slug(primary_chain_id)
    primary = result.deployments[primary_slug]
    primary_web3 = chain_web3[primary_slug]
    primary_usdc = fetch_erc20_details(
        primary_web3, USDC_NATIVE_TOKEN[primary_chain_id.value]
    )
    funding_raw = primary_usdc.convert_to_raw(max(amount * Decimal(100), Decimal(100)))
    fund_erc20_on_anvil(
        primary_web3,
        USDC_NATIVE_TOKEN[primary_chain_id.value],
        primary.safe_address,
        funding_raw,
    )

    # Convert eth-defi output into the small topology used by the executor and
    # the standard deployment JSON consumed by bootstrap.
    return _serialise_simulated_lagoon_deployment(
        result=result,
        chain_web3=chain_web3,
        primary_chain_id=primary_chain_id,
        account_address=account.address,
    )


@dataclass(frozen=True)
class LagoonDeployment:
    """Runtime topology read from a Lagoon deployment artefact.

    ``primary_chain_id`` owns the ERC-4626 Lagoon vault and source Safe.
    ``satellite_modules`` maps every additional chain to its guarded execution
    module; no module discovery is performed from environment variables.
    """

    primary_chain_id: ChainId
    vault_address: str
    module_address: str
    satellite_modules: dict[ChainId, str]


def parse_vault_ids(raw_value: str | None) -> list[VaultSpec]:
    """Parse ordered comma-separated ``VAULT_ID`` input.

    Keep the command-line order; automatic modes deliberately never turn this
    list into a set.
    """

    if not raw_value or not raw_value.strip():
        raise ValueError(
            "VAULT_ID / --vault-id must contain at least one chain-address vault id"
        )

    # Collect all validation problems so a long automatic invocation reports
    # every malformed id in one error.
    result: list[VaultSpec] = []
    seen: set[str] = set()
    failures: list[str] = []
    for raw_item in raw_value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        try:
            spec = VaultSpec.parse_string(item, separator="-")
        except Exception as e:
            failures.append(f"{item!r}: {e}")
            continue
        # Use the canonical eth-defi representation for duplicate detection,
        # while preserving the original list order in ``result``.
        canonical = spec.as_string_id()
        if canonical in seen:
            failures.append(f"{item!r}: duplicate vault id")
            continue
        seen.add(canonical)
        result.append(spec)

    if failures:
        raise ValueError("Invalid VAULT_ID entries:\n - " + "\n - ".join(failures))
    if not result:
        raise ValueError("VAULT_ID / --vault-id did not contain any vault ids")
    return result


def load_lagoon_deployment(deployment_file: Path) -> LagoonDeployment:
    """Load the mandatory state-sibling deployment artefact."""

    if not deployment_file.exists():
        raise RuntimeError(
            f"Missing mandatory Lagoon deployment file: {deployment_file}"
        )

    try:
        payload = json.loads(deployment_file.read_text())
        deployments = payload["deployments"]
    except Exception as e:
        raise RuntimeError(
            f"Malformed Lagoon deployment file: {deployment_file}"
        ) from e

    # Exactly one non-satellite entry establishes the reserve chain, Safe,
    # Lagoon vault and source trading module.
    source_entries = [
        (chain_slug, entry)
        for chain_slug, entry in deployments.items()
        if not entry.get("is_satellite", False)
    ]
    if len(source_entries) != 1:
        raise RuntimeError(
            f"Deployment file must contain exactly one source deployment, got {len(source_entries)}: {deployment_file}"
        )

    source_slug, source = source_entries[0]
    vault_address = source.get("vault_address")
    module_address = source.get("module_address")
    if not vault_address or not module_address:
        raise RuntimeError(
            f"Source deployment is missing vault/module address: {deployment_file}"
        )

    primary_chain_id = ChainId.get_by_slug(source_slug)
    if primary_chain_id is None:
        raise RuntimeError(
            f"Unknown deployment chain slug {source_slug!r}: {deployment_file}"
        )

    # Satellite entries intentionally need only their module address; custody
    # Safe addresses are resolved by normal Lagoon deployment bootstrap.
    satellite_modules: dict[ChainId, str] = {}
    for chain_slug, entry in deployments.items():
        if not entry.get("is_satellite", False):
            continue
        module = entry.get("module_address")
        if not module:
            raise RuntimeError(
                f"Satellite deployment {chain_slug!r} is missing module address"
            )
        chain_id = ChainId.get_by_slug(chain_slug)
        if chain_id is None:
            raise RuntimeError(
                f"Unknown satellite deployment chain slug {chain_slug!r}"
            )
        satellite_modules[chain_id] = module

    return LagoonDeployment(
        primary_chain_id=primary_chain_id,
        vault_address=vault_address,
        module_address=module_address,
        satellite_modules=satellite_modules,
    )


def get_latest_vault_position(
    state: State, vault_spec: VaultSpec
) -> TradingPosition | None:
    """Return the newest diagnostic or traded position for one vault id.

    Metadata matching keeps pre-adapter diagnostic positions discoverable,
    while pair matching retains compatibility with positions created before
    vault-test metadata was stamped.
    """

    vault_id = vault_spec.as_string_id()
    matches: list[TradingPosition] = []
    for position in state.portfolio.get_all_positions():
        attempt = position.other_data.get("vault_test_attempt", {})
        if attempt.get("vault_id") == vault_id:
            matches.append(position)
            continue
        if (
            position.pair.chain_id == vault_spec.chain_id
            and position.pair.pool_address.lower() == vault_spec.vault_address.lower()
        ):
            matches.append(position)
    return max(matches, key=lambda position: position.position_id, default=None)


def get_vault_trade_position(
    state: State,
    vault_spec: VaultSpec,
    *,
    open_only: bool = False,
    simulated: bool | None = None,
) -> TradingPosition | None:
    """Return the latest position that actually traded the selected vault pair.

    ``simulated`` is interpreted only by the vault-test command because its
    dedicated state intentionally contains both fork and real diagnostics.
    """

    matches = [
        position
        for position in state.portfolio.get_all_positions()
        if position.pair.chain_id == vault_spec.chain_id
        and position.pair.pool_address.lower() == vault_spec.vault_address.lower()
        and position.trades
        and (not open_only or position.is_open())
        and (simulated is None or position.simulated is simulated)
    ]
    return max(matches, key=lambda position: position.position_id, default=None)


def get_vault_test_status(position: TradingPosition | None) -> str:
    """Derive the TUI/table status from metadata, pending trades and position state."""

    if position is None:
        return "not tested"

    # Explicit terminal results take priority over inferred trade lifecycle
    # state because adapter and infrastructure failures may have no trades.
    attempt = position.other_data.get("vault_test_attempt", {})
    result = attempt.get("result")
    if result:
        return result.replace("_", " ")

    phase = attempt.get("phase")
    if phase == "bridge_back_pending":
        return "bridge back pending"
    if phase == "bridge_out_pending":
        return "bridge out pending"

    # Pending async requests remain open across command invocations.  Direction
    # distinguishes a deposit ticket from a redemption ticket in the same enum.
    trades = list(position.trades.values())
    if any(
        trade.get_status() == TradeStatus.vault_settlement_pending for trade in trades
    ):
        direction = next(
            (
                trade.other_data.get("vault_direction")
                for trade in reversed(trades)
                if trade.get_status() == TradeStatus.vault_settlement_pending
            ),
            None,
        )
        return "redemption pending" if direction == "redeem" else "deposit pending"
    if position.is_open():
        return "deposited"
    if position.simulated:
        return "success (simulated)"
    if position.is_closed():
        return "success"
    return "failed"


def build_vault_test_universe(
    *,
    client,
    vault_universe,
    vault_spec: VaultSpec,
    reserve_asset,
    primary_chain_id: ChainId,
    execution_context: ExecutionContext,
) -> TradingStrategyUniverse:
    """Build a fresh one-vault executable universe for an action.

    A failed adapter or data translation is intentionally allowed to escape to
    the caller, which records a diagnostic result and continues the batch.
    """

    # Limit before loading pair data so incomplete adapters fail only their own
    # requested vault and cannot prevent unrelated ids from running.
    selected_vault = vault_universe.limit_to_vaults(
        [(ChainId(vault_spec.chain_id), vault_spec.vault_address)],
        check_all_vaults_found=True,
    )
    dataset = load_partial_data(
        client=client,
        time_bucket=TimeBucket.d1,
        pairs=[],
        execution_context=execution_context,
        universe_options=UniverseOptions(history_period=datetime.timedelta(days=1)),
        liquidity=False,
        vaults=selected_vault,
        vault_history_source="none",
        check_all_vaults_found=True,
    )
    if dataset.candles is None or dataset.candles.empty:
        # Vault-only live universes do not download OHLCV data, but the normal
        # universe constructor still expects the canonical empty candle schema.
        dataset.candles = Candle.to_dataframe()
    return TradingStrategyUniverse.create_from_dataset(
        dataset,
        reserve_asset=reserve_asset,
        primary_chain=primary_chain_id,
        auto_generate_cctp_bridges=vault_spec.chain_id != primary_chain_id.value,
    )


def create_vault_test_diagnostic_pair(
    vault_spec: VaultSpec,
    reserve_asset: AssetIdentifier,
    vault=None,
) -> TradingPairIdentifier:
    """Create a serialisable placeholder pair when a vault adapter cannot load.

    The point of ``vault-test-trade`` is to retain adapter and universe failures,
    including vaults whose on-chain adapter support is incomplete.  A normal
    ``TradingPosition`` still needs a pair, so diagnostics use the downloaded
    vault token metadata when available and safe placeholders otherwise.  This
    pair is never routed or executed.
    """

    # Prefer downloaded share-token metadata, but fall back to deterministic
    # serialisable values when metadata loading was the failure being recorded.
    chain_id = vault_spec.chain_id
    base_address = (
        getattr(vault, "share_token_address", None) or vault_spec.vault_address
    )
    base_symbol = (
        getattr(vault, "share_token_symbol", None)
        or getattr(vault, "token_symbol", None)
        or "UNKNOWN"
    )
    base_decimals = getattr(vault, "share_token_decimals", None)
    if base_decimals is None:
        base_decimals = 18
    base_decimals = int(base_decimals)

    # A placeholder pair must live on the target chain even though its reserve
    # metadata originates from the hub executor's denomination asset.
    quote_address = (
        getattr(vault, "denomination_token_address", None) or reserve_asset.address
    )
    quote_symbol = (
        getattr(vault, "denomination_token_symbol", None) or reserve_asset.token_symbol
    )
    quote_decimals = getattr(vault, "denomination_token_decimals", None)
    if quote_decimals is None:
        quote_decimals = reserve_asset.decimals
    quote_decimals = int(quote_decimals)

    base = AssetIdentifier(chain_id, base_address, base_symbol, base_decimals)
    quote = AssetIdentifier(chain_id, quote_address, quote_symbol, quote_decimals)
    # Derive a stable JSON-safe identifier without colliding with normal small
    # pair ids.  The 53-bit mask also keeps it exactly representable in JS UIs.
    internal_id = int.from_bytes(
        hashlib.sha256(
            f"{chain_id}:{vault_spec.vault_address.lower()}".encode("ascii")
        ).digest()[:8],
        "big",
    ) & ((1 << 53) - 1)
    protocol_slug = getattr(vault, "protocol_slug", None) or "unknown"

    return TradingPairIdentifier(
        base=base,
        quote=quote,
        pool_address=vault_spec.vault_address,
        exchange_address=ZERO_ADDRESS_STR,
        internal_id=internal_id,
        fee=0,
        reverse_token_order=False,
        exchange_name=getattr(vault, "name", None) or vault_spec.vault_address,
        kind=TradingPairKind.vault,
        other_data={
            "vault_features": list(getattr(vault, "features", None) or []),
            "vault_protocol": protocol_slug,
        },
    )


def create_vault_test_attempt_metadata(
    vault_spec: VaultSpec, *, simulated: bool
) -> dict:
    """Create JSON-serialisable metadata carried by the authoritative position."""

    return {
        "vault_id": vault_spec.as_string_id(),
        "simulated": simulated,
        "phase": "created",
        "created_at": native_datetime_utc_now().isoformat(),
    }


def stamp_vault_test_attempt(
    state: State,
    vault_spec: VaultSpec,
    *,
    simulated: bool,
    phase: str | None = None,
    result: str | None = None,
    detail: str | None = None,
) -> TradingPosition | None:
    """Attach vault-test provenance and outcome to the latest target position."""

    position = get_latest_vault_position(state, vault_spec)
    if position is None:
        return None
    stamp_position_vault_test_attempt(
        position,
        vault_spec,
        simulated=simulated,
        phase=phase,
        result=result,
        detail=detail,
    )
    return position


def stamp_position_vault_test_attempt(
    position: TradingPosition,
    vault_spec: VaultSpec,
    *,
    simulated: bool,
    phase: str | None = None,
    result: str | None = None,
    detail: str | None = None,
) -> None:
    """Attach vault-test provenance to a specific target or bridge position.

    This is the shared stamping path for traded target and bridge positions.
    General executor accounting deliberately does not interpret the special
    ``TradingPosition.simulated`` marker.
    """

    position.simulated = simulated
    attempt = position.other_data.setdefault(
        "vault_test_attempt",
        create_vault_test_attempt_metadata(vault_spec, simulated=simulated),
    )
    attempt["vault_id"] = vault_spec.as_string_id()
    attempt["simulated"] = simulated
    if phase:
        attempt["phase"] = phase
    if result:
        attempt["result"] = result
    if detail:
        attempt["detail"] = detail


def close_simulated_positions(
    state: State,
    *,
    vault_spec: VaultSpec,
    position_ids: set[int],
    result: str | None = None,
) -> None:
    """Close all newly-created fork positions and stamp their vault-test role.

    Both target-vault and temporary CCTP bridge positions are closed because the
    Anvil snapshot is about to be reverted.  Only the target position receives
    the user-facing vault result metadata.
    """

    now = native_datetime_utc_now()
    vault_id = vault_spec.as_string_id()
    # Close positions still open after deposit-only or failed redemption flows.
    for position in list(state.portfolio.open_positions.values()):
        if position.position_id not in position_ids:
            continue
        position.simulated = True
        if position.pair.pool_address.lower() == vault_spec.vault_address.lower():
            attempt = position.other_data.setdefault("vault_test_attempt", {})
            attempt.setdefault("vault_id", vault_id)
            attempt["simulated"] = True
            if result:
                attempt["result"] = result
        state.portfolio.close_position(position, now)

    # A full instant round trip is already closed by perform_test_trade(); stamp
    # those positions without trying to close them a second time.
    for position in state.portfolio.closed_positions.values():
        if position.position_id not in position_ids:
            continue
        position.simulated = True
        if position.pair.pool_address.lower() == vault_spec.vault_address.lower():
            attempt = position.other_data.setdefault("vault_test_attempt", {})
            attempt.setdefault("vault_id", vault_id)
            attempt["simulated"] = True
            if result:
                attempt["result"] = result


def merge_simulated_attempt(
    *,
    source_state: State,
    target_state: State,
    original_position_ids: set[int],
    original_trade_ids: set[int],
) -> list[TradingPosition]:
    """Copy only fork-created closed diagnostics to the persisted state.

    The caller executes against a deep copy.  This makes it impossible to write
    fork-derived balance, valuation or settlement changes for an existing live
    position into the normal state file.
    """

    # Ignore all pre-existing positions from the copied state.  Importing only
    # newly allocated, explicitly simulated positions prevents fork balances or
    # lifecycle changes from overwriting real history.
    imported: list[TradingPosition] = []
    for position in source_state.portfolio.closed_positions.values():
        if position.position_id in original_position_ids:
            continue
        if not position.simulated:
            continue
        copied = deepcopy(position)
        target_state.portfolio.closed_positions[copied.position_id] = copied
        imported.append(copied)

    if imported:
        # Carry the id counters forward so the next real or simulated attempt
        # cannot reuse identifiers present in the merged diagnostics.
        target_state.portfolio.next_position_id = max(
            target_state.portfolio.next_position_id,
            max(position.position_id for position in imported) + 1,
        )
        max_trade_id = max(
            (
                trade.trade_id
                for position in imported
                for trade in position.trades.values()
                if trade.trade_id not in original_trade_ids
            ),
            default=target_state.portfolio.next_trade_id - 1,
        )
        target_state.portfolio.next_trade_id = max(
            target_state.portfolio.next_trade_id, max_trade_id + 1
        )

    return imported
