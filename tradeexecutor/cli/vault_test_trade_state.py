"""Dedicated position state helpers for ``vault-test-trade``.

The command stores real and fork-only diagnostics in a normal executor
``State``.  Interpretation of the special ``TradingPosition.simulated`` field
belongs here and in the vault-test TUI only; general accounting and analytics
must remain unaware of it.
"""

import hashlib
from copy import deepcopy

from eth_defi.abi import ZERO_ADDRESS_STR
from eth_defi.compat import native_datetime_utc_now
from eth_defi.vault.base import VaultSpec

from tradeexecutor.state.identifier import (
    AssetIdentifier,
    TradingPairIdentifier,
    TradingPairKind,
)
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeStatus


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
    """Derive the TUI/table status from metadata and position state."""

    if position is None:
        return "not tested"

    # Explicit terminal results take priority over inferred trade lifecycle
    # state because adapter and infrastructure failures may have no trades.
    attempt = position.other_data.get("vault_test_attempt", {})
    result = attempt.get("result")
    if result:
        return result.replace("_", " ")

    phase_status = {
        "bridge_back_pending": "bridge back pending",
        "bridge_out_pending": "bridge out pending",
    }.get(attempt.get("phase"))
    if phase_status:
        return phase_status

    # Pending async requests remain open across command invocations. Direction
    # distinguishes a deposit ticket from a redemption ticket in the same enum.
    pending_trade = next(
        (
            trade
            for trade in reversed(position.trades.values())
            if trade.get_status() == TradeStatus.vault_settlement_pending
        ),
        None,
    )
    if pending_trade is not None:
        direction = pending_trade.other_data.get("vault_direction")
        return "redemption pending" if direction == "redeem" else "deposit pending"
    if position.is_open():
        return "deposited"
    if position.simulated:
        return "success (simulated)"
    if position.is_closed():
        return "success"
    return "failed"


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

    base = AssetIdentifier(chain_id, base_address, base_symbol, int(base_decimals))
    quote = AssetIdentifier(chain_id, quote_address, quote_symbol, int(quote_decimals))

    # Derive a stable JSON-safe identifier without colliding with normal small
    # pair ids. The 53-bit mask also keeps it exactly representable in JS UIs.
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


def stamp_position_vault_test_attempt(
    position: TradingPosition,
    vault_spec: VaultSpec,
    *,
    simulated: bool,
    phase: str | None = None,
    result: str | None = None,
    detail: str | None = None,
) -> None:
    """Attach vault-test provenance to a specific target or bridge position."""

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


def record_attempt_result(
    state: State,
    pair: TradingPairIdentifier,
    vault_spec: VaultSpec,
    *,
    simulated: bool,
    result: str,
    detail: str | None = None,
    source_position_id: int | None = None,
) -> TradingPosition:
    """Create a closed diagnostic position in the dedicated vault-test state.

    Some failures happen before a transaction or even an adapter can be
    constructed. They still need one normal ``TradingPosition`` so the latest
    result for the vault remains discoverable by the TUI and subsequent runs.
    """

    reserve = state.portfolio.get_default_reserve_position().asset
    now = native_datetime_utc_now()
    position = state.portfolio.open_new_position(
        now,
        pair,
        assumed_price=1.0,
        reserve_currency=reserve,
        reserve_currency_price=1.0,
    )
    position.simulated = simulated

    attempt = create_vault_test_attempt_metadata(vault_spec, simulated=simulated)
    attempt["result"] = result
    if detail:
        attempt["detail"] = detail
    if source_position_id is not None:
        attempt["source_position_id"] = source_position_id
    position.other_data["vault_test_attempt"] = attempt

    # Diagnostic positions never represent live holdings, so close them at the
    # same timestamp at which they were created.
    state.portfolio.close_position(position, now)
    return position


def close_simulated_positions(
    state: State,
    *,
    vault_spec: VaultSpec,
    position_ids: set[int],
    result: str | None = None,
) -> None:
    """Close all newly-created fork positions and stamp their vault-test role.

    Both target-vault and temporary CCTP bridge positions are closed because the
    Anvil snapshot is about to be reverted. Only the target position receives
    the user-facing vault result metadata.
    """

    now = native_datetime_utc_now()
    vault_id = vault_spec.as_string_id()

    # A full instant round trip is already closed by perform_test_trade(), while
    # deposit-only and failed-redemption positions remain open. Process both
    # collections through the same stamping helper and close only when needed.
    positions = [
        position
        for position in (
            *state.portfolio.open_positions.values(),
            *state.portfolio.closed_positions.values(),
        )
        if position.position_id in position_ids
    ]
    for position in positions:
        position.simulated = True
        if position.pair.pool_address.lower() == vault_spec.vault_address.lower():
            attempt = position.other_data.setdefault("vault_test_attempt", {})
            attempt.setdefault("vault_id", vault_id)
            attempt["simulated"] = True
            if result:
                attempt["result"] = result
        if position.is_open():
            state.portfolio.close_position(position, now)


def merge_simulated_attempt(
    *,
    source_state: State,
    target_state: State,
    original_position_ids: set[int],
    original_trade_ids: set[int],
) -> list[TradingPosition]:
    """Copy only fork-created closed diagnostics to the persisted state.

    The caller executes against a deep copy. This makes it impossible to write
    fork-derived balance, valuation or settlement changes for an existing live
    position into the normal state file.
    """

    # Ignore all pre-existing positions from the copied state. Importing only
    # newly allocated, explicitly simulated positions prevents fork balances or
    # lifecycle changes from overwriting real history.
    imported: list[TradingPosition] = []
    for position in source_state.portfolio.closed_positions.values():
        if position.position_id in original_position_ids or not position.simulated:
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
            target_state.portfolio.next_trade_id,
            max_trade_id + 1,
        )

    return imported
