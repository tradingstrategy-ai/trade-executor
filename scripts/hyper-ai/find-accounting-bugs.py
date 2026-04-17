"""Find Hyper AI accounting bugs and possible stuck money.

Usage::

    source .local-test.env && poetry run python scripts/hyper-ai/find-accounting-bugs.py https://hyper-ai.tradingstrategy.ai/state --live

The script is read-only. It compares state accounting against live
Hyperliquid vault/spot/perp balances and prints likely accounting defects.
"""

import argparse
import datetime
import json
import os
import re
import urllib.request
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from eth_defi.compat import native_datetime_utc_fromtimestamp
from eth_defi.erc_4626.classification import create_vault_instance
from eth_defi.erc_4626.core import ERC4626Feature
from eth_defi.hyperliquid.api import (
    fetch_perp_clearinghouse_state,
    fetch_spot_clearinghouse_state,
    fetch_user_vault_equities,
)
from eth_defi.hyperliquid.session import HYPERLIQUID_API_URL, create_hyperliquid_session
from eth_defi.provider.multi_provider import create_multi_provider_web3


def load_state(source: str) -> dict[str, Any]:
    """Load state JSON from a file path or an executor URL."""
    if source.startswith("http://") or source.startswith("https://"):
        url = source.rstrip("/")
        if not url.endswith("/state"):
            url += "/state"
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "trade-executor-hyper-ai-accounting-audit/1.0"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))

    return json.loads(Path(source).read_text())


def parse_decimal(value: Any, default: Decimal = Decimal(0)) -> Decimal:
    """Parse JSON value as Decimal without losing precision for strings."""
    if value is None:
        return default
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return default


def format_time(value: int | float | str | None) -> str:
    """Format a UNIX timestamp for console output."""
    if value is None:
        return "-"
    dt = native_datetime_utc_fromtimestamp(float(value))
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def format_dt(value: datetime.datetime | None) -> str:
    """Format a datetime for console output."""
    if value is None:
        return "-"
    return value.replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S UTC")


def get_ticker(pair: dict[str, Any]) -> str:
    """Format a human-friendly pair ticker."""
    base = pair.get("base", {}).get("token_symbol") or "?"
    quote = pair.get("quote", {}).get("token_symbol") or "?"
    return f"{base}-{quote}"


def get_vault_address(position: dict[str, Any]) -> str | None:
    """Return a normalised Hypercore vault address for a state position."""
    pair = position.get("pair") or {}
    address = pair.get("pool_address") or pair.get("base", {}).get("address")
    if not address:
        return None
    return address.lower()


def is_hypercore_vault_position(position: dict[str, Any]) -> bool:
    """Return true if a serialised position is a Hypercore native vault."""
    pair = position.get("pair") or {}
    base = pair.get("base") or {}
    other_data = pair.get("other_data") or {}
    return (
        pair.get("kind") == "vault"
        and (
            int(base.get("chain_id") or 0) == 9999
            or "hypercore_native" in set(other_data.get("vault_features") or [])
        )
    )


def iter_positions(state: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Return all portfolio positions with their bucket name."""
    portfolio = state.get("portfolio", {})
    rows = []
    for bucket_name in ("open", "frozen", "closed"):
        positions = portfolio.get(f"{bucket_name}_positions") or {}
        for position in positions.values():
            rows.append((bucket_name, position))
    return rows


def iter_trades(state: dict[str, Any]) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    """Return all trades with parent bucket and position."""
    rows = []
    for bucket_name, position in iter_positions(state):
        trades = position.get("trades") or {}
        if isinstance(trades, dict):
            trades = trades.values()
        for trade in trades:
            rows.append((bucket_name, position, trade))
    return rows


def latest_item(items: list[dict[str, Any]], timestamp_key: str) -> dict[str, Any] | None:
    """Return the newest serialised state row."""
    if not items:
        return None
    return max(
        items,
        key=lambda item: int(item.get(timestamp_key) or 0),
    )


def latest_position_valuation(position: dict[str, Any]) -> dict[str, Any] | None:
    """Return the latest valuation update for a position."""
    updates = position.get("valuation_updates") or []
    if isinstance(updates, dict):
        updates = list(updates.values())
    return latest_item(updates, "created_at")


def latest_position_stats(state: dict[str, Any], position_id: int) -> dict[str, Any] | None:
    """Return the latest statistics row for a position."""
    rows = state.get("stats", {}).get("positions", {}).get(str(position_id)) or []
    return latest_item(rows, "calculated_at")


def latest_portfolio_stats(state: dict[str, Any]) -> dict[str, Any] | None:
    """Return the latest portfolio statistics row."""
    rows = state.get("stats", {}).get("portfolio") or []
    return latest_item(rows, "calculated_at")


def get_cash(state: dict[str, Any]) -> Decimal:
    """Return reserve cash from state."""
    total = Decimal(0)
    reserves = state.get("portfolio", {}).get("reserves") or {}
    for reserve in reserves.values():
        quantity = parse_decimal(reserve.get("quantity"))
        price = parse_decimal(reserve.get("reserve_token_price"), Decimal(1))
        total += quantity * price
    return total


def resolve_lagoon_safe(state: dict[str, Any]) -> str:
    """Resolve Lagoon Safe address from the vault address using HyperEVM RPC."""
    rpc_url = os.environ.get("JSON_RPC_HYPERLIQUID")
    if not rpc_url:
        raise RuntimeError("JSON_RPC_HYPERLIQUID is not set")

    vault_address = state.get("sync", {}).get("deployment", {}).get("address")
    if not vault_address:
        raise RuntimeError("State has no sync.deployment.address")

    web3 = create_multi_provider_web3(rpc_url)
    vault = create_vault_instance(
        web3,
        vault_address,
        features={ERC4626Feature.lagoon_like},
        default_block_identifier="latest",
        require_denomination_token=True,
    )
    if vault is None:
        raise RuntimeError(f"Could not create Lagoon vault instance for {vault_address}")
    return vault.safe_address


def get_spot_usdc_balances(spot_state: Any) -> tuple[Decimal, Decimal]:
    """Extract total and free spot USDC from Hyperliquid spot state."""
    for balance in spot_state.balances:
        if balance.coin == "USDC":
            return balance.total, balance.total - balance.hold
    return Decimal(0), Decimal(0)


def print_state_totals(state: dict[str, Any]) -> None:
    """Print high-level accounting totals."""
    stats = latest_portfolio_stats(state)
    cash = get_cash(state)
    open_value = Decimal(0)
    for bucket_name, position in iter_positions(state):
        if bucket_name != "open":
            continue
        valuation = latest_position_valuation(position)
        if valuation:
            open_value += parse_decimal(valuation.get("new_value"))

    print("State totals")
    print(f"  State last updated: {format_time(state.get('last_updated_at'))}")
    print(f"  Reserve cash: {cash:,.6f} USD")
    print(f"  Open position value from marks: {open_value:,.6f} USD")
    print(f"  Computed equity: {open_value + cash:,.6f} USD")
    if stats:
        print(f"  Latest stats equity: {parse_decimal(stats.get('total_equity')):,.6f} USD")
        print(f"  Latest stats open value: {parse_decimal(stats.get('open_position_equity')):,.6f} USD")
        print(f"  Latest stats cash: {parse_decimal(stats.get('free_cash')):,.6f} USD")
        deposits = parse_decimal(stats.get("net_asset_flow"))
        if deposits:
            print(f"  Net deposits/redemptions: {deposits:,.6f} USD")
            print(f"  Deposits minus equity: {deposits - parse_decimal(stats.get('total_equity')):,.6f} USD")


def analyse_closed_state_values(state: dict[str, Any], threshold: Decimal) -> None:
    """Report closed positions that still carry stale value rows."""
    rows = []
    for bucket_name, position in iter_positions(state):
        if bucket_name != "closed":
            continue
        position_id = int(position.get("position_id"))
        valuation = latest_position_valuation(position)
        stats = latest_position_stats(state, position_id)
        valuation_value = parse_decimal(valuation.get("new_value")) if valuation else Decimal(0)
        stats_value = parse_decimal(stats.get("value")) if stats else Decimal(0)
        value = max(abs(valuation_value), abs(stats_value))
        if value < threshold:
            continue
        rows.append((value, position_id, get_ticker(position.get("pair") or {}), valuation_value, stats_value))

    rows.sort(reverse=True)
    print()
    print("Closed positions with non-zero state value")
    if not rows:
        print(f"  None above {threshold} USD")
        return
    print("  id   pair              valuation      stats")
    for _value, position_id, ticker, valuation_value, stats_value in rows[:40]:
        print(f"  {position_id:>3}  {ticker[:16]:<16} {valuation_value:>12,.6f} {stats_value:>12,.6f}")


def analyse_live_hypercore(state: dict[str, Any], safe_address: str, threshold: Decimal) -> None:
    """Compare state positions with live Hyperliquid balances."""
    session = create_hyperliquid_session(api_url=HYPERLIQUID_API_URL)
    live_equities = fetch_user_vault_equities(session, user=safe_address)
    live_by_address = {
        item.vault_address.lower(): item
        for item in live_equities
    }

    positions_by_vault: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for bucket_name, position in iter_positions(state):
        address = get_vault_address(position)
        if not address:
            continue
        if not is_hypercore_vault_position(position):
            continue
        positions_by_vault.setdefault(address, []).append((bucket_name, position))

    open_or_frozen_addresses = {
        address
        for address, positions in positions_by_vault.items()
        if any(bucket_name in {"open", "frozen"} for bucket_name, _position in positions)
    }

    print()
    print("Live Hypercore balances")
    print(f"  Lagoon Safe: {safe_address}")
    print("  id/status     pair              state value     live equity    drift      lock")

    tracked_state_total = Decimal(0)
    tracked_live_total = Decimal(0)
    for address in sorted(open_or_frozen_addresses):
        live = live_by_address.get(address)
        live_equity = live.equity if live is not None else Decimal(0)
        for bucket_name, position in positions_by_vault[address]:
            if bucket_name not in {"open", "frozen"}:
                continue
            position_id = int(position.get("position_id"))
            valuation = latest_position_valuation(position)
            state_value = parse_decimal(valuation.get("new_value")) if valuation else Decimal(0)
            tracked_state_total += state_value
            tracked_live_total += live_equity
            lock = "?"
            if live is not None:
                lock = "open" if live.is_lockup_expired else f"locked until {format_dt(live.locked_until)}"
            print(
                f"  {position_id:>3}/{bucket_name:<6} {get_ticker(position.get('pair') or {})[:16]:<16} "
                f"{state_value:>12,.6f} {live_equity:>12,.6f} {live_equity - state_value:>+10,.6f} {lock}"
            )

    print(f"  Tracked open/frozen state value: {tracked_state_total:,.6f} USD")
    print(f"  Tracked open/frozen live equity: {tracked_live_total:,.6f} USD")
    print(f"  Live drift: {tracked_live_total - tracked_state_total:+,.6f} USD")

    untracked = []
    for address, live in live_by_address.items():
        if address in open_or_frozen_addresses:
            continue
        if live.equity < threshold:
            continue
        state_positions = positions_by_vault.get(address, [])
        untracked.append((live.equity, address, live, state_positions))

    untracked.sort(reverse=True, key=lambda item: item[0])
    print()
    print("Live vault equity without open/frozen state position")
    if not untracked:
        print(f"  None above {threshold} USD")
    else:
        print("  equity       address                                      state positions")
        for equity, address, live, state_positions in untracked:
            labels = ", ".join(
                f"#{position.get('position_id')} {bucket_name} {get_ticker(position.get('pair') or {})}"
                for bucket_name, position in state_positions
            ) or "none"
            lock = "open" if live.is_lockup_expired else f"locked until {format_dt(live.locked_until)}"
            print(f"  {equity:>10,.6f} {address}  {labels}; {lock}")

    spot_state = fetch_spot_clearinghouse_state(session, user=safe_address)
    perp_state = fetch_perp_clearinghouse_state(session, user=safe_address)
    spot_total, spot_free = get_spot_usdc_balances(spot_state)

    print()
    print("Live HyperCore transit balances")
    print(f"  Spot USDC total: {spot_total:,.6f} USD")
    print(f"  Spot USDC free: {spot_free:,.6f} USD")
    print(f"  Perp withdrawable: {perp_state.withdrawable:,.6f} USD")
    print(f"  Perp asset positions: {len(perp_state.asset_positions)}")


def analyse_stranded_markers(state: dict[str, Any], threshold: Decimal) -> None:
    """Report explicit stranded-USDC markers and zero-execution trades."""
    stranded = []
    zero_success = []
    for bucket_name, position, trade in iter_trades(state):
        other_data = trade.get("other_data") or {}
        marker = other_data.get("hypercore_stranded_usdc")
        if marker:
            amount = parse_decimal(marker.get("amount_human"))
            stranded.append((amount, bucket_name, position, trade, marker))

        executed_at = trade.get("executed_at")
        failed_at = trade.get("failed_at")
        repaired_at = trade.get("repaired_at")
        executed_quantity = abs(parse_decimal(trade.get("executed_quantity")))
        executed_reserve = abs(parse_decimal(trade.get("executed_reserve")))
        planned_reserve = abs(parse_decimal(trade.get("planned_reserve")))
        if executed_at and not failed_at and not repaired_at:
            if executed_quantity == 0 and executed_reserve == 0 and planned_reserve >= threshold:
                zero_success.append((planned_reserve, bucket_name, position, trade))

    stranded.sort(reverse=True, key=lambda item: item[0])
    zero_success.sort(reverse=True, key=lambda item: item[0])

    print()
    print("Stranded-USDC trade markers")
    if not stranded:
        print("  None")
    else:
        print("  amount      trade     position    location")
        for amount, bucket_name, position, trade, marker in stranded:
            print(
                f"  {amount:>10,.6f} #{trade.get('trade_id'):<7} "
                f"#{position.get('position_id')} {bucket_name:<6} {marker.get('location')}"
            )

    print()
    print("Successful zero-execution trades with planned notional")
    if not zero_success:
        print(f"  None above {threshold} USD")
    else:
        print("  planned     trade     position    pair")
        for planned, bucket_name, position, trade in zero_success[:40]:
            print(
                f"  {planned:>10,.6f} #{trade.get('trade_id'):<7} "
                f"#{position.get('position_id')} {bucket_name:<6} {get_ticker(position.get('pair') or {})}"
            )


def parse_correction_notes(notes: str | None) -> tuple[Decimal | None, Decimal | None]:
    """Parse expected and actual balances from accounting correction notes."""
    if not notes:
        return None, None

    expected_match = re.search(r"internal ledger balance was\s+([0-9.Ee+-]+)", notes)
    actual_match = re.search(r"On-chain balance was\s+([0-9.Ee+-]+)", notes)
    expected = parse_decimal(expected_match.group(1)) if expected_match else None
    actual = parse_decimal(actual_match.group(1)) if actual_match else None
    return expected, actual


def analyse_correction_events(state: dict[str, Any], threshold: Decimal) -> None:
    """Report large accounting corrections and old_balance inconsistencies."""
    rows = []
    old_balance_bugs = []

    holders: list[tuple[str, dict[str, Any], str]] = []
    for bucket_name, position in iter_positions(state):
        holders.append((bucket_name, position, f"#{position.get('position_id')} {get_ticker(position.get('pair') or {})}"))
    for reserve in (state.get("portfolio", {}).get("reserves") or {}).values():
        holders.append(("reserve", reserve, "reserve USDC"))

    for bucket_name, holder, label in holders:
        updates = holder.get("balance_updates") or {}
        if isinstance(updates, list):
            updates = {str(item.get("balance_update_id")): item for item in updates}
        for update in updates.values():
            cause = update.get("cause")
            if cause != "correction":
                continue
            quantity = parse_decimal(update.get("quantity"))
            if abs(quantity) >= threshold:
                rows.append((abs(quantity), bucket_name, label, update))
            expected, actual = parse_correction_notes(update.get("notes"))
            if expected is None or actual is None:
                continue
            old_balance = parse_decimal(update.get("old_balance"))
            if abs(old_balance - actual) < Decimal("0.000001") and abs(expected - actual) >= threshold:
                old_balance_bugs.append((bucket_name, label, update, expected, actual))

    rows.sort(reverse=True, key=lambda item: item[0])

    print()
    print("Large accounting correction events")
    if not rows:
        print(f"  None above {threshold} USD/token")
    else:
        print("  qty         update    bucket    position")
        for _abs_quantity, bucket_name, label, update in rows[:30]:
            print(
                f"  {parse_decimal(update.get('quantity')):>+11,.6f} "
                f"#{update.get('balance_update_id'):<7} {bucket_name:<8} "
                f"{label}"
            )

    print()
    print("Correction old_balance bugs")
    if not old_balance_bugs:
        print(f"  None above {threshold} USD/token")
    else:
        print("  update    bucket    recorded old     expected old     actual")
        for bucket_name, label, update, expected, actual in old_balance_bugs[:30]:
            print(
                f"  #{update.get('balance_update_id'):<7} {bucket_name:<8} "
                f"{parse_decimal(update.get('old_balance')):>13,.6f} "
                f"{expected:>13,.6f} {actual:>13,.6f} {label}"
            )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="State JSON file or executor URL")
    parser.add_argument("--live", action="store_true", help="Fetch live Hyperliquid balances")
    parser.add_argument("--threshold", type=Decimal, default=Decimal("0.01"))
    args = parser.parse_args()

    state = load_state(args.source)

    print_state_totals(state)
    analyse_closed_state_values(state, args.threshold)
    analyse_correction_events(state, args.threshold)
    analyse_stranded_markers(state, args.threshold)

    if args.live:
        safe_address = resolve_lagoon_safe(state)
        analyse_live_hypercore(state, safe_address, args.threshold)


if __name__ == "__main__":
    main()
