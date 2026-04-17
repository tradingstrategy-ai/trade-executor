"""Analyse Hyper AI state valuation and rebalance health.

Usage::

    source .local-test.env && poetry run python scripts/hyper-ai/analyse-state.py https://hyper-ai.tradingstrategy.ai/state --live

The script can read either a downloaded state JSON file or the executor
``/state`` endpoint. With ``--live`` it also resolves the Lagoon Safe from
the HyperEVM vault and compares state-side Hypercore vault marks with the
current Hyperliquid API equity.
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
from eth_defi.hyperliquid.api import fetch_user_vault_equity
from eth_defi.hyperliquid.session import HYPERLIQUID_API_URL, create_hyperliquid_session
from eth_defi.provider.multi_provider import create_multi_provider_web3


DECISION_KEYS = {
    "Cycle",
    "Rebalanced",
    "Max position value change",
    "Rebalance threshold",
    "Trades decided",
    "Pairs meeting inclusion criteria",
    "Candidate signals created",
    "Selected survivor signals",
    "Total equity",
    "Cash",
    "Redeemable capital",
    "Locked capital carried forward",
    "Pending redemptions",
    "Investable equity",
    "Accepted investable equity",
    "Allocated to signals",
    "Rebalance volume",
    "Blocked redemption checks",
    "Blocked redemption reasons",
}


def timestamp_to_datetime(value: int | float | str | None) -> datetime.datetime | None:
    """Convert a UNIX timestamp to a naive UTC datetime."""
    if value is None:
        return None
    return native_datetime_utc_fromtimestamp(float(value))


def format_time(value: int | float | str | None) -> str:
    """Format a UNIX timestamp for console output."""
    dt = timestamp_to_datetime(value)
    if dt is None:
        return "-"
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def format_age(seconds: Decimal | int | float | None) -> str:
    """Format an age in seconds."""
    if seconds is None:
        return "-"
    seconds = int(seconds)
    if seconds < 0:
        seconds = 0
    hours, remainder = divmod(seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m"


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


def parse_number_from_text(value: str | None) -> Decimal | None:
    """Parse values like '$1,234.56 USD' from decision messages."""
    if value is None:
        return None
    cleaned = value.replace("$", "").replace(",", "").replace("USD", "").strip()
    cleaned = cleaned.replace("%", "").strip()
    if not cleaned:
        return None
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def parse_since(value: str) -> int:
    """Parse an ISO timestamp as naive UTC and return a UNIX timestamp."""
    cleaned = value.rstrip("Z")
    dt = datetime.datetime.fromisoformat(cleaned)
    epoch = datetime.datetime(1970, 1, 1)
    return int((dt - epoch).total_seconds())


def load_state(source: str) -> dict[str, Any]:
    """Load state JSON from a file path or an executor URL."""
    if source.startswith("http://") or source.startswith("https://"):
        url = source.rstrip("/")
        if not url.endswith("/state"):
            url += "/state"
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "trade-executor-hyper-ai-analyse/1.0"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))

    return json.loads(Path(source).read_text())


def get_ticker(pair: dict[str, Any]) -> str:
    """Format a human-friendly pair ticker."""
    base = pair.get("base", {}).get("token_symbol") or "?"
    quote = pair.get("quote", {}).get("token_symbol") or "?"
    return f"{base}-{quote}"


def get_pair_id(pair: dict[str, Any]) -> int | None:
    """Read a pair internal id."""
    value = pair.get("internal_id")
    if value is None:
        return None
    return int(value)


def get_position_buckets(state: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Return all portfolio position buckets."""
    portfolio = state.get("portfolio", {})
    return [
        ("open", portfolio.get("open_positions") or {}),
        ("closed", portfolio.get("closed_positions") or {}),
        ("frozen", portfolio.get("frozen_positions") or {}),
    ]


def iter_trades(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten all trades from all position buckets."""
    trades = []
    for bucket_name, positions in get_position_buckets(state):
        for position in positions.values():
            raw_trades = position.get("trades") or {}
            position_trades = raw_trades.values() if isinstance(raw_trades, dict) else raw_trades
            for trade in position_trades:
                trade_copy = dict(trade)
                trade_copy["_position_bucket"] = bucket_name
                trade_copy["_position_ticker"] = get_ticker(position.get("pair") or {})
                trades.append(trade_copy)
    return trades


def get_trade_status(trade: dict[str, Any]) -> str:
    """Resolve trade status from persisted trade timestamps."""
    if trade.get("repaired_at"):
        return "repaired"
    if trade.get("failed_at"):
        return "failed"
    if trade.get("executed_at"):
        return "success"
    if trade.get("broadcasted_at"):
        return "broadcasted"
    if trade.get("started_at"):
        return "started"
    if trade.get("expired_at"):
        return "expired"
    return "planned"


def latest_item(items: list[dict[str, Any]], timestamp_key: str = "created_at") -> dict[str, Any] | None:
    """Return the latest dict from a serialised state list."""
    if not items:
        return None
    _, item = max(
        enumerate(items),
        key=lambda indexed_item: (int(indexed_item[1].get(timestamp_key) or 0), indexed_item[0]),
    )
    return item


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
    reserves = state.get("portfolio", {}).get("reserves") or {}
    total = Decimal(0)
    for reserve in reserves.values():
        quantity = parse_decimal(reserve.get("quantity"))
        price = parse_decimal(reserve.get("reserve_token_price"), Decimal(1))
        total += quantity * price
    return total


def parse_decision_report(text: str) -> dict[str, str]:
    """Parse the key-value lines saved by Hyper AI decide_trades()."""
    data = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in DECISION_KEYS or key.startswith("Signals with flag"):
            data[key] = value.strip()
    return data


def get_decision_messages(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Return strategy decision messages sorted by timestamp."""
    messages = state.get("visualisation", {}).get("messages") or {}
    decisions = []
    for raw_timestamp, values in messages.items():
        timestamp = int(raw_timestamp)
        text = "\n".join(values) if isinstance(values, list) else str(values)
        if "Trades decided:" not in text or "Cycle:" not in text:
            continue
        parsed = parse_decision_report(text)
        cycle = parsed.get("Cycle", "").lstrip("#")
        decisions.append(
            {
                "timestamp": timestamp,
                "cycle": int(cycle) if cycle.isdigit() else None,
                "text": text,
                "data": parsed,
            }
        )
    return sorted(decisions, key=lambda item: item["timestamp"])


def get_latest_completed_cycle_time(state: dict[str, Any]) -> tuple[int | None, int | None]:
    """Return latest completed cycle number and timestamp from uptime data."""
    cycles = state.get("uptime", {}).get("cycles_completed_at") or {}
    if not cycles:
        return None, None
    latest_cycle = max(int(cycle) for cycle in cycles)
    return latest_cycle, int(cycles[str(latest_cycle)])


def get_alpha_model_timestamp(state: dict[str, Any]) -> int | None:
    """Return the timestamp of the saved alpha model snapshot."""
    alpha_model = state.get("visualisation", {}).get("discardable_data", {}).get("alpha_model") or {}
    timestamp = alpha_model.get("timestamp")
    if timestamp is None:
        return None
    return int(timestamp)


def get_signal_lookup(state: dict[str, Any]) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    """Build signal lookup by position id and pair id."""
    alpha_model = state.get("visualisation", {}).get("discardable_data", {}).get("alpha_model") or {}
    signals = alpha_model.get("signals") or {}
    by_position = {}
    by_pair = {}
    for signal in signals.values():
        position_id = signal.get("position_id")
        pair = signal.get("pair") or {}
        pair_id = get_pair_id(pair)
        if position_id is not None:
            by_position[int(position_id)] = signal
        if pair_id is not None:
            by_pair[pair_id] = signal
    return by_position, by_pair


def expected_trade_signals(state: dict[str, Any], threshold: Decimal) -> list[dict[str, Any]]:
    """Return alpha model signals that should have generated a trade."""
    alpha_model = state.get("visualisation", {}).get("discardable_data", {}).get("alpha_model") or {}
    signals = alpha_model.get("signals") or {}
    expected = []
    for signal in signals.values():
        adjust = parse_decimal(signal.get("position_adjust_usd"))
        flags = set(signal.get("flags") or [])
        if abs(adjust) < threshold:
            continue
        if "cannot_redeem" in flags:
            continue
        if "individual_trade_size_too_small" in flags:
            continue
        expected.append(signal)
    return expected


def get_trade_signal(
    trade: dict[str, Any],
    by_position: dict[int, dict[str, Any]],
    by_pair: dict[int, dict[str, Any]],
) -> dict[str, Any] | None:
    """Find the alpha model signal corresponding to a trade."""
    position_id = trade.get("position_id")
    if position_id is not None and int(position_id) in by_position:
        return by_position[int(position_id)]

    pair_id = get_pair_id(trade.get("pair") or {})
    if pair_id is not None:
        return by_pair.get(pair_id)

    return None


def get_trade_direction(trade: dict[str, Any]) -> str:
    """Return buy or sell from planned quantity."""
    planned_quantity = parse_decimal(trade.get("planned_quantity"))
    return "buy" if planned_quantity > 0 else "sell"


def get_signal_direction(signal: dict[str, Any]) -> str:
    """Return buy or sell from alpha model adjustment."""
    adjust = parse_decimal(signal.get("position_adjust_usd"))
    return "buy" if adjust > 0 else "sell"


def get_status_counts(trades: list[dict[str, Any]]) -> dict[str, int]:
    """Count trades by status."""
    counts = {}
    for trade in trades:
        status = get_trade_status(trade)
        counts[status] = counts.get(status, 0) + 1
    return counts


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


def fetch_live_equities(state: dict[str, Any], safe_address: str) -> dict[int, dict[str, Any]]:
    """Fetch current Hyperliquid vault equity for open positions."""
    session = create_hyperliquid_session(api_url=HYPERLIQUID_API_URL)
    live = {}
    for position in state.get("portfolio", {}).get("open_positions", {}).values():
        position_id = int(position["position_id"])
        vault_address = position.get("pair", {}).get("pool_address")
        if not vault_address:
            continue
        eq = fetch_user_vault_equity(
            session,
            user=safe_address,
            vault_address=vault_address,
            bypass_cache=True,
        )
        live[position_id] = {
            "equity": eq.equity if eq is not None else Decimal(0),
            "lock_expired": eq.is_lockup_expired if eq is not None else None,
            "locked_until": eq.locked_until if eq is not None else None,
        }
    return live


def print_state_summary(state: dict[str, Any]) -> None:
    """Print state-level timestamps."""
    latest_cycle, completed_at = get_latest_completed_cycle_time(state)
    stats_row = latest_portfolio_stats(state)

    print("State summary")
    print(f"  State last updated: {format_time(state.get('last_updated_at'))}")
    print(f"  State cycle: {state.get('cycle')}")
    if latest_cycle is not None:
        print(f"  Latest completed cycle: #{latest_cycle} at {format_time(completed_at)}")
    if stats_row:
        print(f"  Latest portfolio stats: {format_time(stats_row.get('calculated_at'))}")
    print(f"  Stats refreshes completed: {state.get('uptime', {}).get('stats_refresh_completed')}")
    print(f"  Post-valuation settlements completed: {state.get('uptime', {}).get('post_valuation_settlements_completed')}")


def analyse_valuations(
    state: dict[str, Any],
    *,
    freshness_hours: Decimal,
    live_equities: dict[int, dict[str, Any]] | None,
) -> tuple[list[str], list[str]]:
    """Analyse valuation freshness and consistency."""
    issues = []
    notes = []
    reference_ts = int(state.get("last_updated_at") or 0)
    open_positions = state.get("portfolio", {}).get("open_positions", {}) or {}
    stats_row = latest_portfolio_stats(state)
    freshness_seconds = freshness_hours * Decimal(3600)

    print()
    print("Valuation checks")
    print("  id   pair              value       quantity    price       age    live       drift      lock")

    open_total = Decimal(0)
    live_total = Decimal(0)
    valuation_times = []

    for position in sorted(open_positions.values(), key=lambda item: int(item["position_id"])):
        position_id = int(position["position_id"])
        ticker = get_ticker(position.get("pair") or {})[:16]
        valuation = latest_position_valuation(position)
        if valuation is None:
            issues.append(f"Position #{position_id} {ticker} has no valuation updates")
            continue

        value = parse_decimal(valuation.get("new_value"))
        quantity = parse_decimal(valuation.get("quantity"))
        price = parse_decimal(valuation.get("new_price"))
        valuation_ts = int(valuation.get("created_at") or valuation.get("valued_at") or 0)
        valuation_times.append(valuation_ts)
        open_total += value

        age = Decimal(reference_ts - valuation_ts)
        if age > freshness_seconds:
            issues.append(
                f"Position #{position_id} {ticker} valuation is stale: "
                f"{format_age(age)} old at state write time"
            )

        formula_diff = value - quantity * price
        if abs(formula_diff) > Decimal("0.01"):
            issues.append(
                f"Position #{position_id} {ticker} value does not match quantity*price: "
                f"diff {formula_diff:.6f} USD"
            )

        stats = latest_position_stats(state, position_id)
        if stats is not None:
            stats_value = parse_decimal(stats.get("value"))
            stats_diff = stats_value - value
            if abs(stats_diff) > Decimal("0.01"):
                issues.append(
                    f"Position #{position_id} {ticker} latest stats value differs from valuation: "
                    f"{stats_diff:.6f} USD"
                )

        live_value_text = "-"
        drift_text = "-"
        lock_text = "-"
        if live_equities is not None and position_id in live_equities:
            live_value = parse_decimal(live_equities[position_id]["equity"])
            live_total += live_value
            drift = live_value - value
            live_value_text = f"{live_value:,.2f}"
            drift_text = f"{drift:+,.2f}"
            lock_expired = live_equities[position_id].get("lock_expired")
            if lock_expired is True:
                lock_text = "open"
            elif lock_expired is False:
                lock_text = "locked"
            else:
                lock_text = "?"

        print(
            f"  {position_id:>3}  {ticker:<16} "
            f"{value:>10,.2f} {quantity:>12,.2f} {price:>8,.5f} "
            f"{format_age(age):>6} {live_value_text:>10} {drift_text:>10} {lock_text:>7}"
        )

    cash = get_cash(state)
    computed_total = open_total + cash

    print()
    print(f"  Open position value from latest marks: {open_total:,.2f} USD")
    print(f"  Reserve cash: {cash:,.2f} USD")
    print(f"  Computed equity: {computed_total:,.2f} USD")

    if stats_row is not None:
        stats_total = parse_decimal(stats_row.get("total_equity"))
        stats_open = parse_decimal(stats_row.get("open_position_equity"))
        stats_cash = parse_decimal(stats_row.get("free_cash"))
        total_diff = computed_total - stats_total
        open_diff = open_total - stats_open
        cash_diff = cash - stats_cash
        print(f"  Latest stats equity: {stats_total:,.2f} USD (diff {total_diff:+,.6f})")
        print(f"  Latest stats open value: {stats_open:,.2f} USD (diff {open_diff:+,.6f})")
        print(f"  Latest stats cash: {stats_cash:,.2f} USD (diff {cash_diff:+,.6f})")
        if abs(total_diff) > Decimal("0.01"):
            issues.append(f"Computed equity differs from latest stats by {total_diff:.6f} USD")

    if valuation_times:
        print(
            f"  Open valuation timestamp range: "
            f"{format_time(min(valuation_times))} .. {format_time(max(valuation_times))}"
        )

    if live_equities is not None:
        live_total_with_cash = live_total + cash
        live_drift = live_total - open_total
        notes.append(
            f"Live Hyperliquid open equity is {live_total:,.2f} USD, "
            f"{live_drift:+,.2f} USD versus state marks"
        )
        print(f"  Live Hyperliquid open equity: {live_total:,.2f} USD (drift {live_drift:+,.2f})")
        print(f"  Live Hyperliquid equity plus cash: {live_total_with_cash:,.2f} USD")

    duplicate_groups: dict[str, list[int]] = {}
    for position in open_positions.values():
        pair = position.get("pair") or {}
        key = pair.get("pool_address") or str(get_pair_id(pair))
        duplicate_groups.setdefault(key, []).append(int(position["position_id"]))
    duplicates = [ids for ids in duplicate_groups.values() if len(ids) > 1]
    if duplicates:
        issues.append(f"Open duplicate position groups found: {duplicates}")
    else:
        print("  Open duplicate position groups: none")

    frozen_count = len(state.get("portfolio", {}).get("frozen_positions") or {})
    if frozen_count:
        issues.append(f"Frozen positions present: {frozen_count}")
    else:
        print("  Frozen positions: none")

    return issues, notes


def analyse_rebalance(state: dict[str, Any], amount_tolerance_pct: Decimal) -> tuple[list[str], list[str]]:
    """Analyse the latest Hyper AI rebalance decision."""
    issues = []
    notes = []
    trades = iter_trades(state)
    decisions = get_decision_messages(state)
    if not decisions:
        return ["No decision messages found"], notes

    decision = decisions[-1]
    decision_data = decision["data"]
    decision_timestamp = decision["timestamp"]
    decision_trades = [
        trade
        for trade in trades
        if int(trade.get("opened_at") or 0) == decision_timestamp
    ]
    decision_trades = sorted(decision_trades, key=lambda trade: int(trade.get("trade_id") or 0))

    threshold = parse_number_from_text(decision_data.get("Rebalance threshold")) or Decimal(0)
    alpha_model_timestamp = get_alpha_model_timestamp(state)
    alpha_model_matches_decision = alpha_model_timestamp == decision_timestamp
    if alpha_model_matches_decision:
        expected_signals = expected_trade_signals(state, threshold)
        by_position, by_pair = get_signal_lookup(state)
    else:
        expected_signals = []
        by_position, by_pair = {}, {}

    decided_count = int(decision_data.get("Trades decided", "0"))
    if len(decision_trades) != decided_count:
        issues.append(
            f"Decision said {decided_count} trades, but {len(decision_trades)} trades share the decision timestamp"
        )

    if alpha_model_matches_decision and len(expected_signals) != decided_count:
        issues.append(
            f"Alpha model has {len(expected_signals)} trade-sized adjustments, but decision said {decided_count} trades"
        )

    statuses = get_status_counts(decision_trades)
    unfinished = [
        trade
        for trade in decision_trades
        if get_trade_status(trade) not in {"success", "repaired", "expired"}
    ]
    if unfinished:
        issues.append(f"{len(unfinished)} latest decision trades are not executed or settled: {statuses}")

    tolerance = amount_tolerance_pct / Decimal(100)
    planned_trade_value = Decimal(0)
    trade_rows = []
    for trade in decision_trades:
        signal = get_trade_signal(trade, by_position, by_pair)
        ticker = get_ticker(trade.get("pair") or {})
        planned_reserve = abs(parse_decimal(trade.get("planned_reserve")))
        planned_trade_value += planned_reserve
        trade_direction = get_trade_direction(trade)
        signal_adjust = None
        amount_diff_pct = None

        if alpha_model_matches_decision and signal is not None:
            signal_adjust = parse_decimal(signal.get("position_adjust_usd"))
            signal_direction = get_signal_direction(signal)
            if signal_direction != trade_direction:
                issues.append(
                    f"Trade #{trade.get('trade_id')} {ticker} direction {trade_direction} "
                    f"does not match signal {signal_direction}"
                )

            expected_amount = abs(signal_adjust)
            if expected_amount > 0:
                amount_diff_pct = abs(planned_reserve - expected_amount) / expected_amount
                if amount_diff_pct > tolerance:
                    issues.append(
                        f"Trade #{trade.get('trade_id')} {ticker} amount differs from signal by "
                        f"{amount_diff_pct * Decimal(100):.2f}%"
                    )
        elif alpha_model_matches_decision:
            issues.append(f"Trade #{trade.get('trade_id')} {ticker} has no matching alpha model signal")

        trade_rows.append(
            {
                "id": trade.get("trade_id"),
                "ticker": ticker,
                "direction": trade_direction,
                "planned_reserve": planned_reserve,
                "executed_reserve": abs(parse_decimal(trade.get("executed_reserve"))),
                "status": get_trade_status(trade),
                "signal_adjust": signal_adjust,
                "amount_diff_pct": amount_diff_pct,
            }
        )

    reported_rebalance_volume = parse_number_from_text(decision_data.get("Rebalance volume"))
    if (
        reported_rebalance_volume is not None
        and planned_trade_value > Decimal("0.01")
        and abs(reported_rebalance_volume - planned_trade_value) > Decimal("0.01")
    ):
        notes.append(
            f"Decision report Rebalance volume is {reported_rebalance_volume:,.2f} USD, "
            f"but planned decision trade notional is {planned_trade_value:,.2f} USD"
        )

    print()
    print("Rebalance checks")
    print(f"  Decision analysed: cycle #{decision['cycle']} at {format_time(decision_timestamp)}")
    print(f"  Alpha model snapshot: {format_time(alpha_model_timestamp)}")
    for key in [
        "Rebalanced",
        "Trades decided",
        "Max position value change",
        "Rebalance threshold",
        "Total equity",
        "Cash",
        "Redeemable capital",
        "Locked capital carried forward",
        "Investable equity",
        "Accepted investable equity",
        "Allocated to signals",
        "Rebalance volume",
        "Blocked redemption checks",
        "Blocked redemption reasons",
    ]:
        value = decision_data.get(key)
        if value is not None:
            print(f"  {key}: {value}")
    print(f"  Decision trade statuses: {statuses}")
    print("  Latest decision trades")
    for row in trade_rows:
        expected = "-"
        if row["signal_adjust"] is not None:
            expected = f"{row['signal_adjust']:.4f}"
        diff = "-"
        if row["amount_diff_pct"] is not None:
            diff = f"{row['amount_diff_pct'] * Decimal(100):.2f}%"
        print(
            f"    #{row['id']:>3} {row['ticker']:<18} {row['direction']:<4} "
            f"planned {row['planned_reserve']:>10,.4f} "
            f"executed {row['executed_reserve']:>10,.4f} "
            f"expected {expected:>11} diff {diff:>7} status={row['status']}"
        )

    return issues, notes


def analyse_recent_incidents(state: dict[str, Any], since_ts: int) -> tuple[list[str], list[str]]:
    """Summarise recent failure and repair patterns."""
    issues = []
    notes = []
    trades = iter_trades(state)
    recent = [trade for trade in trades if int(trade.get("opened_at") or 0) >= since_ts]
    failed = [trade for trade in recent if trade.get("failed_at")]
    repaired_originals = [trade for trade in recent if trade.get("repaired_at")]
    repair_trades = [trade for trade in recent if trade.get("repaired_trade_id")]
    unrepaired_failed = [trade for trade in failed if not trade.get("repaired_at")]
    decisions = [decision for decision in get_decision_messages(state) if decision["timestamp"] >= since_ts]

    print()
    print("Recent incident checks")
    print(f"  Window starts: {format_time(since_ts)}")
    print(f"  Trades opened in window: {len(recent)}")
    print(f"  Failed trades in window: {len(failed)}")
    print(f"  Repaired original trades in window: {len(repaired_originals)}")
    print(f"  Repair trades in window: {len(repair_trades)}")

    if unrepaired_failed:
        issues.append(f"Unrepaired failed trades remain: {[trade.get('trade_id') for trade in unrepaired_failed]}")
    else:
        print("  Unrepaired failed trades: none")

    if decisions:
        print("  Recent decision history")
        for decision in decisions[-10:]:
            decision_trades = [
                trade for trade in trades if int(trade.get("opened_at") or 0) == decision["timestamp"]
            ]
            statuses = get_status_counts(decision_trades)
            decided = decision["data"].get("Trades decided", "?")
            print(
                f"    {format_time(decision['timestamp'])} "
                f"cycle #{decision['cycle']} decided {decided}, "
                f"state trades {len(decision_trades)}, statuses {statuses}"
            )

    if repaired_originals:
        notes.append(
            "Recent repaired trades are still visible in history; "
            "latest live cycle no longer has unrepaired failures"
        )

    return issues, notes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="State JSON file path, executor base URL, or executor /state URL")
    parser.add_argument("--live", action="store_true", help="Compare current Hyperliquid live equity with state marks")
    parser.add_argument(
        "--freshness-hours",
        type=Decimal,
        default=Decimal("4"),
        help="Maximum allowed valuation age at the state write timestamp",
    )
    parser.add_argument(
        "--amount-tolerance-pct",
        type=Decimal,
        default=Decimal("15"),
        help="Allowed planned amount vs alpha model adjustment difference",
    )
    parser.add_argument(
        "--since",
        default="2026-04-14T00:00:00",
        help="Incident summary start time as a naive UTC ISO timestamp",
    )
    args = parser.parse_args()

    state = load_state(args.source)
    print_state_summary(state)

    live_equities = None
    if args.live:
        safe_address = resolve_lagoon_safe(state)
        print(f"  Lagoon Safe for live Hyperliquid comparison: {safe_address}")
        live_equities = fetch_live_equities(state, safe_address)

    valuation_issues, valuation_notes = analyse_valuations(
        state,
        freshness_hours=args.freshness_hours,
        live_equities=live_equities,
    )
    rebalance_issues, rebalance_notes = analyse_rebalance(
        state,
        amount_tolerance_pct=args.amount_tolerance_pct,
    )
    incident_issues, incident_notes = analyse_recent_incidents(state, parse_since(args.since))

    issues = valuation_issues + rebalance_issues + incident_issues
    notes = valuation_notes + rebalance_notes + incident_notes

    print()
    if notes:
        print("Notes")
        for note in notes:
            print(f"  - {note}")

    print()
    if issues:
        print("Result: FAIL")
        for issue in issues:
            print(f"  - {issue}")
        raise SystemExit(1)

    print("Result: PASS")


if __name__ == "__main__":
    main()
