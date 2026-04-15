"""Check the latest Hyper AI rebalance decision against state trades.

Usage::

    source .local-test.env && poetry run python scripts/hyper-ai/check-rebalance.py https://hyper-ai.tradingstrategy.ai/state

The script reads the latest strategy decision message from the state,
finds trades opened at the same decision timestamp, and compares them with
the alpha model signal adjustments saved in ``visualisation.discardable_data``.
"""

import argparse
import datetime
import json
import re
import sys
import urllib.request
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from eth_defi.compat import native_datetime_utc_fromtimestamp


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


def load_state(source: str) -> dict[str, Any]:
    """Load state JSON from a file path or an executor URL."""
    if source.startswith("http://") or source.startswith("https://"):
        url = source.rstrip("/")
        if not url.endswith("/state"):
            url += "/state"
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "trade-executor-hyper-ai-check/1.0"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))

    return json.loads(Path(source).read_text())


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
                trade["_position_bucket"] = bucket_name
                trades.append(trade)
    return trades


def get_trade_status(trade: dict[str, Any]) -> str:
    """Resolve trade status from persisted trade timestamps."""
    if trade.get("repaired_trade_id"):
        return "success"
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


def get_pair_id(pair: dict[str, Any]) -> int | None:
    """Read a pair internal id."""
    value = pair.get("internal_id")
    if value is None:
        return None
    return int(value)


def get_ticker(pair: dict[str, Any]) -> str:
    """Format a human-friendly pair ticker."""
    base = pair.get("base", {}).get("token_symbol") or "?"
    quote = pair.get("quote", {}).get("token_symbol") or "?"
    return f"{base}-{quote}"


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


def choose_decision(state: dict[str, Any], mode: str) -> dict[str, Any]:
    """Pick which decision message to analyse."""
    decisions = get_decision_messages(state)
    if not decisions:
        raise RuntimeError("No decision messages found in state.visualisation.messages")

    if mode == "latest":
        return decisions[-1]

    latest_cycle, completed_at = get_latest_completed_cycle_time(state)
    if latest_cycle is None or completed_at is None:
        raise RuntimeError("No completed cycle timestamps found in state.uptime.cycles_completed_at")

    completed_decisions = [
        decision
        for decision in decisions
        if decision["cycle"] == latest_cycle and decision["timestamp"] <= completed_at
    ]
    if not completed_decisions:
        raise RuntimeError(f"No decision message found for latest completed cycle #{latest_cycle}")
    return completed_decisions[-1]


def get_status_counts(trades: list[dict[str, Any]]) -> dict[str, int]:
    """Count trades by status."""
    counts = {}
    for trade in trades:
        status = get_trade_status(trade)
        counts[status] = counts.get(status, 0) + 1
    return counts


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


def get_alpha_model_timestamp(state: dict[str, Any]) -> int | None:
    """Return the timestamp of the saved alpha model snapshot."""
    alpha_model = state.get("visualisation", {}).get("discardable_data", {}).get("alpha_model") or {}
    timestamp = alpha_model.get("timestamp")
    if timestamp is None:
        return None
    return int(timestamp)


def expected_trade_signals(
    state: dict[str, Any],
    threshold: Decimal,
) -> list[dict[str, Any]]:
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


def print_history(state: dict[str, Any], trades: list[dict[str, Any]], count: int) -> None:
    """Print recent decision history."""
    if count <= 0:
        return

    print()
    print("Recent decision history")
    for decision in get_decision_messages(state)[-count:]:
        decision_trades = [trade for trade in trades if int(trade.get("opened_at") or 0) == decision["timestamp"]]
        statuses = get_status_counts(decision_trades)
        decided = decision["data"].get("Trades decided", "?")
        print(
            f"  {format_time(decision['timestamp'])} "
            f"cycle #{decision['cycle']} decided {decided}, "
            f"state trades {len(decision_trades)}, statuses {statuses}"
        )


def check_rebalance(args: argparse.Namespace) -> int:
    """Run the Hyper AI rebalance check."""
    state = load_state(args.source)
    trades = iter_trades(state)
    decision = choose_decision(state, args.decision)
    decision_data = decision["data"]
    decision_timestamp = decision["timestamp"]
    decision_trades = [
        trade
        for trade in trades
        if int(trade.get("opened_at") or 0) == decision_timestamp
    ]
    decision_trades = sorted(decision_trades, key=lambda trade: int(trade.get("trade_id") or 0))

    latest_completed_cycle, completed_at = get_latest_completed_cycle_time(state)
    threshold = parse_number_from_text(decision_data.get("Rebalance threshold")) or Decimal("0")
    alpha_model_timestamp = get_alpha_model_timestamp(state)
    alpha_model_matches_decision = alpha_model_timestamp == decision_timestamp
    if alpha_model_matches_decision:
        expected_signals = expected_trade_signals(state, threshold)
        by_position, by_pair = get_signal_lookup(state)
    else:
        expected_signals = []
        by_position, by_pair = {}, {}

    issues = []
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
        status_text = ", ".join(f"{status}={count}" for status, count in sorted(statuses.items()))
        issues.append(f"{len(unfinished)} decision trades are not executed/settled ({status_text})")

    amount_tolerance = Decimal(str(args.amount_tolerance_pct)) / Decimal(100)
    trade_rows = []
    for trade in decision_trades:
        signal = get_trade_signal(trade, by_position, by_pair)
        ticker = get_ticker(trade.get("pair") or {})
        planned_reserve = abs(parse_decimal(trade.get("planned_reserve")))
        trade_direction = get_trade_direction(trade)
        status = get_trade_status(trade)
        signal_adjust = None
        amount_diff_pct = None
        direction_ok = None

        if not alpha_model_matches_decision:
            pass
        elif signal is None:
            issues.append(f"Trade #{trade.get('trade_id')} {ticker} has no matching alpha model signal")
        else:
            signal_adjust = parse_decimal(signal.get("position_adjust_usd"))
            signal_direction = get_signal_direction(signal)
            direction_ok = signal_direction == trade_direction
            if not direction_ok:
                issues.append(
                    f"Trade #{trade.get('trade_id')} {ticker} direction {trade_direction} "
                    f"does not match signal {signal_direction}"
                )

            expected_amount = abs(signal_adjust)
            if expected_amount > 0:
                amount_diff_pct = abs(planned_reserve - expected_amount) / expected_amount
                if amount_diff_pct > amount_tolerance:
                    issues.append(
                        f"Trade #{trade.get('trade_id')} {ticker} amount differs from signal by "
                        f"{amount_diff_pct * Decimal(100):.2f}%"
                    )

        trade_rows.append(
            {
                "id": trade.get("trade_id"),
                "ticker": ticker,
                "direction": trade_direction,
                "amount": planned_reserve,
                "signal_adjust": signal_adjust,
                "diff_pct": amount_diff_pct,
                "status": status,
                "direction_ok": direction_ok,
                "flags": ",".join(trade.get("flags") or []),
            }
        )

    print("Hyper AI rebalance check")
    print(f"Source: {args.source}")
    print(f"State last updated: {format_time(state.get('last_updated_at'))}")
    print(f"State cycle: {state.get('cycle')}")
    if latest_completed_cycle is not None:
        print(f"Latest completed cycle: #{latest_completed_cycle} at {format_time(completed_at)}")
    print(f"Decision analysed: cycle #{decision['cycle']} at {format_time(decision_timestamp)} ({args.decision})")
    if not alpha_model_matches_decision:
        print(
            "Alpha model snapshot: "
            f"{format_time(alpha_model_timestamp)}; skipping historical amount matching"
        )
    print()
    print("Decision summary")
    for key in [
        "Rebalanced",
        "Trades decided",
        "Max position value change",
        "Rebalance threshold",
        "Selected survivor signals",
        "Total equity",
        "Cash",
        "Redeemable capital",
        "Locked capital carried forward",
        "Investable equity",
        "Accepted investable equity",
        "Allocated to signals",
        "Rebalance volume",
    ]:
        value = decision_data.get(key)
        if value is not None:
            print(f"  {key}: {value}")
    print(f"  Trade statuses: {statuses}")

    print()
    print("Decision trades")
    for row in trade_rows:
        expected = "-"
        if row["signal_adjust"] is not None:
            expected = f"{row['signal_adjust']:.4f}"
        diff = "-"
        if row["diff_pct"] is not None:
            diff = f"{row['diff_pct'] * Decimal(100):.2f}%"
        flags = f" flags={row['flags']}" if row["flags"] else ""
        print(
            f"  #{row['id']:>3} {row['ticker']:<18} {row['direction']:<4} "
            f"{row['amount']:>10.4f} expected {expected:>11} "
            f"diff {diff:>7} status={row['status']}{flags}"
        )

    print_history(state, trades, args.history)

    print()
    if issues:
        print("Result: FAIL")
        for issue in issues:
            print(f"  - {issue}")
        return 1

    print("Result: PASS")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        help="State JSON file path, executor base URL, or executor /state URL",
    )
    parser.add_argument(
        "--decision",
        choices=("latest", "latest-completed"),
        default="latest",
        help="Which decision message to analyse",
    )
    parser.add_argument(
        "--amount-tolerance-pct",
        type=float,
        default=15.0,
        help="Allowed planned amount vs alpha model adjustment difference",
    )
    parser.add_argument(
        "--history",
        type=int,
        default=5,
        help="Number of recent decision history rows to print",
    )
    raise SystemExit(check_rebalance(parser.parse_args()))


if __name__ == "__main__":
    main()
