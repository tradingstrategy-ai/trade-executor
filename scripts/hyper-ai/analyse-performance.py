"""Analyse Hyper AI live state performance.

This script reads a trade-executor state JSON file or URL and prints a compact
performance breakdown for the Hyper AI strategy.
"""

import argparse
import datetime
import json
import math
import sys
import urllib.request
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any


EPOCH = datetime.datetime(1970, 1, 1)


def format_ts(value: int | float | str | None) -> str:
    if not value:
        return "-"
    return (EPOCH + datetime.timedelta(seconds=int(value))).strftime("%Y-%m-%d %H:%M:%S UTC")


def format_money(value: float | Decimal | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):,.2f}"


def format_pct(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "-"
    return f"{value * 100:,.2f}%"


def parse_decimal(value: Any) -> Decimal:
    if value is None:
        return Decimal(0)
    return Decimal(str(value))


def load_state(source: str) -> dict[str, Any]:
    if source.startswith("http://") or source.startswith("https://"):
        request = urllib.request.Request(
            source,
            headers={
                "User-Agent": "trade-executor-hyper-ai-performance-analysis/1.0",
            },
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read())

    return json.loads(Path(source).read_text())


def get_positions(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    portfolio = state["portfolio"]
    positions = {}
    for bucket in ("open_positions", "closed_positions", "frozen_positions"):
        for position_id, position in portfolio.get(bucket, {}).items():
            position["_status"] = bucket.removesuffix("_positions")
            positions[position_id] = position
    return positions


def get_position_symbol(position: dict[str, Any]) -> str:
    other_data = position["pair"]["base"].get("other_data") or {}
    return other_data.get("vault_name") or position["pair"]["base"]["token_symbol"]


def print_table(headers: list[str], rows: list[list[Any]]) -> None:
    widths = [
        max(len(str(header)), *(len(str(row[index])) for row in rows)) if rows else len(str(header))
        for index, header in enumerate(headers)
    ]
    print("  ".join(str(header).ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(str(value).ljust(widths[index]) for index, value in enumerate(row)))


def analyse_portfolio(state: dict[str, Any]) -> None:
    stats = state["stats"]["portfolio"]
    first = stats[0]
    last = stats[-1]
    deposits = [
        ref for ref in state.get("sync", {}).get("treasury", {}).get("balance_update_refs", [])
        if abs(float(ref.get("usd_value") or 0)) > 0.000001
    ]

    share_prices = [
        (int(row["calculated_at"]), float(row["share_price_usd"]), float(row["total_equity"]))
        for row in stats
        if row.get("share_price_usd") is not None
    ]
    peak_ts, peak_share_price, peak_equity = max(share_prices, key=lambda item: item[1])
    trough_ts, trough_share_price, trough_equity = min(share_prices, key=lambda item: item[1])

    running_peak = share_prices[0]
    max_drawdown = (0.0, share_prices[0], share_prices[0])
    for item in share_prices:
        if item[1] > running_peak[1]:
            running_peak = item
        drawdown = item[1] / running_peak[1] - 1
        if drawdown < max_drawdown[0]:
            max_drawdown = (drawdown, running_peak, item)

    print("Portfolio")
    print(f"State updated: {format_ts(state['last_updated_at'])}")
    print(f"Stats rows: {len(stats)}")
    print(f"External deposits/redemptions in state: {format_money(sum(float(ref['usd_value']) for ref in deposits))} USD")
    print(f"Current equity: {format_money(last['total_equity'])} USD")
    print(f"Current cash: {format_money(last['free_cash'])} USD")
    print(f"Open position equity: {format_money(last['open_position_equity'])} USD")
    print(f"Current share price: {last['share_price_usd']:.6f} USD")
    print(f"Reported unrealised strategy return: {format_pct(last.get('unrealised_profitability'))}")
    print(f"Reported realised PnL: {format_money(last.get('realised_profit_usd'))} USD")
    print(f"Reported unrealised PnL: {format_money(last.get('unrealised_profit_usd'))} USD")
    print(f"Peak share price: {peak_share_price:.6f} on {format_ts(peak_ts)}")
    print(f"Lowest share price: {trough_share_price:.6f} on {format_ts(trough_ts)}")
    print(
        "Max share-price drawdown: "
        f"{format_pct(max_drawdown[0])} from {format_ts(max_drawdown[1][0])} to {format_ts(max_drawdown[2][0])}"
    )
    print()


def analyse_share_price_moves(state: dict[str, Any], limit: int) -> None:
    rows = []
    previous = None
    for row in state["stats"]["portfolio"]:
        if previous is not None:
            previous_share_price = float(previous["share_price_usd"])
            share_price = float(row["share_price_usd"])
            change = share_price / previous_share_price - 1
            equity_change = float(row["total_equity"]) - float(previous["total_equity"])
            rows.append((change, equity_change, previous, row))
        previous = row

    print("Worst share-price moves")
    table = []
    for change, equity_change, previous, row in sorted(rows, key=lambda item: item[0])[:limit]:
        table.append([
            format_ts(row["calculated_at"]),
            format_pct(change),
            format_money(equity_change),
            format_money(row["total_equity"]),
            format_money(row["free_cash"]),
            row["open_position_count"],
        ])
    print_table(["at", "share price move", "equity move USD", "equity USD", "cash USD", "open positions"], table)
    print()


def analyse_position_stats(state: dict[str, Any], limit: int) -> None:
    positions = get_positions(state)
    rows = []
    for position_id, stats in (state.get("stats", {}).get("positions") or {}).items():
        if not stats:
            continue
        position = positions.get(position_id)
        if not position:
            continue
        latest = stats[-1]
        rows.append({
            "position_id": int(position_id),
            "symbol": get_position_symbol(position),
            "status": position["_status"],
            "value": float(latest.get("value") or 0),
            "profit_usd": float(latest.get("profit_usd") or 0),
            "profit_pct": float(latest.get("profitability") or 0),
            "internal_profit_usd": latest.get("internal_profit_usd"),
            "internal_profit_pct": latest.get("internal_profit_pct"),
        })

    print("Current open losses")
    table = []
    for row in sorted((row for row in rows if row["status"] == "open"), key=lambda item: item["profit_usd"])[:limit]:
        table.append([
            row["position_id"],
            row["symbol"][:32],
            format_money(row["value"]),
            format_money(row["profit_usd"]),
            format_pct(row["profit_pct"]),
            format_money(row["internal_profit_usd"]),
            format_pct(row["internal_profit_pct"]),
        ])
    print_table(["id", "vault", "value USD", "reported PnL USD", "reported PnL", "share PnL USD", "share PnL"], table)
    print()

    print("Closed realised losses")
    table = []
    for row in sorted((row for row in rows if row["status"] == "closed"), key=lambda item: item["profit_usd"])[:limit]:
        table.append([
            row["position_id"],
            row["symbol"][:32],
            format_money(row["profit_usd"]),
            format_pct(row["profit_pct"]),
            format_money(row["internal_profit_usd"]),
            format_pct(row["internal_profit_pct"]),
        ])
    print_table(["id", "vault", "reported PnL USD", "reported PnL", "share PnL USD", "share PnL"], table)
    print()


def analyse_balance_events(state: dict[str, Any], limit: int) -> None:
    positions = get_positions(state)
    rows = []
    grouped = defaultdict(float)
    for position_id, position in positions.items():
        for update in (position.get("balance_updates") or {}).values():
            value = float(update.get("usd_value") or 0)
            if value == 0:
                continue
            rows.append({
                "at": update.get("strategy_cycle_included_at") or update.get("created_at"),
                "value": value,
                "position_id": int(position_id),
                "status": position["_status"],
                "symbol": get_position_symbol(position),
                "update_id": update["balance_update_id"],
            })
            grouped[(position_id, get_position_symbol(position))] += value

    print("Largest negative vault-flow balance updates")
    table = []
    for row in sorted(rows, key=lambda item: item["value"])[:limit]:
        table.append([
            format_ts(row["at"]),
            row["position_id"],
            row["symbol"][:32],
            row["status"],
            format_money(row["value"]),
            row["update_id"],
        ])
    print_table(["at", "id", "vault", "status", "usd value", "update"], table)
    print()

    print("Most negative net vault-flow by position")
    table = []
    for (position_id, symbol), value in sorted(grouped.items(), key=lambda item: item[1])[:limit]:
        table.append([position_id, symbol[:32], format_money(value)])
    print_table(["id", "vault", "net vault-flow USD"], table)
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("state", help="State JSON file or URL")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    state = load_state(args.state)
    analyse_portfolio(state)
    analyse_share_price_moves(state, args.limit)
    analyse_position_stats(state, args.limit)
    analyse_balance_events(state, args.limit)


if __name__ == "__main__":
    sys.exit(main())
