"""Download and sanity check a remote trade-executor state.

Fetches state JSON from a running trade-executor, displays position tables,
runs validation checks, and exits with code 1 if any check fails.

For vault strategies, also displays vault sync state including pending
deposits/redemptions (requires JSON_RPC_* env var for the vault's chain).

Usage::

    source .local-test.env && poetry run python scripts/check-state.py https://hyper-ai.tradingstrategy.ai
"""

import datetime
import logging
import math
import os
import sys

import requests
from tabulate import tabulate

from tradeexecutor.analysis.trade_analyser import build_trade_analysis
from tradeexecutor.cli.double_position import check_double_position
from tradeexecutor.state.balance_update import BalanceUpdateCause
from tradeexecutor.utils.state_downloader import download_state

logger = logging.getLogger(__name__)


def format_duration(td: datetime.timedelta | None) -> str:
    if td is None:
        return "-"
    days = td.days
    hours = td.seconds // 3600
    return f"{days}d {hours}h"


def format_time(dt: datetime.datetime | None) -> str:
    if dt is None:
        return "-"
    return dt.strftime("%Y-%m-%d %H:%M")


def format_pct(val: float | None) -> str:
    if val is None:
        return "-"
    return f"{val * 100:+.2f}%"


def format_usd(val: float | None) -> str:
    if val is None:
        return "-"
    return f"${val:,.2f}"


def format_decimal(val) -> str:
    if val is None:
        return "-"
    return f"{float(val):,.6f}"


def get_asset_management_mode(url: str) -> str | None:
    """Fetch asset_management_mode from the executor's /metadata endpoint."""
    try:
        resp = requests.get(f"{url}/metadata", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            on_chain = data.get("on_chain_data", {})
            return on_chain.get("asset_management_mode")
    except Exception as e:
        logger.warning("Could not fetch metadata: %s", e)
    return None


def get_vault_sync_info_from_state(state) -> dict:
    """Extract vault sync information from state data (no web3 needed)."""
    treasury = state.sync.treasury
    deployment = state.sync.deployment

    info = {
        "vault_address": deployment.address,
        "chain_id": deployment.chain_id,
        "vault_name": deployment.vault_token_name,
        "vault_symbol": deployment.vault_token_symbol,
        "last_vault_sync": treasury.last_updated_at,
        "last_block_scanned": treasury.last_block_scanned,
        "treasury_share_count": treasury.share_count,
        "treasury_pending_redemptions": treasury.pending_redemptions,
    }

    # Find last deposit timestamp from balance update refs
    deposit_refs = [
        ref for ref in treasury.balance_update_refs
        if ref.cause in (BalanceUpdateCause.deposit, BalanceUpdateCause.deposit_and_redemption)
    ]
    if deposit_refs:
        last_deposit_ref = max(deposit_refs, key=lambda r: r.strategy_cycle_included_at or datetime.datetime.min)
        info["last_deposit_at"] = last_deposit_ref.strategy_cycle_included_at
        info["last_deposit_usd"] = last_deposit_ref.usd_value
    else:
        info["last_deposit_at"] = None
        info["last_deposit_usd"] = None

    # Calculate sync frequency from balance_update_refs timestamps (last 10)
    all_refs = sorted(
        [ref for ref in treasury.balance_update_refs if ref.strategy_cycle_included_at],
        key=lambda r: r.strategy_cycle_included_at,
    )
    recent_refs = all_refs[-10:] if len(all_refs) > 10 else all_refs
    if len(recent_refs) >= 2:
        intervals = []
        for i in range(1, len(recent_refs)):
            delta = recent_refs[i].strategy_cycle_included_at - recent_refs[i - 1].strategy_cycle_included_at
            intervals.append(delta)
        avg_interval = sum(intervals, datetime.timedelta()) / len(intervals)
        info["sync_frequency_avg"] = avg_interval
        info["sync_sample_count"] = len(recent_refs)
    else:
        info["sync_frequency_avg"] = None
        info["sync_sample_count"] = len(recent_refs)

    return info


def fetch_pending_deposits_redemptions_web3(chain_id, vault_address: str) -> dict | None:
    """Use web3 to query on-chain pending deposits and redemptions for Lagoon vaults.

    Returns None if RPC is not available or vault type is not supported.
    """
    try:
        from eth_defi.provider.multi_provider import create_multi_provider_web3
        from eth_defi.erc_4626.vault_protocol.lagoon.vault import LagoonVault
        from eth_defi.vault.base import VaultSpec
        from tradeexecutor.ethereum.web3config import get_rpc_env_var_name
    except ImportError as e:
        logger.warning("Cannot import web3 dependencies: %s", e)
        return None

    env_var = get_rpc_env_var_name(chain_id)
    rpc_url = os.environ.get(env_var)
    if not rpc_url:
        print(f"  (Skipping on-chain queries: {env_var} not set)")
        return None

    try:
        web3 = create_multi_provider_web3(rpc_url)
        spec = VaultSpec(chain_id=chain_id.value, vault_address=vault_address)
        vault = LagoonVault(web3, spec)
        flow_manager = vault.get_flow_manager()

        pending_deposit = flow_manager.fetch_pending_deposit("latest")
        pending_redemption = flow_manager.fetch_pending_redemption("latest")

        return {
            "pending_deposit": pending_deposit,
            "pending_redemption_shares": pending_redemption,
        }
    except Exception as e:
        print(f"  (On-chain query failed: {e})")
        return None


def display_vault_sync_state(url: str, state):
    """Display vault sync state section if this is a vault strategy."""
    deployment = state.sync.deployment

    # Detect if this is a vault strategy
    asset_mode = get_asset_management_mode(url)
    is_vault = asset_mode in ("enzyme", "lagoon", "velvet")

    # Fallback: check if deployment has vault-specific data
    if not is_vault and deployment.vault_token_name:
        is_vault = True

    if not is_vault:
        return

    print()
    print("=" * 60)
    print("VAULT SYNC STATE")
    print("=" * 60)

    info = get_vault_sync_info_from_state(state)

    rows = [
        ["Vault type", asset_mode or "unknown"],
        ["Vault address", info["vault_address"] or "-"],
        ["Chain", str(info["chain_id"].name) if info["chain_id"] else "-"],
        ["Vault name", info["vault_name"] or "-"],
        ["Vault symbol", info["vault_symbol"] or "-"],
        ["Last vault sync", format_time(info["last_vault_sync"])],
        ["Last block scanned", f"{info['last_block_scanned']:,}" if info["last_block_scanned"] else "-"],
        ["Last deposit at", format_time(info["last_deposit_at"])],
        ["Last deposit value", format_usd(info["last_deposit_usd"])],
        ["Share count (state)", format_decimal(info["treasury_share_count"])],
        ["Pending redemptions (state)", format_usd(info["treasury_pending_redemptions"])],
        ["Sync frequency (avg)", format_duration(info["sync_frequency_avg"])],
        ["Sync sample count", info["sync_sample_count"]],
    ]
    print(tabulate(rows, tablefmt="fancy_grid"))

    # On-chain pending deposits/redemptions via web3 (Lagoon only)
    if asset_mode == "lagoon" and info["chain_id"] and info["vault_address"]:
        print()
        print("On-chain pending flows (web3):")
        web3_data = fetch_pending_deposits_redemptions_web3(info["chain_id"], info["vault_address"])
        if web3_data:
            onchain_rows = [
                ["Pending deposits (underlying)", format_decimal(web3_data["pending_deposit"])],
                ["Pending redemptions (shares)", format_decimal(web3_data["pending_redemption_shares"])],
            ]
            print(tabulate(onchain_rows, tablefmt="fancy_grid"))


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/check-state.py <executor-url>")
        print("Example: python scripts/check-state.py https://hyper-ai.tradingstrategy.ai")
        sys.exit(1)

    url = sys.argv[1].rstrip("/")
    if url.endswith("/state"):
        url = url[: -len("/state")]

    # 1. Download state
    print(f"Downloading state from {url}...")
    state = download_state(url)
    portfolio = state.portfolio

    # 2. Overview
    print()
    print("=" * 60)
    print("OVERVIEW")
    print("=" * 60)
    overview = [
        ["Strategy name", state.name or "-"],
        ["Last updated", format_time(state.last_updated_at)],
        ["Cycle", state.cycle],
        ["Total equity", format_usd(portfolio.calculate_total_equity())],
        ["Cash reserves", format_usd(portfolio.get_cash())],
        ["NAV", format_usd(portfolio.get_net_asset_value())],
        ["Open positions", len(portfolio.open_positions)],
        ["Closed positions", len(portfolio.closed_positions)],
        ["Frozen positions", len(portfolio.frozen_positions)],
    ]
    print(tabulate(overview, tablefmt="fancy_grid"))

    # 3. Open positions
    print()
    print("=" * 60)
    print("OPEN POSITIONS")
    print("=" * 60)
    if portfolio.open_positions:
        rows = []
        for pos in sorted(portfolio.open_positions.values(), key=lambda p: p.get_value(), reverse=True):
            rows.append({
                "Id": pos.position_id,
                "Ticker": pos.pair.get_ticker(),
                "Value": format_usd(pos.get_value()),
                "Qty": f"{pos.get_quantity():,.6f}",
                "Unreal PnL $": format_usd(pos.get_unrealised_profit_usd()),
                "Unreal PnL %": format_pct(pos.get_unrealised_profit_pct()),
                "Duration": format_duration(pos.get_duration()),
                "Last priced": format_time(pos.last_pricing_at),
                "Trades": pos.get_trade_count(),
            })
        print(tabulate(rows, headers="keys", tablefmt="fancy_grid"))
    else:
        print("No open positions")

    # 4. Closed positions
    print()
    print("=" * 60)
    print("CLOSED POSITIONS")
    print("=" * 60)
    if portfolio.closed_positions:
        rows = []
        for pos in sorted(portfolio.closed_positions.values(), key=lambda p: p.position_id):
            rows.append({
                "Id": pos.position_id,
                "Ticker": pos.pair.get_ticker(),
                "Realised PnL $": format_usd(pos.get_realised_profit_usd()),
                "Realised PnL %": format_pct(pos.get_realised_profit_percent()),
                "Duration": format_duration(pos.get_duration()),
                "Trades": pos.get_trade_count(),
            })
        print(tabulate(rows, headers="keys", tablefmt="fancy_grid"))
    else:
        print("No closed positions")

    # 5. Trade summary
    print()
    print("=" * 60)
    print("TRADE SUMMARY")
    print("=" * 60)
    try:
        analysis = build_trade_analysis(portfolio)
        summary = analysis.calculate_summary_statistics()
        summary_rows = [
            ["Won", summary.won],
            ["Lost", summary.lost],
            ["Zero loss", summary.zero_loss],
            ["Stop losses", summary.stop_losses],
            ["Undecided (open)", summary.undecided],
            ["Realised profit", format_usd(summary.realised_profit)],
            ["Open value", format_usd(summary.open_value)],
            ["Uninvested cash", format_usd(summary.uninvested_cash)],
            ["Avg winning trade %", format_pct(summary.average_winning_trade_profit_pc) if summary.average_winning_trade_profit_pc else "-"],
            ["Avg losing trade %", format_pct(summary.average_losing_trade_loss_pc) if summary.average_losing_trade_loss_pc else "-"],
            ["Biggest winner %", format_pct(summary.biggest_winning_trade_pc) if summary.biggest_winning_trade_pc else "-"],
            ["Biggest loser %", format_pct(summary.biggest_losing_trade_pc) if summary.biggest_losing_trade_pc else "-"],
        ]
        print(tabulate(summary_rows, tablefmt="fancy_grid"))
    except Exception as e:
        print(f"Could not calculate trade summary: {e}")

    # 6. Vault sync state (if vault strategy)
    display_vault_sync_state(url, state)

    # 7. Validation checks
    print()
    print("=" * 60)
    print("VALIDATION CHECKS")
    print("=" * 60)

    failures = []

    def check(name: str, passed: bool, detail: str = ""):
        status = "PASS" if passed else "FAIL"
        msg = f"  [{status}] {name}"
        if detail:
            msg += f" - {detail}"
        print(msg)
        if not passed:
            failures.append(name)

    # No frozen positions
    check(
        "No frozen positions",
        len(portfolio.frozen_positions) == 0,
        f"{len(portfolio.frozen_positions)} frozen" if portfolio.frozen_positions else "",
    )

    # No double positions
    has_doubles = check_double_position(state, printer=lambda x: None)
    check("No double positions", not has_doubles)

    # Portfolio value > 0
    equity = portfolio.calculate_total_equity()
    check("Portfolio value > 0", equity > 0, format_usd(equity))

    # All open position trades succeeded
    failed_trades = []
    for pos in portfolio.open_positions.values():
        for t in pos.trades.values():
            if t.is_failed():
                failed_trades.append(f"trade {t.trade_id} in position {pos.position_id}")
    check(
        "No failed trades in open positions",
        len(failed_trades) == 0,
        ", ".join(failed_trades) if failed_trades else "",
    )

    # No NaN/inf in position values
    bad_values = []
    for pos in portfolio.open_positions.values():
        val = pos.get_value()
        if not math.isfinite(val):
            bad_values.append(f"position {pos.position_id} value={val}")
    for pos in portfolio.closed_positions.values():
        pnl = pos.get_realised_profit_usd()
        if pnl is not None and not math.isfinite(pnl):
            bad_values.append(f"position {pos.position_id} pnl={pnl}")
    check(
        "No NaN/inf in position values",
        len(bad_values) == 0,
        ", ".join(bad_values) if bad_values else "",
    )

    # Pricing freshness for open positions
    now = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
    stale = []
    for pos in portfolio.open_positions.values():
        if pos.last_pricing_at and (now - pos.last_pricing_at) > datetime.timedelta(hours=1):
            age = now - pos.last_pricing_at
            stale.append(f"{pos.pair.get_ticker()} ({format_duration(age)} ago)")
    check(
        "Open position pricing fresh (<1h)",
        len(stale) == 0,
        ", ".join(stale) if stale else "",
    )

    # No failed trades anywhere
    all_failed = [t for t in portfolio.get_all_trades() if t.is_failed()]
    check(
        "No failed trades in portfolio",
        len(all_failed) == 0,
        f"{len(all_failed)} failed trades" if all_failed else "",
    )

    # Final result
    print()
    if failures:
        print(f"RESULT: FAIL ({len(failures)} check(s) failed: {', '.join(failures)})")
        sys.exit(1)
    else:
        print("RESULT: ALL CHECKS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
