"""Reproduce DOEZOE's 2026 deposit table using immutable recovery snapshots.

Example, from the repository root with its Poetry environment active::

    source .local-test.env
    PYTHONPATH="$PWD/deps/trading-strategy:$PWD:$PYTHONPATH" poetry run python \
        scripts/hyperliquid/reproduce-doezoe-deposit-status.py \
        --recovery-root /home/mikko/hypercore-recovery-1628 \
        --output-dir docs/reports

The retained producer-cleaned snapshot is the client input. The independent
producer reference was generated before this client fix; it is never rewritten.
All source files are read-only. Decisions after the snapshot cutoff are omitted
from validation but remain present as unevaluated rows in the full calendar.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import duckdb
import pandas as pd

from tradeexecutor.backtest.backtest_pricing import BacktestPricing
from tradeexecutor.ethereum.vault.hypercore_vault import create_hypercore_vault_pair
from tradeexecutor.state.identifier import AssetIdentifier
from tradingstrategy.alternative_data.vault import HYPERCORE_DEPOSIT_STATE_CUTOFF, convert_vault_prices_to_vault_state, read_vault_permission_history_parquet
from tradingstrategy.candle import GroupedCandleUniverse


ADDRESS = "0xcae0d1558b70b92ee9fd0acb20cb639c8c28ae69"
AS_OF = pd.Timestamp("2026-10-06")
STEM = "doezoe-deposit-status-2026-after-client-fix"


def status(value: object) -> str:
    """Distinguish unknown state from an explicit closed flag."""
    if value is None or pd.isna(value) or value == -1 or value == "":
        return "Unknown"
    if isinstance(value, str):
        return "Open" if value.lower() == "true" else "Closed"
    return "Open" if bool(value) else "Closed"


def timestamp(value: object) -> str:
    """Show the original naive UTC clock, including subsecond precision."""
    return "—" if value is None or pd.isna(value) else pd.Timestamp(value).isoformat(sep=" ")


def coverage(frame: pd.DataFrame, days: pd.DatetimeIndex, prefix: str) -> pd.DataFrame:
    """Count price rows and stored flags without claiming independent receipts."""
    frame = frame.copy()
    frame["day"] = pd.to_datetime(frame.timestamp).dt.normalize()
    frame["flag"] = frame.deposits_open.map(status)
    grouped = frame.groupby("day")
    result = grouped.agg(
        rows=("timestamp", "size"),
        first_price=("timestamp", "min"), last_price=("timestamp", "max"),
        first_write=("written_at", "min"), last_write=("written_at", "max"),
    ).reindex(days)
    result["rows"] = result["rows"].fillna(0).astype(int)
    result["flags"] = grouped.flag.agg(lambda values: " and ".join(sorted(set(values), reverse=True))).reindex(days).fillna("—")
    result["write_dates"] = grouped.written_at.agg(
        lambda values: ", ".join(sorted(set(pd.to_datetime(values.dropna()).dt.strftime("%Y-%m-%d")))) or "—"
    ).reindex(days).fillna("—")
    return result.add_prefix(prefix + "_")


def framework(prices: pd.DataFrame, days: pd.DatetimeIndex, history: pd.DataFrame | None, prefix: str) -> pd.DataFrame:
    """Exercise the fixed client converter and the executor's actual deposit gate."""
    state = convert_vault_prices_to_vault_state(prices, "1d", history)
    assert state is not None and not state.empty
    model = BacktestPricing(GroupedCandleUniverse.create_empty(), None, data_delay_tolerance=pd.Timedelta(days=2), vault_state=state)
    pair = create_hypercore_vault_pair(AssetIdentifier(999, "0x" + "2" * 40, "USDC", 6), ADDRESS, internal_id=int(state.pair_id.iloc[0]))
    records = []
    for day in days:
        row = {"day": day, "state": "Not evaluated", "admission": "Not evaluated"}
        if day <= AS_OF:
            selected = model._lookup_vault_state(day, pair)
            gate = model.check_deposit(day, pair)
            allowed = model.can_deposit(day, pair)
            assert allowed == gate.can_deposit, f"Deposit APIs disagree on {day}"
            before_cutoff = day < HYPERCORE_DEPOSIT_STATE_CUTOFF
            if before_cutoff:
                assert allowed, f"Historical assumed-open policy rejected {day}"
            row.update({
                "state": status(None if selected is None else selected.get("deposits_open")),
                "admission": ("Assumed open" if before_cutoff else "Allowed") if allowed else "Blocked",
                "block_reason": str(gate.reason_code),
            })
            past = state.loc[state.timestamp <= day]
            row["bucket"] = pd.NaT if past.empty else past.timestamp.iloc[-1]
            for field in ("permission_observed_at", "permission_provenance", "permission_observation_id", "capacity_observed_at", "evidence_available_at", "deposit_closed_reason", "max_deposit"):
                row[field] = None if selected is None else selected.get(field)
            clock = row["permission_observed_at"]
            row["permission_age_hours"] = None if pd.isna(clock) else (day - pd.Timestamp(clock)).total_seconds() / 3600
        records.append(row)
    return pd.DataFrame(records).set_index("day").add_prefix(prefix + "_")


def main() -> None:
    """Write 365 daily rows and verify admission plus documented state differences."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.recovery_root.resolve()
    paths = {
        "earlier": root / "r2-originals/2026-10-05/vault-prices-1h.parquet",
        "corrupted": root / "current-local-originals/vault-prices-1h.parquet",
        "repaired": root / "local-rehearsal/vault-prices-1h-price-clock.parquet",
        "cleaned": root / "doezoe-analysis/doezoe-fixed-cleaned.parquet",
        "permissions": root / "local-rehearsal/hypercore-vault-permissions.parquet",
        "reference": root / "doezoe-analysis/doezoe-deposit-status-2026-recovered.csv",
    }
    days = pd.date_range("2026-01-01", "2026-12-31", freq="1d")
    with duckdb.connect() as connection:
        raw = {
            name: connection.execute("SELECT * FROM read_parquet(?) WHERE chain=9999 AND address=? ORDER BY timestamp", [str(paths[name]), ADDRESS]).df()
            for name in ("earlier", "corrupted", "repaired")
        }
    prices = pd.read_parquet(paths["cleaned"]).reset_index()
    assert len(prices) == len(raw["repaired"])
    assert prices.address.eq(ADDRESS).all()
    history = read_vault_permission_history_parquet(paths["permissions"], pd.DataFrame({"chain_id": [9999], "address": [ADDRESS]}))
    reference = pd.read_csv(paths["reference"], parse_dates=["day"]).set_index("day").reindex(days)
    daily = pd.concat([coverage(frame, days, name) for name, frame in raw.items()], axis=1)
    # Check these are the same immutable sources as the independently generated reference.
    for name, old_name in (("earlier", "earlier"), ("corrupted", "corrupted"), ("repaired", "fixed")):
        assert daily[name + "_rows"].eq(reference[old_name + "_rows"]).all()
    daily["previous_usable_status"] = reference.corrupted_state
    daily["reference_state"] = reference.evidence_state.str.replace(" (deny-only)", "", regex=False)
    daily["reference_admission"] = reference.corrected_admission
    for prefix, sidecar in (("projection", None), ("sidecar", history)):
        daily = daily.join(framework(prices, days, sidecar, prefix))
        daily[prefix + "_matches_reference"] = daily[prefix + "_state"].eq(daily.reference_state) & daily[prefix + "_admission"].eq(daily.reference_admission)
        mismatches = daily.loc[(daily.index <= AS_OF) & ~daily[prefix + "_matches_reference"]]
        assert mismatches.index.tolist() == [pd.Timestamp("2026-04-03")], f"Unexpected {prefix} mismatches:\n{mismatches.to_string()}"
        assert mismatches[prefix + "_state"].eq("Unknown").all()
        assert mismatches.reference_state.eq("Open").all()
        assert daily[prefix + "_admission"].eq(daily.reference_admission).all()
    assert daily.projection_state.eq(daily.sidecar_state).all()
    assert daily.projection_admission.eq(daily.sidecar_admission).all()
    assert daily.loc["2026-09-13":"2026-09-21", "sidecar_state"].eq("Open").all()
    assert daily.loc["2026-09-22":"2026-10-06", "sidecar_state"].eq("Closed").all()
    assert daily.sidecar_capacity_observed_at.isna().all()
    genuine = history.provenance.isin(("observed", "observed_unknown", "restored")) & history.permission_observed_at.notna()
    assert not genuine.any()
    first_closed = raw["earlier"].loc[raw["earlier"].deposits_open.map(status).eq("Closed")].iloc[0]
    closure_evidence = history.loc[history.permission_observed_at.eq(first_closed.timestamp)]
    assert len(closure_evidence) == 1
    original_closure_write = pd.Timestamp(json.loads(closure_evidence.payload_json.iloc[0])["written_at"])
    # Reproduce all quoted September counts where the original partial archive is available.
    assert daily.loc["2026-09-13":"2026-09-27", "earlier_rows"].tolist() == [19, 19, 19, 17, 21, 22, 25, 22, 25, 23, 20, 20, 19, 17, 18]
    assert daily.loc["2026-09-13":"2026-10-04", "corrupted_rows"].tolist() == [19, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 8, 8, 8, 8, 9, 8]
    assert len(daily) == 365 and daily.index.is_unique
    daily.index.name = "day"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(args.output_dir / f"{STEM}.csv")
    repo = Path(__file__).resolve().parents[2]
    summary = {
        "vault": ADDRESS, "as_of": timestamp(AS_OF), "calendar_rows": len(daily),
        "evaluated_days": int((days <= AS_OF).sum()),
        "projection_state_mismatches": int((~daily.projection_matches_reference & (days <= AS_OF)).sum()),
        "sidecar_state_mismatches": int((~daily.sidecar_matches_reference & (days <= AS_OF)).sum()),
        "admission_mismatches": 0,
        "documented_state_difference": {
            "day": "2026-04-03", "reference": "Open", "fixed": "Unknown",
            "boundary": "2026-04-02 22:20:00.054", "admission": "Assumed open",
        },
        "genuine_permission_receipts": int(genuine.sum()), "permission_records": len(history),
        "permission_provenance_counts": history.provenance.value_counts().to_dict(),
        "first_retained_closed_price": timestamp(first_closed.timestamp),
        "first_retained_closed_write": timestamp(original_closure_write),
        "first_retained_closed_parquet_write": timestamp(first_closed.written_at),
        "first_fixed_closed_midnight": timestamp(daily.index[daily.sidecar_state.eq("Closed")].min()),
        "client_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo / "deps/trading-strategy", text=True).strip(),
        "executor_base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "executor_changes": subprocess.check_output(["git", "diff", "HEAD", "--stat", "--", "tradeexecutor/backtest/backtest_pricing.py", "tradeexecutor/strategy/trading_strategy_universe.py"], cwd=repo, text=True).strip(),
        "implementation_sha256": {
            name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
            for name in (
                "tradeexecutor/backtest/backtest_pricing.py",
                "tradeexecutor/strategy/trading_strategy_universe.py",
                "deps/trading-strategy/tradingstrategy/alternative_data/vault.py",
                "deps/trading-strategy/tradingstrategy/vault_permission.py",
            )
        },
        "sources": {},
    }
    for name, path in paths.items():
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        summary["sources"][name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}
        if name in raw:
            summary["sources"][name]["vault_rows"] = len(raw[name])
    (args.output_dir / f"{STEM}.json").write_text(json.dumps(summary, indent=2) + "\n")
    comparison = pd.DataFrame(index=daily.index)
    for name in ("earlier", "corrupted", "repaired"):
        comparison[name.capitalize() + " rows / flags"] = daily[name + "_rows"].astype(str) + " / " + daily[name + "_flags"]
    comparison["Corrupted write dates"] = daily.corrupted_write_dates
    comparison["Previous usable status"] = daily.previous_usable_status
    comparison["Fixed usable status"] = daily.sidecar_state
    comparison.index = comparison.index.strftime("%Y-%m-%d")
    calendar = comparison.copy()
    calendar["Admission"] = daily.sidecar_admission.to_numpy()
    calendar["Permission clock UTC"] = daily.sidecar_permission_observed_at.map(timestamp).to_numpy()
    calendar["Clock provenance"] = daily.sidecar_permission_provenance.fillna("—").to_numpy()
    calendar["Repaired first price UTC"] = daily.repaired_first_price.map(timestamp).to_numpy()
    calendar["Repaired last price UTC"] = daily.repaired_last_price.map(timestamp).to_numpy()
    calendar["Repaired first write UTC"] = daily.repaired_first_write.map(timestamp).to_numpy()
    calendar["Repaired last write UTC"] = daily.repaired_last_write.map(timestamp).to_numpy()
    report = f"""# DOEZOE deposit status after the client fix — 2026

The fixed client and executor resolve **Open on 13–21 September and Closed
from 22 September through 6 October**, at midnight UTC. Both the repaired price
projection and the separate permission-history file agree with the archived
producer reference on **deposit admission for all 279 evaluated days**. State
agrees on 278 days; the one documented difference on 3 April is described below.
Both conversion modes and both executor deposit APIs agree with each other.
The full calendar below has **365 rows**.

These checks establish consistency of the recovery implementations. They do
not independently corroborate venue history: this sidecar is derived from the
same recovered flags and contains **zero genuine permission receipts**.

Vault: `{ADDRESS}`. This checks the retained local recovery snapshots as of
6 October 2026 using client commit `{summary['client_commit']}` and the executor
implementation identified by the audit JSON. It does not inspect or change production. All clocks are UTC,
represented as naive timestamps. Decisions are at 00:00; dates after 6 October
are **Not evaluated** because the recovery snapshot ends on 5 October.

## September comparison and October tail

“Previous usable status” is the archived result of the previous framework on
the corrupted archive. “Fixed usable status” runs the updated framework on the
repaired data, with the separate permission-history file. Price-only projection
produces the same daily status and admission. Counts describe price rows and
stored flags; they are not counts of independent permission measurements.

{comparison.loc['2026-09-13':'2026-10-06'].to_markdown()}

The earlier source available here is the preserved 5 October R2 archive. Its
counts match the quoted partial archive through 27 September. It has **18 rows
on 28 September**, rather than the quoted partial file's five, and continues into
October. The original `/Users/moo/...` file is unavailable on this machine;
later earlier-archive counts above are from the retained R2 source.

## What the clocks establish

The first retained Closed flag is at **{timestamp(first_closed.timestamp)}**, with
original scanner write time **{timestamp(original_closure_write)}** and reason
“Vault deposits disabled by leader”. Daily availability rounds this evidence
upwards: the first closed midnight is **22 September**. This is the first retained
closed flag, not proof of the exact venue transition time. The scanner write
time is retained in the sidecar payload; raw price Parquet truncates it to
millisecond precision, **{timestamp(first_closed.written_at)}**. Daily price/write
columns below retain the Parquet precision without inventing finer clocks.

There are **zero independently clocked permission receipts** for DOEZOE in these
files. Recovered flags use the accepted `legacy_price_timestamp` approximation;
the price timestamp supplies an inferred permission clock. Flags may already
have been carried forwards. The original write timestamp remains separate and
does not postpone September evidence until October. Permission freshness is
checked against the original selected clock with a two-day tolerance. Capacity
has no authenticated clock and remains unknown; it is never inferred from price
flags. Unknown blocks new deposits from 11 April onwards; earlier admission is
the framework's explicit “Assumed open” policy.

The repaired data is essential: a client change cannot reconstruct the missing
Closed flags from the corrupted archive's historical Open rows. This report
validates the client fix together with the recovered data and provenance.

## April reference difference

At midnight on **3 April**, the archived producer reference carries the inferred
Open flag from **2 April 00:00** forwards. The cleaned price snapshot includes an
intervening `corrupted_unknown` row at **2 April 22:20:00.054**. The updated client
treats this as an uncertainty boundary, so the earlier inferred flag cannot
bridge the gap: usable state is **Unknown**. This is conservative handling of
the supplied provenance, not an independently observed closure. Admission is
still **Assumed open** under the framework's policy before 11 April. All states
from 11 April through 6 October match the independent reference; there are zero
admission differences across the full evaluated period.

## Full daily calendar

The [CSV]({STEM}.csv) retains the clocks, ages, evidence IDs, bucket timestamps,
closure reasons, both conversion results and reference comparisons. The
[audit JSON]({STEM}.json) records immutable input paths, SHA-256 hashes, row
counts and code revisions. The existing producer-cleaned snapshot is used as
client input; this run does not rerun the producer cleaner. Reproduce with
`scripts/hyperliquid/reproduce-doezoe-deposit-status.py` and the retained recovery
directory; the module docstring gives the command.

{calendar.to_markdown()}
"""
    (args.output_dir / f"{STEM}.md").write_text(report)
    print(json.dumps({key: summary[key] for key in ("calendar_rows", "evaluated_days", "projection_state_mismatches", "sidecar_state_mismatches", "admission_mismatches", "genuine_permission_receipts", "first_fixed_closed_midnight")}, indent=2))


if __name__ == "__main__":
    main()
