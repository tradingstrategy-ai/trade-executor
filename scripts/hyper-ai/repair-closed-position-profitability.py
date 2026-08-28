"""Repair historical Hyper AI closed-position profitability.

The command previews changes by default.  Pass ``--write`` to create a backup
and atomically replace the local state file.

Usage::

    source .local-test.env && poetry run python scripts/hyper-ai/repair-closed-position-profitability.py state/hyper-ai.json
    source .local-test.env && poetry run python scripts/hyper-ai/repair-closed-position-profitability.py state/hyper-ai.json --write
"""

import argparse
from pathlib import Path

from tabulate import tabulate

from tradeexecutor.cli.bootstrap import backup_state
from tradeexecutor.state.repair import repair_hypercore_closed_position_profitability
from tradeexecutor.state.state import State


def format_pct(value: float) -> str:
    """Format a profitability value."""
    return f"{value:+.4%}"


def format_usd(value: float) -> str:
    """Format a dollar value."""
    return f"{value:+,.2f}"


def main() -> None:
    """Run the local state profitability repair."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "state_file", type=Path, help="Local trade-executor state JSON file"
    )
    parser.add_argument(
        "--position-id",
        type=int,
        action="append",
        dest="position_ids",
        help="Repair only this closed position id; may be supplied more than once",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Back up and replace the state file; without this flag only preview changes",
    )
    args = parser.parse_args()

    if not args.state_file.is_file():
        parser.error(f"State file does not exist: {args.state_file}")

    selected_ids = set(args.position_ids) if args.position_ids else None
    state = State.read_json_file(args.state_file)
    try:
        repairs = repair_hypercore_closed_position_profitability(
            state,
            position_ids=selected_ids,
        )
    except ValueError as e:
        parser.error(str(e))

    rows = [
        (
            repair.position_id,
            repair.symbol,
            format_pct(repair.old_profitability),
            format_pct(repair.new_profitability),
            format_usd(repair.old_profit_usd),
            format_usd(repair.new_profit_usd),
        )
        for repair in repairs
    ]
    if rows:
        print(
            tabulate(
                rows,
                headers=(
                    "position",
                    "pair",
                    "old return",
                    "new return",
                    "old PnL",
                    "new PnL",
                ),
            )
        )
    else:
        print("No closed Hypercore profitability records need repair.")

    if args.write and repairs:
        store, state = backup_state(
            args.state_file,
            backup_suffix="hypercore-profitability-backup",
        )
        written_repairs = repair_hypercore_closed_position_profitability(
            state,
            position_ids=selected_ids,
        )
        assert written_repairs == repairs, (
            "State file changed between preview and write"
        )
        store.sync(state)
        print(f"Repaired {len(repairs)} position(s) in {args.state_file}.")
    elif args.write:
        print("State file was not changed.")
    else:
        print(
            f"Preview only: {len(repairs)} position(s) would be repaired; pass --write to apply."
        )


if __name__ == "__main__":
    main()
