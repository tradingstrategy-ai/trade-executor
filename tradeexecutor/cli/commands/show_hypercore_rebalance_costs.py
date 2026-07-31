"""Display HyperCore vault rebalance costs from an executor state file."""

from pathlib import Path

import pandas as pd
from tabulate import tabulate
from typer import Option

from tradeexecutor.analysis.hypercore_rebalance_cost import (
    build_hypercore_rebalance_cost_report,
    fetch_hype_hourly_price_lookup,
)
from tradeexecutor.cli.bootstrap import prepare_executor_id
from tradeexecutor.cli.commands import shared_options
from tradeexecutor.cli.commands.app import app
from tradeexecutor.state.state import State


def _display_table(table: pd.DataFrame) -> None:
    """Print a dataframe without turning unavailable values into zeroes."""
    if len(table) == 0:
        print("None")
    else:
        display_table = table.astype(object).where(pd.notna(table), None)
        print(
            tabulate(
                display_table,
                headers="keys",
                tablefmt="rounded_outline",
                showindex=False,
                missingval="N/A",
            )
        )


@app.command()
def show_hypercore_rebalance_costs(
    id: str = shared_options.id,
    state_file: Path | None = shared_options.state_file,
    strategy_file: Path | None = shared_options.optional_strategy_file,
    details: bool = Option(False, help="Also show one row per eligible trade."),
    no_historical_prices: bool = Option(
        False, help="Do not fetch HYPE hourly candles for legacy gas receipts."
    ),
):
    """Tabulate normal HyperCore rebalance costs and historic-data coverage.

    The command is read-only. Historic cost values are lower bounds unless the
    trade says its HyperCore cost data was captured during live settlement.
    Repaired, repair, unsuccessful, test, and transactionless trades are excluded.
    """
    if not state_file:
        id = prepare_executor_id(id, strategy_file)
        assert id, "Executor id must be given if state file is not specified"
        state_file = Path(f"state/{id}.json")

    assert state_file.exists(), f"State file does not exist: {state_file}"
    state = State.read_json_file(state_file)

    trades = list(state.portfolio.get_all_trades())
    price_lookup = None
    if not no_historical_prices:
        try:
            price_lookup = fetch_hype_hourly_price_lookup(trades)
        except Exception as e:
            print(
                f"Could not fetch HYPE hourly prices; legacy gas remains unavailable: {e}"
            )

    report = build_hypercore_rebalance_cost_report(state, price_lookup)
    print(f"HyperCore rebalance costs for {state.name or state_file}")
    print(
        "Known costs are lower bounds for legacy trades; N/A components were not recorded."
    )
    print()
    print("Rebalances")
    _display_table(report.rebalances)
    print()
    print("Strategy summary")
    _display_table(report.summary)
    print()
    print("Ignored HyperCore history")
    _display_table(report.ignored)
    if details:
        print()
        print("Eligible trade details")
        _display_table(report.trades)
