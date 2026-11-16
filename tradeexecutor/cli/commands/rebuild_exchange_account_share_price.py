"""CLI command to rebuild an exchange-account position's internal share price.

Fixes positions where a capital withdrawal was booked as a valuation loss, as
``lagoon-lighter-test-trade`` did before it booked its transfers. Run it in
the service container with the executor stopped; with no options it finds
the test-trade journals next to the state file:

.. code-block:: shell

    docker compose run --rm lighter-ai rebuild-exchange-account-share-price --dry-run
    docker compose run --rm lighter-ai rebuild-exchange-account-share-price

"""
import sys
import warnings
from pathlib import Path
from typing import Optional

from typer import Option

from . import shared_options
from .app import app
from tradeexecutor.cli.bootstrap import backup_state, create_state_store, prepare_executor_id
from tradeexecutor.strategy.position_internal_share_price_repair import (
    SharePriceRepair,
    SharePriceRepairError,
    apply_share_price_repair,
    detect_test_trade_withdrawals,
    find_lighter_test_trade_journals,
    plan_share_price_repair,
)


def _print_repair(repair: SharePriceRepair) -> None:
    stored = repair.stored_state
    rebuilt = repair.rebuilt_state
    print(f"\nPosition #{repair.position_id}")
    for bu_id, reason in sorted(repair.withdrawals.items()):
        print(f"  Withdrawal: balance update #{bu_id} ({reason})")
    for suspect in repair.suspects:
        print(
            f"  Suspect, not applied: balance update #{suspect.balance_update_id} at {suspect.at} of {suspect.quantity} "
            f"moved the share price {suspect.share_price_before:.6f} -> {suspect.share_price_after:.6f}"
        )
    print(f"  {'':<22} {'stored':>16} {'rebuilt':>16}")
    print(f"  {'share price':<22} {stored.current_share_price:>16.6f} {rebuilt.current_share_price:>16.6f}")
    print(f"  {'profit %':<22} {stored.current_share_price / stored.initial_share_price - 1:>16.2%} {rebuilt.current_share_price / rebuilt.initial_share_price - 1:>16.2%}")
    print(f"  {'total supply':<22} {stored.total_supply:>16,.2f} {rebuilt.total_supply:>16,.2f}")
    print(f"  {'total invested':<22} {stored.total_invested:>16,.2f} {rebuilt.total_invested:>16,.2f}")
    withdrawn = "unknown" if stored.total_withdrawn is None else f"{stored.total_withdrawn:,.6f}"
    print(f"  {'total withdrawn':<22} {withdrawn:>16} {rebuilt.total_withdrawn:>16,.6f}")
    if repair.samples:
        last = repair.samples[-1]
        print(f"  Statistics samples changed: {repair.samples_changed} of {len(repair.samples)}; latest profit {last.profitability:.2%}, {last.profit_usd:,.2f} USD")


@app.command()
def rebuild_exchange_account_share_price(
    id: str = shared_options.id,
    strategy_file: Optional[Path] = shared_options.optional_strategy_file,
    state_file: Optional[Path] = shared_options.state_file,
    unit_testing: bool = shared_options.unit_testing,
    position_id: Optional[list[int]] = Option(
        None,
        "--position-id",
        help="Rebuild only these positions. Default: every open exchange-account position.",
    ),
    lighter_test_trade_journal: Optional[list[Path]] = Option(
        None,
        "--lighter-test-trade-journal",
        help="Test-trade journals to read. Default: <executor-id>.lighter-test-trade*.json next to the state file.",
    ),
    capital_withdrawal_balance_update_id: Optional[list[int]] = Option(
        None,
        "--capital-withdrawal-balance-update-id",
        help="Also apply this valuation balance update as a capital withdrawal.",
    ),
    ignore_balance_update_id: Optional[list[int]] = Option(
        None,
        "--ignore-balance-update-id",
        help="Do not apply this detected withdrawal.",
    ),
    dry_run: bool = Option(False, "--dry-run", help="Print the repair and write nothing."),
    auto_approve: bool = Option(False, envvar="AUTO_APPROVE", help="Write without asking for confirmation."),
):
    """Rebuild the internal share price of exchange-account positions.

    Applies capital withdrawals that were booked as valuation losses, found
    from Lighter test-trade journals or named with
    --capital-withdrawal-balance-update-id, then rewrites each position's
    share-price state and statistics samples. Positions with nothing to
    change are left alone. Quantities, trades and reserves are not modified.

    Refuses to write unless a replay of each position's history reproduces
    its stored share-price state and every stored statistics sample.
    """
    id = prepare_executor_id(id, strategy_file)
    if not state_file:
        state_file = Path(f"state/{id}.json")
    print(f"State file: {state_file}")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="dataclasses_json")
        state = create_state_store(Path(state_file)).load()

    try:
        journals = lighter_test_trade_journal if lighter_test_trade_journal else find_lighter_test_trade_journals(Path(state_file), id)
        print(f"Test-trade journals: {', '.join(str(j) for j in journals) or 'none'}")
        detected, report = detect_test_trade_withdrawals(state, journals)
        for line in report:
            print(f"  {line}")

        for bu_id in capital_withdrawal_balance_update_id or []:
            owners = [p for p in state.portfolio.open_positions.values() if bu_id in p.balance_updates]
            if len(owners) != 1 or not owners[0].is_exchange_account():
                raise SharePriceRepairError(f"Balance update #{bu_id} is not on an open exchange-account position")
            detected.setdefault(owners[0].position_id, {})[bu_id] = "--capital-withdrawal-balance-update-id"
        for withdrawals in detected.values():
            for bu_id in ignore_balance_update_id or []:
                withdrawals.pop(bu_id, None)

        if position_id:
            positions = [state.portfolio.get_position_by_id(pid) for pid in position_id]
        else:
            positions = [p for p in state.portfolio.open_positions.values() if p.is_exchange_account()]

        repairs = []
        for position in positions:
            withdrawals = detected.get(position.position_id, {})
            try:
                repair = plan_share_price_repair(state, position, withdrawals)
            except SharePriceRepairError as e:
                if withdrawals or position_id:
                    raise
                # A scanned position with nothing to apply must not block the others
                print(f"\nPosition #{position.position_id} skipped: {e}")
                continue
            _print_repair(repair)
            if not repair.is_noop():
                repairs.append(repair)
    except SharePriceRepairError as e:
        print(f"\nError: {e}\nNothing was written.")
        raise SystemExit(1)

    if not repairs:
        print("\nNothing to repair.")
        return

    if dry_run:
        print("\nDry run: nothing was written.")
        return

    if not auto_approve:
        if not sys.stdin.isatty():
            print("\nNo terminal to confirm on: pass --auto-approve to write. Nothing was written.")
            raise SystemExit(1)
        answer = input(f"\nRewrite {len(repairs)} position(s)? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            raise SystemExit(0)

    store, state = backup_state(str(state_file), backup_suffix="share-price-rebuild-backup", unit_testing=unit_testing)
    for repair in repairs:
        apply_share_price_repair(state, repair)
    store.sync(state)
    print(f"\nRebuilt {len(repairs)} position(s).")
