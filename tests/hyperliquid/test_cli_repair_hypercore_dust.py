"""Test CLI repair flow for Hypercore dust positions and duplicates."""

import datetime
from decimal import Decimal
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tradeexecutor.cli.main import app
from tradeexecutor.ethereum.vault.hypercore_vault import create_hypercore_vault_pair
from tradeexecutor.state.balance_update import (
    BalanceUpdate,
    BalanceUpdateCause,
    BalanceUpdatePositionType,
)
from tradeexecutor.state.state import State
from tradeexecutor.state.store import JSONFileStore
from tradeexecutor.state.trade import TradeFlag, TradeType
from tradeexecutor.state.identifier import AssetIdentifier


pytestmark = pytest.mark.timeout(60)


def _build_hypercore_duplicate_state(
    *,
    dust_quantity: Decimal | None,
    initial_reserve: Decimal = Decimal("1.00"),
    duplicate_reserve: Decimal = Decimal("25"),
    duplicate_flags: set[TradeFlag] | None = None,
    duplicate_balance_update_quantity: Decimal | None = None,
    duplicate_last_token_price: float | None = None,
    vault_address: str = "0x2222222222222222222222222222222222222222",
) -> tuple[State, int, int]:
    """Build a Hypercore state with one original position and one duplicate."""

    reserve_asset = AssetIdentifier(
        chain_id=999,
        address="0xb88339cb7199b77e23db6e890353e22632ba630f",
        token_symbol="USDC",
        decimals=6,
    )
    pair = create_hypercore_vault_pair(
        quote=reserve_asset,
        vault_address=vault_address,
    )

    state = State()
    state.portfolio.initialise_reserves(reserve_asset, reserve_token_price=1.0)
    state.portfolio.adjust_reserves(reserve_asset, Decimal("100"), "Initial reserve")

    dust_position, dust_trade, _created = state.create_trade(
        strategy_cycle_at=datetime.datetime(2026, 4, 13),
        pair=pair,
        quantity=None,
        reserve=initial_reserve,
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=reserve_asset,
        reserve_currency_price=1.0,
        notes="Create initial Hypercore position",
    )
    dust_trade.mark_success(
        executed_at=datetime.datetime(2026, 4, 13, 0, 1),
        executed_price=1.0,
        executed_quantity=initial_reserve,
        executed_reserve=initial_reserve,
        lp_fees=0,
        native_token_price=0,
        force=True,
    )

    if dust_quantity is not None:
        assert initial_reserve >= dust_quantity
        dust_position.balance_updates[1] = BalanceUpdate(
            balance_update_id=1,
            cause=BalanceUpdateCause.vault_flow,
            position_type=BalanceUpdatePositionType.open_position,
            asset=pair.base,
            block_mined_at=datetime.datetime(2026, 4, 13, 0, 2),
            strategy_cycle_included_at=datetime.datetime(2026, 4, 13),
            chain_id=pair.base.chain_id,
            quantity=-(initial_reserve - dust_quantity),
            old_balance=initial_reserve,
            usd_value=float(-(initial_reserve - dust_quantity)),
            position_id=dust_position.position_id,
            notes="Simulate Hypercore withdrawal dust",
            block_number=1,
        )

    duplicate_position, duplicate_trade, _duplicate_created = state.create_trade(
        strategy_cycle_at=datetime.datetime(2026, 4, 14),
        pair=pair,
        quantity=None,
        reserve=duplicate_reserve,
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=reserve_asset,
        reserve_currency_price=1.0,
        notes="Force duplicate Hypercore position",
        flags={TradeFlag.ignore_open},
    )
    duplicate_trade.mark_success(
        executed_at=datetime.datetime(2026, 4, 14, 0, 1),
        executed_price=1.0,
        executed_quantity=duplicate_reserve,
        executed_reserve=duplicate_reserve,
        lp_fees=0,
        native_token_price=0,
        force=True,
    )

    if duplicate_flags is not None:
        duplicate_trade.flags = duplicate_flags

    if duplicate_balance_update_quantity is not None:
        duplicate_position.balance_updates[2] = BalanceUpdate(
            balance_update_id=2,
            cause=BalanceUpdateCause.vault_flow,
            position_type=BalanceUpdatePositionType.open_position,
            asset=pair.base,
            block_mined_at=datetime.datetime(2026, 4, 14, 0, 2),
            strategy_cycle_included_at=datetime.datetime(2026, 4, 14),
            chain_id=pair.base.chain_id,
            quantity=duplicate_balance_update_quantity,
            old_balance=duplicate_reserve,
            usd_value=float(duplicate_balance_update_quantity),
            position_id=duplicate_position.position_id,
            notes="Simulate duplicate-side balance update",
            block_number=2,
        )

    if duplicate_last_token_price is not None:
        duplicate_position.last_token_price = duplicate_last_token_price

    return state, dust_position.position_id, duplicate_position.position_id


def _append_hypercore_duplicate_group_to_state(
    state: State,
    *,
    vault_address: str,
    initial_reserve: Decimal,
    duplicate_reserve: Decimal,
    dust_quantity: Decimal | None = None,
    duplicate_flags: set[TradeFlag] | None = None,
    duplicate_balance_update_quantity: Decimal | None = None,
    duplicate_last_token_price: float | None = None,
) -> tuple[int, int]:
    """Append one Hypercore duplicate-position group to an existing state."""

    reserve_asset = state.portfolio.get_default_reserve_position().asset
    pair = create_hypercore_vault_pair(
        quote=reserve_asset,
        vault_address=vault_address,
    )

    original_position, original_trade, _created = state.create_trade(
        strategy_cycle_at=datetime.datetime(2026, 4, 15),
        pair=pair,
        quantity=None,
        reserve=initial_reserve,
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=reserve_asset,
        reserve_currency_price=1.0,
        notes="Create original Hypercore position",
        flags={TradeFlag.ignore_open},
    )
    original_trade.mark_success(
        executed_at=datetime.datetime(2026, 4, 15, 0, 1),
        executed_price=1.0,
        executed_quantity=initial_reserve,
        executed_reserve=initial_reserve,
        lp_fees=0,
        native_token_price=0,
        force=True,
    )

    if dust_quantity is not None:
        original_position.balance_updates[state.portfolio.next_balance_update_id] = BalanceUpdate(
            balance_update_id=state.portfolio.next_balance_update_id,
            cause=BalanceUpdateCause.vault_flow,
            position_type=BalanceUpdatePositionType.open_position,
            asset=pair.base,
            block_mined_at=datetime.datetime(2026, 4, 15, 0, 2),
            strategy_cycle_included_at=datetime.datetime(2026, 4, 15),
            chain_id=pair.base.chain_id,
            quantity=-(initial_reserve - dust_quantity),
            old_balance=initial_reserve,
            usd_value=float(-(initial_reserve - dust_quantity)),
            position_id=original_position.position_id,
            notes="Simulate Hypercore withdrawal dust",
            block_number=10,
        )
        state.portfolio.next_balance_update_id += 1

    duplicate_position, duplicate_trade, _duplicate_created = state.create_trade(
        strategy_cycle_at=datetime.datetime(2026, 4, 16),
        pair=pair,
        quantity=None,
        reserve=duplicate_reserve,
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=reserve_asset,
        reserve_currency_price=1.0,
        notes="Create duplicate Hypercore position",
        flags={TradeFlag.ignore_open},
    )
    duplicate_trade.mark_success(
        executed_at=datetime.datetime(2026, 4, 16, 0, 1),
        executed_price=1.0,
        executed_quantity=duplicate_reserve,
        executed_reserve=duplicate_reserve,
        lp_fees=0,
        native_token_price=0,
        force=True,
    )

    if duplicate_flags is not None:
        duplicate_trade.flags = duplicate_flags

    if duplicate_balance_update_quantity is not None:
        duplicate_position.balance_updates[state.portfolio.next_balance_update_id] = BalanceUpdate(
            balance_update_id=state.portfolio.next_balance_update_id,
            cause=BalanceUpdateCause.vault_flow,
            position_type=BalanceUpdatePositionType.open_position,
            asset=pair.base,
            block_mined_at=datetime.datetime(2026, 4, 16, 0, 2),
            strategy_cycle_included_at=datetime.datetime(2026, 4, 16),
            chain_id=pair.base.chain_id,
            quantity=duplicate_balance_update_quantity,
            old_balance=duplicate_reserve,
            usd_value=float(duplicate_balance_update_quantity),
            position_id=duplicate_position.position_id,
            notes="Simulate duplicate-side balance update",
            block_number=11,
        )
        state.portfolio.next_balance_update_id += 1

    if duplicate_last_token_price is not None:
        duplicate_position.last_token_price = duplicate_last_token_price

    return original_position.position_id, duplicate_position.position_id


def test_repair_hypercore_dust_cli_closes_duplicate_residual_position(
    tmp_path: Path,
) -> None:
    """Test the CLI closes stale Hypercore dust duplicates and leaves the live position open.

    1. Create a state file with one dusty Hypercore position and one duplicate live position for the same vault.
    2. Run the CLI repair command with auto approval.
    3. Verify the dusty duplicate is closed, the live position remains open, and the command exits cleanly.
    """

    # 1. Create a state file with one dusty Hypercore position and one duplicate live position for the same vault.
    state, dust_position_id, live_position_id = _build_hypercore_duplicate_state(
        dust_quantity=Decimal("0.10"),
    )
    state_file = tmp_path / "hypercore-duplicate-state.json"
    JSONFileStore(state_file).sync(state)

    # 2. Run the CLI repair command with auto approval.
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "repair-hypercore-dust",
            "--state-file",
            str(state_file),
            "--auto-approve",
            "--unit-testing",
            "--log-level",
            "disabled",
        ],
    )

    # 3. Verify the dusty duplicate is closed, the live position remains open, and the command exits cleanly.
    assert result.exit_code == 0, result.stdout

    repaired_state = State.read_json_file(state_file)
    assert dust_position_id in repaired_state.portfolio.closed_positions
    assert dust_position_id not in repaired_state.portfolio.open_positions
    assert live_position_id in repaired_state.portfolio.open_positions
    assert live_position_id not in repaired_state.portfolio.closed_positions


def test_repair_hypercore_dust_cli_warns_for_non_dust_duplicates(
    tmp_path: Path,
) -> None:
    """Test the CLI warns and fails when Hypercore duplicates are not closeable dust.

    1. Create a state file with two non-dust Hypercore positions for the same vault.
    2. Run the CLI repair command with auto approval.
    3. Verify the command fails with a duplicate warning and leaves both positions open.
    """

    # 1. Create a state file with two non-dust Hypercore positions for the same vault.
    state, first_position_id, second_position_id = _build_hypercore_duplicate_state(
        dust_quantity=None,
    )
    state_file = tmp_path / "hypercore-non-dust-duplicate-state.json"
    JSONFileStore(state_file).sync(state)

    # 2. Run the CLI repair command with auto approval.
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "repair-hypercore-dust",
            "--state-file",
            str(state_file),
            "--auto-approve",
            "--unit-testing",
            "--log-level",
            "disabled",
        ],
    )

    # 3. Verify the command fails with a duplicate warning and leaves both positions open.
    assert result.exit_code != 0
    assert "Hypercore duplicate positions still remain after dust cleanup" in str(result.exception)

    repaired_state = State.read_json_file(state_file)
    assert first_position_id in repaired_state.portfolio.open_positions
    assert second_position_id in repaired_state.portfolio.open_positions


def test_repair_hypercore_dust_cli_suppresses_safe_later_clone_with_confirmation(
    tmp_path: Path,
) -> None:
    """Test the CLI suppresses a safe later Hypercore clone after explicit confirmation.

    1. Create a state file with two exact-match non-dust Hypercore positions for the same vault.
    2. Run the CLI repair command with ``--merge-dustless-duplicates`` and explicit `y` confirmations.
    3. Verify the older position stays open and the later clone moves to suppressed duplicates.
    """

    # 1. Create a state file with two exact-match non-dust Hypercore positions for the same vault.
    state, survivor_position_id, clone_position_id = _build_hypercore_duplicate_state(
        dust_quantity=None,
        initial_reserve=Decimal("25"),
        duplicate_reserve=Decimal("25"),
    )
    state_file = tmp_path / "hypercore-safe-clone-state.json"
    JSONFileStore(state_file).sync(state)

    # 2. Run the CLI repair command with ``--merge-dustless-duplicates`` and explicit `y` confirmations.
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "repair-hypercore-dust",
            "--state-file",
            str(state_file),
            "--merge-dustless-duplicates",
            "--log-level",
            "disabled",
        ],
        input="y\ny\n",
    )

    # 3. Verify the older position stays open and the later clone moves to suppressed duplicates.
    assert result.exit_code == 0, result.stdout

    repaired_state = State.read_json_file(state_file)
    assert survivor_position_id in repaired_state.portfolio.open_positions
    assert clone_position_id not in repaired_state.portfolio.open_positions
    assert clone_position_id in repaired_state.portfolio.suppressed_duplicate_positions


def test_repair_hypercore_dust_cli_safe_clone_still_fails_without_merge_flag(
    tmp_path: Path,
) -> None:
    """Test the CLI still fails on a safe later clone when merge mode is not enabled.

    1. Create a state file with one older Hypercore position and one exact-match later clone.
    2. Run the CLI repair command without ``--merge-dustless-duplicates``.
    3. Verify both positions remain open because the command refuses to suppress duplicates by default.
    """

    # 1. Create a state file with one older Hypercore position and one exact-match later clone.
    state, survivor_position_id, clone_position_id = _build_hypercore_duplicate_state(
        dust_quantity=None,
        initial_reserve=Decimal("25"),
        duplicate_reserve=Decimal("25"),
    )
    state_file = tmp_path / "hypercore-safe-clone-no-merge-state.json"
    JSONFileStore(state_file).sync(state)

    # 2. Run the CLI repair command without ``--merge-dustless-duplicates``.
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "repair-hypercore-dust",
            "--state-file",
            str(state_file),
            "--auto-approve",
            "--unit-testing",
            "--log-level",
            "disabled",
        ],
    )

    # 3. Verify both positions remain open because the command refuses to suppress duplicates by default.
    assert result.exit_code != 0

    repaired_state = State.read_json_file(state_file)
    assert survivor_position_id in repaired_state.portfolio.open_positions
    assert clone_position_id in repaired_state.portfolio.open_positions


def test_repair_hypercore_dust_cli_combines_dust_cleanup_and_clone_suppression(
    tmp_path: Path,
) -> None:
    """Test the CLI can finish one dust cleanup group and one safe clone group in one transactional save.

    1. Create a state file with one dust duplicate group and one safe later-clone duplicate group.
    2. Run the CLI repair command with merge mode and explicit confirmations.
    3. Verify the dust residual is closed, the clone is suppressed, and the survivors remain open.
    """

    # 1. Create a state file with one dust duplicate group and one safe later-clone duplicate group.
    state, dust_position_id, dust_live_position_id = _build_hypercore_duplicate_state(
        dust_quantity=Decimal("0.10"),
    )
    clone_survivor_position_id, clone_position_id = _append_hypercore_duplicate_group_to_state(
        state,
        vault_address="0x3333333333333333333333333333333333333333",
        initial_reserve=Decimal("25"),
        duplicate_reserve=Decimal("25"),
    )
    state_file = tmp_path / "hypercore-dust-and-clone-state.json"
    JSONFileStore(state_file).sync(state)

    # 2. Run the CLI repair command with merge mode and explicit confirmations.
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "repair-hypercore-dust",
            "--state-file",
            str(state_file),
            "--merge-dustless-duplicates",
            "--log-level",
            "disabled",
        ],
        input="y\ny\n",
    )

    # 3. Verify the dust residual is closed, the clone is suppressed, and the survivors remain open.
    assert result.exit_code == 0, result.stdout

    repaired_state = State.read_json_file(state_file)
    assert dust_position_id in repaired_state.portfolio.closed_positions
    assert dust_live_position_id in repaired_state.portfolio.open_positions
    assert clone_survivor_position_id in repaired_state.portfolio.open_positions
    assert clone_position_id not in repaired_state.portfolio.open_positions
    assert clone_position_id in repaired_state.portfolio.suppressed_duplicate_positions


def test_repair_hypercore_dust_cli_does_not_save_partial_results_when_unsafe_group_remains(
    tmp_path: Path,
) -> None:
    """Test merge mode keeps the state file untouched when any remaining duplicate group is unsafe.

    1. Create a state file with one safe clone group and one unsafe duplicate group.
    2. Run the CLI repair command with merge mode.
    3. Verify the command fails before saving and the state file still contains only open duplicate positions.
    """

    # 1. Create a state file with one safe clone group and one unsafe duplicate group.
    state, safe_survivor_position_id, safe_clone_position_id = _build_hypercore_duplicate_state(
        dust_quantity=None,
        initial_reserve=Decimal("25"),
        duplicate_reserve=Decimal("25"),
    )
    unsafe_first_position_id, unsafe_second_position_id = _append_hypercore_duplicate_group_to_state(
        state,
        vault_address="0x4444444444444444444444444444444444444444",
        initial_reserve=Decimal("25"),
        duplicate_reserve=Decimal("25"),
        duplicate_balance_update_quantity=Decimal("0.01"),
    )
    state_file = tmp_path / "hypercore-unsafe-mixed-state.json"
    JSONFileStore(state_file).sync(state)

    # 2. Run the CLI repair command with merge mode.
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "repair-hypercore-dust",
            "--state-file",
            str(state_file),
            "--merge-dustless-duplicates",
            "--unit-testing",
            "--log-level",
            "disabled",
        ],
    )

    # 3. Verify the command fails before saving and the state file still contains only open duplicate positions.
    assert result.exit_code != 0
    assert "The state file was not updated" in str(result.exception)

    repaired_state = State.read_json_file(state_file)
    assert safe_survivor_position_id in repaired_state.portfolio.open_positions
    assert safe_clone_position_id in repaired_state.portfolio.open_positions
    assert unsafe_first_position_id in repaired_state.portfolio.open_positions
    assert unsafe_second_position_id in repaired_state.portfolio.open_positions
    assert len(repaired_state.portfolio.suppressed_duplicate_positions) == 0


def test_repair_hypercore_dust_cli_rejects_auto_approve_for_merge_mode(
    tmp_path: Path,
) -> None:
    """Test merge mode refuses ``--auto-approve`` because dangerous suppressions require explicit `y/n`.

    1. Create a state file with a safe later-clone duplicate group.
    2. Run the CLI repair command with both merge mode and ``--auto-approve``.
    3. Verify the command fails and does not suppress the duplicate clone.
    """

    # 1. Create a state file with a safe later-clone duplicate group.
    state, survivor_position_id, clone_position_id = _build_hypercore_duplicate_state(
        dust_quantity=None,
        initial_reserve=Decimal("25"),
        duplicate_reserve=Decimal("25"),
    )
    state_file = tmp_path / "hypercore-auto-approve-merge-state.json"
    JSONFileStore(state_file).sync(state)

    # 2. Run the CLI repair command with both merge mode and ``--auto-approve``.
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "repair-hypercore-dust",
            "--state-file",
            str(state_file),
            "--merge-dustless-duplicates",
            "--auto-approve",
            "--log-level",
            "disabled",
        ],
        input="y\n",
    )

    # 3. Verify the command fails and does not suppress the duplicate clone.
    assert result.exit_code != 0
    assert "requires an explicit y/n confirmation" in str(result.exception)

    repaired_state = State.read_json_file(state_file)
    assert survivor_position_id in repaired_state.portfolio.open_positions
    assert clone_position_id in repaired_state.portfolio.open_positions


def test_repair_hypercore_dust_cli_infers_default_state_file_from_id() -> None:
    """Test the CLI uses the standard inferred ``state/{id}.json`` path when state file is omitted.

    1. Create a state file under the default ``state/`` directory for a chosen executor id.
    2. Run the CLI repair command with only ``--id`` and no explicit state path.
    3. Verify the inferred state file is repaired successfully.
    """

    # 1. Create a state file under the default ``state/`` directory for a chosen executor id.
    state, dust_position_id, live_position_id = _build_hypercore_duplicate_state(
        dust_quantity=Decimal("0.10"),
    )
    runner = CliRunner()

    with runner.isolated_filesystem():
        state_dir = Path("state")
        state_dir.mkdir()
        state_file = state_dir / "hyper-ai.json"
        JSONFileStore(state_file).sync(state)

        # 2. Run the CLI repair command with only ``--id`` and no explicit state path.
        result = runner.invoke(
            app,
            [
                "repair-hypercore-dust",
                "--id",
                "hyper-ai",
                "--auto-approve",
                "--unit-testing",
                "--log-level",
                "disabled",
            ],
        )

        # 3. Verify the inferred state file is repaired successfully.
        assert result.exit_code == 0, result.stdout

        repaired_state = State.read_json_file(state_file)
        assert dust_position_id in repaired_state.portfolio.closed_positions
        assert dust_position_id not in repaired_state.portfolio.open_positions
        assert live_position_id in repaired_state.portfolio.open_positions


def test_repair_hypercore_dust_cli_creates_backup_before_saving(
    tmp_path: Path,
) -> None:
    """Test the CLI follows the normal mutating-command backup pattern before saving state.

    1. Create a state file with a closeable Hypercore dust duplicate.
    2. Run the CLI repair command without unit-testing mode so backup creation stays enabled.
    3. Verify the repair succeeds and a dedicated backup file is created alongside the state file.
    """

    # 1. Create a state file with a closeable Hypercore dust duplicate.
    state, dust_position_id, live_position_id = _build_hypercore_duplicate_state(
        dust_quantity=Decimal("0.10"),
    )
    state_file = tmp_path / "hypercore-backup-state.json"
    JSONFileStore(state_file).sync(state)

    # 2. Run the CLI repair command without unit-testing mode so backup creation stays enabled.
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "repair-hypercore-dust",
            "--state-file",
            str(state_file),
            "--auto-approve",
            "--log-level",
            "disabled",
        ],
    )

    # 3. Verify the repair succeeds and a dedicated backup file is created alongside the state file.
    assert result.exit_code == 0, result.stdout

    repaired_state = State.read_json_file(state_file)
    assert dust_position_id in repaired_state.portfolio.closed_positions
    assert live_position_id in repaired_state.portfolio.open_positions

    backup_file = state_file.with_suffix(".repair-hypercore-dust-backup-1.json")
    assert backup_file.exists()
