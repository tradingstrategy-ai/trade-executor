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
        vault_address="0x2222222222222222222222222222222222222222",
    )

    state = State()
    state.portfolio.initialise_reserves(reserve_asset, reserve_token_price=1.0)
    state.portfolio.adjust_reserves(reserve_asset, Decimal("100"), "Initial reserve")

    dust_position, dust_trade, _created = state.create_trade(
        strategy_cycle_at=datetime.datetime(2026, 4, 13),
        pair=pair,
        quantity=None,
        reserve=Decimal("1.00"),
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=reserve_asset,
        reserve_currency_price=1.0,
        notes="Create initial Hypercore position",
    )
    dust_trade.mark_success(
        executed_at=datetime.datetime(2026, 4, 13, 0, 1),
        executed_price=1.0,
        executed_quantity=Decimal("1.00"),
        executed_reserve=Decimal("1.00"),
        lp_fees=0,
        native_token_price=0,
        force=True,
    )

    if dust_quantity is not None:
        dust_position.balance_updates[1] = BalanceUpdate(
            balance_update_id=1,
            cause=BalanceUpdateCause.vault_flow,
            position_type=BalanceUpdatePositionType.open_position,
            asset=pair.base,
            block_mined_at=datetime.datetime(2026, 4, 13, 0, 2),
            strategy_cycle_included_at=datetime.datetime(2026, 4, 13),
            chain_id=pair.base.chain_id,
            quantity=-(Decimal("1.00") - dust_quantity),
            old_balance=Decimal("1.00"),
            usd_value=float(-(Decimal("1.00") - dust_quantity)),
            position_id=dust_position.position_id,
            notes="Simulate Hypercore withdrawal dust",
            block_number=1,
        )

    duplicate_position, duplicate_trade, _duplicate_created = state.create_trade(
        strategy_cycle_at=datetime.datetime(2026, 4, 14),
        pair=pair,
        quantity=None,
        reserve=Decimal("25"),
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
        executed_quantity=Decimal("25"),
        executed_reserve=Decimal("25"),
        lp_fees=0,
        native_token_price=0,
        force=True,
    )

    return state, dust_position.position_id, duplicate_position.position_id


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
