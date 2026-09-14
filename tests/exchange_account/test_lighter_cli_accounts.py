"""Typer black-box account checks for a Lighter exchange position.

Ethereum reserve balances come from a fixed-block shared Anvil fork. Only the
current Lighter equity response is mocked because it is sequencer state that
Anvil cannot reproduce.
"""

import os
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import pytest
from eth_defi.compat import native_datetime_utc_now
from eth_defi.testing.anvil_fork_pool import AnvilForkPool
from eth_defi.testing.fork_blocks import ETHEREUM_MIDNIGHT_BLOCK
from eth_defi.token import USDC_NATIVE_TOKEN
from pytest_mock import MockerFixture
from typer.main import get_command
from web3 import Web3

from tradeexecutor.cli.main import app
from tradeexecutor.exchange_account.lighter import create_lighter_exchange_account_pair
from tradeexecutor.exchange_account.state import open_exchange_account_position
from tradeexecutor.state.balance_update import BalanceUpdateCause
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.state import State

#: Optional upstream RPC used to enable the fixed-block Ethereum fork.
JSON_RPC_ETHEREUM = os.environ.get("JSON_RPC_ETHEREUM")
#: Public deterministic Anvil account zero key; never a production secret.
DEPLOYER_PRIVATE_KEY = (
    "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
)
#: Synthetic account index whose public API response is mocked in these tests.
LIGHTER_ACCOUNT_INDEX = 321
#: Equity recorded in state before the simulated Lighter loss.
TRACKED_EQUITY = Decimal("10")
#: Lower live equity verifies loss-direction mismatches are never hidden.
CHANGED_EQUITY = Decimal("7.5")
#: One correction read followed by one verification read.
EXPECTED_EQUITY_READ_COUNT = 2

pytestmark = [
    pytest.mark.skipif(
        not JSON_RPC_ETHEREUM,
        reason="JSON_RPC_ETHEREUM environment variable required",
    ),
    pytest.mark.warm_rpc_test_group,
    pytest.mark.xdist_group("fork:ethereum:midnight"),
]


@dataclass
class StubLighterEquity:
    """Mutable public Lighter equity response."""

    total: Decimal

    def get_total(self) -> Decimal:
        return self.total


@pytest.fixture(scope="module")
def web3(anvil_fork_pool: AnvilForkPool) -> Web3:
    """Connect to the canonical fixed Ethereum fork shared by the test suite."""
    return anvil_fork_pool.get_web3(
        JSON_RPC_ETHEREUM,
        ETHEREUM_MIDNIGHT_BLOCK,
    )


@pytest.fixture()
def strategy_file() -> Path:
    """Return the static Lighter strategy used by CLI integration tests."""
    return (
        Path(__file__).resolve().parents[2]
        / "strategies"
        / "test_only"
        / "minimal_lighter_strategy.py"
    )


@pytest.fixture()
def state_file(tmp_path: Path) -> Path:
    """Create state representing the last successfully synced Lighter equity."""
    usdc = AssetIdentifier(
        chain_id=1,
        address=USDC_NATIVE_TOKEN[1],
        token_symbol="USDC",
        decimals=6,
    )
    pair = create_lighter_exchange_account_pair(
        quote=usdc,
        account_index=LIGHTER_ACCOUNT_INDEX,
    )
    state = State()
    open_exchange_account_position(
        state=state,
        strategy_cycle_at=native_datetime_utc_now(),
        pair=pair,
        reserve_currency=usdc,
        reserve_amount=TRACKED_EQUITY,
        notes="Previously synced Lighter equity",
    )
    state.portfolio.initialise_reserves(usdc)

    path = tmp_path / "lighter-cli-accounts.json"
    path.write_text(state.to_json_safe())
    return path


@pytest.fixture()
def environment(web3: Web3, strategy_file: Path, state_file: Path) -> dict[str, str]:
    """Configure both account commands against the shared Anvil endpoint."""
    return {
        "PATH": os.environ["PATH"],
        "EXECUTOR_ID": "lighter-cli-accounts",
        "STRATEGY_FILE": strategy_file.as_posix(),
        "STATE_FILE": state_file.as_posix(),
        "JSON_RPC_ETHEREUM": web3.provider.endpoint_uri,
        "PRIVATE_KEY": DEPLOYER_PRIVATE_KEY,
        "ASSET_MANAGEMENT_MODE": "hot_wallet",
        "LIGHTER_ACCOUNT_INDEX": str(LIGHTER_ACCOUNT_INDEX),
        "UNIT_TESTING": "true",
        "LOG_LEVEL": "disabled",
    }


def _run_cli_command(command: str, expected_exit_code: int) -> None:
    """Invoke a command through the real Typer command tree."""
    cli = get_command(app)
    with pytest.raises(SystemExit) as raised:
        cli.main(args=[command], standalone_mode=False)
    assert raised.value.code == expected_exit_code


def test_check_accounts_detects_changed_lighter_equity(
    environment: dict[str, str],
    state_file: Path,
    mocker: MockerFixture,
) -> None:
    """Report Lighter equity drift without changing the persisted state.

    1. Mock only Lighter's public equity at a value below the tracked position.
    2. Run the real ``check-accounts`` Typer command against fixed-block Anvil.
    3. Verify the command reports a mismatch and leaves accounting unchanged.
    """
    # 1. Mock only Lighter's public equity at a value below the tracked position.
    equity = StubLighterEquity(CHANGED_EQUITY)
    reader = mocker.patch(
        "tradeexecutor.exchange_account.lighter.fetch_lighter_total_equity",
        return_value=equity,
    )

    # 2. Run the real ``check-accounts`` Typer command against fixed-block Anvil.
    mocker.patch.dict("os.environ", environment, clear=True)
    _run_cli_command("check-accounts", expected_exit_code=1)

    # 3. Verify the command reports a mismatch and leaves accounting unchanged.
    state = State.read_json_file(state_file)
    position = next(iter(state.portfolio.open_positions.values()))
    assert position.get_quantity() == TRACKED_EQUITY
    assert position.balance_updates == {}
    assert state.sync.accounting.balance_update_refs == []
    reader.assert_called_once()
    assert reader.call_args.args[1] == LIGHTER_ACCOUNT_INDEX


def test_correct_accounts_updates_changed_lighter_equity(
    environment: dict[str, str],
    state_file: Path,
    mocker: MockerFixture,
) -> None:
    """Persist Lighter equity drift and make the next account check clean.

    1. Mock only Lighter's public equity at a value below the tracked position.
    2. Run the real ``correct-accounts`` Typer command and reload its state.
    3. Verify one block-anchored balance update records the complete change.
    4. Run ``check-accounts`` and verify the corrected state now matches.
    """
    # 1. Mock only Lighter's public equity at a value below the tracked position.
    equity = StubLighterEquity(CHANGED_EQUITY)
    reader = mocker.patch(
        "tradeexecutor.exchange_account.lighter.fetch_lighter_total_equity",
        return_value=equity,
    )
    mocker.patch.dict("os.environ", environment, clear=True)

    # 2. Run the real ``correct-accounts`` Typer command and reload its state.
    _run_cli_command("correct-accounts", expected_exit_code=0)
    state = State.read_json_file(state_file)
    position = next(iter(state.portfolio.open_positions.values()))

    # 3. Verify one block-anchored balance update records the complete change.
    assert position.get_quantity() == CHANGED_EQUITY
    assert len(position.balance_updates) == 1
    update = next(iter(position.balance_updates.values()))
    assert update.cause == BalanceUpdateCause.vault_flow
    assert update.old_balance == TRACKED_EQUITY
    assert update.quantity == CHANGED_EQUITY - TRACKED_EQUITY
    assert update.block_number == ETHEREUM_MIDNIGHT_BLOCK
    assert len(state.sync.accounting.balance_update_refs) == 1

    # 4. Run ``check-accounts`` and verify the corrected state now matches.
    _run_cli_command("check-accounts", expected_exit_code=0)
    assert reader.call_count == EXPECTED_EQUITY_READ_COUNT
    assert all(call.args[1] == LIGHTER_ACCOUNT_INDEX for call in reader.call_args_list)


def test_repair_preserves_healthy_lighter_position(
    environment: dict[str, str],
    state_file: Path,
    web3: Web3,
    mocker: MockerFixture,
) -> None:
    """Run repair without treating a Lighter position as an EVM trade.

    1. Guard the unavailable Lighter API and capture the original state.
    2. Run the real ``repair`` Typer command against fixed-block Anvil.
    3. Verify the healthy Lighter position and its accounting are unchanged.
    """
    # 1. Guard the unavailable Lighter API and capture the original state.
    reader = mocker.patch(
        "tradeexecutor.exchange_account.lighter.fetch_lighter_total_equity",
        side_effect=AssertionError("repair must not fetch Lighter equity"),
    )
    initial_state = State.read_json_file(state_file)
    initial_position = next(iter(initial_state.portfolio.open_positions.values()))
    initial_trade = next(iter(initial_position.trades.values()))

    # 2. Run the real ``repair`` Typer command against fixed-block Anvil.
    repair_environment = {**environment, "AUTO_APPROVE": "true"}
    mocker.patch.dict("os.environ", repair_environment, clear=True)
    get_command(app).main(args=["repair"], standalone_mode=False)

    # 3. Verify the healthy Lighter position and its accounting are unchanged.
    state = State.read_json_file(state_file)
    position = next(iter(state.portfolio.open_positions.values()))
    trade = next(iter(position.trades.values()))
    assert position.is_exchange_account()
    assert position.pair.get_exchange_account_protocol() == "lighter"
    assert position.get_quantity() == TRACKED_EQUITY
    assert position.balance_updates == initial_position.balance_updates == {}
    assert position.trades.keys() == initial_position.trades.keys()
    assert trade.get_status() == initial_trade.get_status()
    assert trade.executed_quantity == initial_trade.executed_quantity
    assert state.portfolio.next_trade_id == initial_state.portfolio.next_trade_id
    assert state.sync.accounting.balance_update_refs == []
    assert web3.eth.block_number == ETHEREUM_MIDNIGHT_BLOCK
    reader.assert_not_called()
