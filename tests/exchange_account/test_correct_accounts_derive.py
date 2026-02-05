"""Test correct-accounts CLI command with real Derive API.

Following pattern from tests/cli/test_cli_correct_account_price_missing.py
and tests/lagoon/test_lagoon_e2e.py.
"""

import datetime
import os
import tempfile
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pytest
from eth_defi.hotwallet import HotWallet
from typer.main import get_command

from tradeexecutor.cli.main import app
from tradeexecutor.state.state import State

pytestmark = pytest.mark.skipif(
    not os.environ.get("DERIVE_OWNER_PRIVATE_KEY") or not os.environ.get("DERIVE_SESSION_PRIVATE_KEY"),
    reason="Set DERIVE_OWNER_PRIVATE_KEY and DERIVE_SESSION_PRIVATE_KEY for Derive tests",
)


@pytest.fixture()
def strategy_file() -> Path:
    """Path to the exchange account strategy module."""
    return Path(os.path.dirname(__file__)) / "../../strategies/test_only/exchange_account_strategy.py"


@pytest.fixture()
def strategy_file_anvil() -> Path:
    """Path to the minimal strategy module for Anvil chain CLI testing."""
    return Path(os.path.dirname(__file__)) / "../../strategies/test_only/minimal_derive_strategy.py"


@pytest.fixture()
def state_file() -> Path:
    """Create a temporary state file path."""
    return Path(tempfile.mkdtemp()) / "test-correct-accounts-derive.json"


@pytest.fixture()
def environment(
    anvil,
    hot_wallet: HotWallet,
    state_file: Path,
    strategy_file: Path,
) -> dict:
    """Set up environment vars for correct-accounts CLI command.

    Following pattern from tests/cli/test_cli_correct_account_price_missing.py
    """
    environment = {
        "EXECUTOR_ID": "test_correct_accounts_derive",
        "STRATEGY_FILE": strategy_file.as_posix(),
        "PRIVATE_KEY": hot_wallet.account.key.hex(),
        "JSON_RPC_ANVIL": anvil.json_rpc_url,
        "STATE_FILE": state_file.as_posix(),
        "ASSET_MANAGEMENT_MODE": "hot_wallet",
        "UNIT_TESTING": "true",
        "TRADING_STRATEGY_API_KEY": os.environ.get("TRADING_STRATEGY_API_KEY", ""),
        # Derive credentials from environment
        "DERIVE_OWNER_PRIVATE_KEY": os.environ["DERIVE_OWNER_PRIVATE_KEY"],
        "DERIVE_SESSION_PRIVATE_KEY": os.environ["DERIVE_SESSION_PRIVATE_KEY"],
        "DERIVE_NETWORK": "testnet",
        # PATH needed for subprocess
        "PATH": os.environ.get("PATH", ""),
    }
    return environment


@pytest.fixture()
def environment_anvil(
    anvil,
    hot_wallet: HotWallet,
    state_file: Path,
    strategy_file_anvil: Path,
    usdc,
    dummy_token,
) -> dict:
    """Set up environment vars for init/start CLI commands on Anvil chain.

    Uses the anvil strategy which is configured for ChainId.anvil (31337).
    """
    environment = {
        "EXECUTOR_ID": "test_derive_cli_start",
        "STRATEGY_FILE": strategy_file_anvil.as_posix(),
        "PRIVATE_KEY": hot_wallet.account.key.hex(),
        "JSON_RPC_ANVIL": anvil.json_rpc_url,
        "STATE_FILE": state_file.as_posix(),
        "ASSET_MANAGEMENT_MODE": "hot_wallet",
        "UNIT_TESTING": "true",
        "TRADING_STRATEGY_API_KEY": os.environ.get("TRADING_STRATEGY_API_KEY", ""),
        # Derive credentials from environment
        "DERIVE_OWNER_PRIVATE_KEY": os.environ["DERIVE_OWNER_PRIVATE_KEY"],
        "DERIVE_SESSION_PRIVATE_KEY": os.environ["DERIVE_SESSION_PRIVATE_KEY"],
        "DERIVE_NETWORK": "testnet",
        # PATH needed for subprocess
        "PATH": os.environ.get("PATH", ""),
        # Token addresses for strategy to use
        "TEST_USDC_ADDRESS": usdc.address,
        "TEST_DUMMY_TOKEN_ADDRESS": dummy_token.address,
        # Run single cycle for start command tests
        "RUN_SINGLE_CYCLE": "true",
    }
    return environment


def _create_test_state_with_derive_position(state_file: Path) -> tuple[State, Decimal]:
    """Create test state with a Derive exchange account position.

    Creates a position with tracked value at 99% of actual Derive balance
    so that correct-accounts will detect a difference.

    :return: Tuple of (state, actual_value from Derive API)
    """
    from eth_account import Account
    from eth_defi.derive.account import fetch_subaccount_ids
    from eth_defi.derive.authentication import DeriveApiClient
    from eth_defi.derive.onboarding import fetch_derive_wallet_address

    from tradeexecutor.exchange_account.derive import \
        create_derive_account_value_func
    from tradeexecutor.state.identifier import (AssetIdentifier,
                                                TradingPairIdentifier,
                                                TradingPairKind)
    from tradeexecutor.state.position import TradingPosition
    from tradeexecutor.state.trade import TradeExecution, TradeType

    # Set up Derive client
    owner_private_key = os.environ["DERIVE_OWNER_PRIVATE_KEY"]
    session_private_key = os.environ["DERIVE_SESSION_PRIVATE_KEY"]

    owner_account = Account.from_key(owner_private_key)
    derive_wallet_address = fetch_derive_wallet_address(owner_account.address, is_testnet=True)

    client = DeriveApiClient(
        owner_account=owner_account,
        derive_wallet_address=derive_wallet_address,
        is_testnet=True,
        session_key_private=session_private_key,
    )

    ids = fetch_subaccount_ids(client)
    if not ids:
        pytest.skip("Account has no subaccounts yet")
    client.subaccount_id = ids[0]

    # Get actual account value
    clients = {client.subaccount_id: client}
    account_value_func = create_derive_account_value_func(clients)

    # Create exchange account pair
    chain_id = 901  # Derive testnet chain ID

    usdc = AssetIdentifier(
        chain_id=chain_id,
        address="0x0000000000000000000000000000000000000001",
        token_symbol="USDC",
        decimals=6,
    )
    derive_account = AssetIdentifier(
        chain_id=chain_id,
        address="0x0000000000000000000000000000000000000002",
        token_symbol="DERIVE-ACCOUNT",
        decimals=6,
    )
    exchange_account_pair = TradingPairIdentifier(
        base=derive_account,
        quote=usdc,
        pool_address="0x0000000000000000000000000000000000000003",
        exchange_address="0x0000000000000000000000000000000000000004",
        internal_id=1,
        internal_exchange_id=1,
        fee=0.0,
        kind=TradingPairKind.exchange_account,
        exchange_name="derive",
        other_data={
            "exchange_protocol": "derive",
            "exchange_subaccount_id": client.subaccount_id,
            "exchange_is_testnet": True,
        },
    )

    actual_value = account_value_func(exchange_account_pair)
    assert actual_value > 0, "Derive testnet account should have funds"

    # Create position with 99% of actual value to trigger sync
    tracked_value = (actual_value * Decimal("0.99")).quantize(Decimal("0.01"))

    state = State()
    opened_at = datetime.datetime(2024, 1, 1)

    position = TradingPosition(
        position_id=1,
        pair=exchange_account_pair,
        opened_at=opened_at,
        last_pricing_at=opened_at,
        last_token_price=1.0,
        last_reserve_price=1.0,
        reserve_currency=exchange_account_pair.quote,
    )

    # Add initial deposit trade with tracked value
    trade = TradeExecution(
        trade_id=1,
        position_id=1,
        trade_type=TradeType.rebalance,
        pair=exchange_account_pair,
        opened_at=opened_at,
        planned_quantity=tracked_value,
        planned_price=1.0,
        planned_reserve=tracked_value,
        reserve_currency=exchange_account_pair.quote,
    )
    trade.started_at = opened_at
    trade.mark_broadcasted(opened_at)
    trade.mark_success(
        executed_at=opened_at,
        executed_price=1.0,
        executed_quantity=tracked_value,
        executed_reserve=tracked_value,
        lp_fees=0.0,
        native_token_price=1.0,
    )
    position.trades[1] = trade
    state.portfolio.open_positions[1] = position

    # Save state file
    with state_file.open("wt") as f:
        f.write(state.to_json_safe())

    return state, actual_value


def test_correct_accounts_derive(
    environment: dict,
    state_file: Path,
):
    """Test correct-accounts syncs Derive exchange account positions.

    Following pattern from test_cli_correct_account_price_missing.py:
    1. Create state file with exchange account position (tracked value 99% of actual)
    2. Invoke correct-accounts CLI command
    3. Verify state file was updated with balance correction
    """
    _, actual_value = _create_test_state_with_derive_position(state_file)

    cli = get_command(app)
    with patch.dict(os.environ, environment, clear=True):
        with pytest.raises(SystemExit) as e:
            cli.main(args=["correct-accounts"])

        assert e.value.code == 0, f"CLI command failed with exit code {e.value.code}"

    # Verify state was updated
    final_state = State.read_json_file(state_file)
    final_position = final_state.portfolio.open_positions[1]
    final_quantity = final_position.get_quantity()

    # Balance should have been corrected (within small tolerance for API timing)
    assert abs(float(final_quantity) - float(actual_value)) < float(actual_value) * 0.02, \
        f"Expected quantity ~{actual_value}, got {final_quantity}"

    # Balance update should be recorded
    assert len(final_position.balance_updates) == 1
    assert len(final_state.sync.accounting.balance_update_refs) == 1


def test_derive_cli_start(
    environment_anvil: dict,
    state_file: Path,
):
    """Test init CLI command with Derive exchange account support.

    Following pattern from tests/lagoon/test_lagoon_e2e.py test_cli_lagoon_start.
    Tests that the strategy can be initialised with Derive environment variables.

    Note: Full start command testing requires DEX routing infrastructure
    which is complex to set up. This test verifies init works correctly.
    """
    cli = get_command(app)

    with patch.dict(os.environ, environment_anvil, clear=True):
        cli.main(args=["init"], standalone_mode=False)

    # Verify state was created
    state = State.read_json_file(state_file)

    assert state is not None
    assert state.sync is not None

