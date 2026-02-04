"""Test exchange account sync CLI command with real Derive API.

This test follows the CLI test pattern from tests/cli/test_cli_correct_account_price_missing.py:
- Uses environment variables
- Uses state file persistence
- Invokes the sync-exchange-account CLI command via get_command(app)

Required environment variables:
- DERIVE_OWNER_PRIVATE_KEY: Owner wallet private key (from web UI wallet)
- DERIVE_SESSION_PRIVATE_KEY: Session key private key (from testnet developer page)
- DERIVE_WALLET_ADDRESS: Derive wallet address (optional, auto-derived from owner key)

See tests/derive/derive-test-key-setup.md for detailed instructions.
"""

import datetime
import logging
import os
import tempfile
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.main import get_command

from tradeexecutor.cli.main import app
from tradeexecutor.state.state import State


logger = logging.getLogger(__name__)


pytestmark = pytest.mark.skipif(
    not os.environ.get("DERIVE_OWNER_PRIVATE_KEY") or not os.environ.get("DERIVE_SESSION_PRIVATE_KEY"),
    reason="Set DERIVE_OWNER_PRIVATE_KEY and DERIVE_SESSION_PRIVATE_KEY for Derive tests. See tests/derive/derive-test-key-setup.md",
)


@pytest.fixture()
def strategy_file() -> Path:
    """Path to the exchange account strategy module."""
    return Path(os.path.dirname(__file__)) / "../../strategies/test_only/exchange_account_strategy.py"


@pytest.fixture()
def state_file() -> Path:
    """Create a temporary state file path."""
    return Path(tempfile.mkdtemp()) / "test-derive-sync-state.json"


@pytest.fixture()
def environment(
    strategy_file: Path,
    state_file: Path,
) -> dict:
    """Set up environment vars for Derive CLI test.

    Following pattern from tests/cli/test_cli_correct_account_price_missing.py
    """
    environment = {
        "EXECUTOR_ID": "test_exchange_account_derive",
        "STRATEGY_FILE": strategy_file.as_posix(),
        "STATE_FILE": state_file.as_posix(),
        "UNIT_TESTING": "true",
        "LOG_LEVEL": "info",
        # Derive-specific environment variables
        "DERIVE_OWNER_PRIVATE_KEY": os.environ.get("DERIVE_OWNER_PRIVATE_KEY", ""),
        "DERIVE_SESSION_PRIVATE_KEY": os.environ.get("DERIVE_SESSION_PRIVATE_KEY", ""),
        "DERIVE_WALLET_ADDRESS": os.environ.get("DERIVE_WALLET_ADDRESS", ""),
        "DERIVE_IS_TESTNET": "true",
        # PATH needed for subprocess
        "PATH": os.environ.get("PATH", ""),
    }
    return environment


def test_sync_exchange_account_cli(
    logger: logging.Logger,
    environment: dict,
    state_file: Path,
):
    """Test sync-exchange-account CLI command with real Derive API.

    This test follows the CLI test pattern:
    1. Creates a state file with an exchange account position (tracked value 99% of actual)
    2. Invokes sync-exchange-account CLI command
    3. Verifies the state file was updated with balance correction

    The CLI command:
    - Reads environment variables for Derive credentials
    - Connects to Derive API and fetches actual account value
    - Updates position balance to match actual value
    - Saves the corrected state file
    """
    from eth_account import Account
    from eth_defi.derive.authentication import DeriveApiClient
    from eth_defi.derive.account import fetch_subaccount_ids
    from eth_defi.derive.onboarding import fetch_derive_wallet_address

    from tradeexecutor.exchange_account.derive import create_derive_account_value_func
    from tradeexecutor.state.identifier import (
        AssetIdentifier,
        TradingPairIdentifier,
        TradingPairKind,
    )
    from tradeexecutor.state.position import TradingPosition
    from tradeexecutor.state.trade import TradeExecution, TradeType

    # ==========================================================================
    # Step 1: Set up Derive client to get actual value and subaccount ID
    # ==========================================================================
    owner_private_key = environment["DERIVE_OWNER_PRIVATE_KEY"]
    session_private_key = environment["DERIVE_SESSION_PRIVATE_KEY"]
    wallet_address = environment.get("DERIVE_WALLET_ADDRESS")

    owner_account = Account.from_key(owner_private_key)

    if not wallet_address:
        wallet_address = fetch_derive_wallet_address(owner_account.address, is_testnet=True)

    client = DeriveApiClient(
        owner_account=owner_account,
        derive_wallet_address=wallet_address,
        is_testnet=True,
        session_key_private=session_private_key,
    )

    # Resolve subaccount ID from the API
    ids = fetch_subaccount_ids(client)
    if not ids:
        pytest.skip("Account has no subaccounts yet")
    client.subaccount_id = ids[0]

    logger.info("Derive subaccount ID: %s", client.subaccount_id)

    # Get actual account value
    clients = {client.subaccount_id: client}
    account_value_func = create_derive_account_value_func(clients)

    # ==========================================================================
    # Step 2: Create exchange account pair
    # ==========================================================================
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
    logger.info("Actual Derive account value: %s", actual_value)

    # ==========================================================================
    # Step 3: Create state file with position (tracked value 99% of actual)
    # ==========================================================================
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

    logger.info("State file created at %s with tracked value %s", state_file, tracked_value)

    # Verify initial state
    initial_state = State.read_json_file(state_file)
    assert initial_state.portfolio.open_positions[1].get_quantity() == tracked_value

    # ==========================================================================
    # Step 4: Invoke CLI command
    # ==========================================================================
    cli = get_command(app)
    with patch.dict(os.environ, environment, clear=True):
        with pytest.raises(SystemExit) as e:
            cli.main(args=["sync-exchange-account"])

        assert e.value.code == 0, f"CLI command failed with exit code {e.value.code}"

    # ==========================================================================
    # Step 5: Verify state file was updated
    # ==========================================================================
    final_state = State.read_json_file(state_file)
    final_position = final_state.portfolio.open_positions[1]

    # Position should now have actual value (within small tolerance for API timing)
    final_quantity = final_position.get_quantity()
    logger.info("Final quantity: %s (expected ~%s)", final_quantity, actual_value)

    # Allow small tolerance for value changes during test execution
    assert abs(float(final_quantity) - float(actual_value)) < float(actual_value) * 0.02, \
        f"Expected quantity ~{actual_value}, got {final_quantity}"

    # Balance update should be recorded
    assert len(final_position.balance_updates) == 1
    assert len(final_state.sync.accounting.balance_update_refs) == 1

    logger.info("CLI sync-exchange-account test passed")


def test_sync_exchange_account_cli_no_change(
    logger: logging.Logger,
    environment: dict,
    state_file: Path,
):
    """Test sync-exchange-account CLI when account value matches tracked value.

    When the tracked value equals the actual account value, no balance update
    should be created.
    """
    from eth_account import Account
    from eth_defi.derive.authentication import DeriveApiClient
    from eth_defi.derive.account import fetch_subaccount_ids
    from eth_defi.derive.onboarding import fetch_derive_wallet_address

    from tradeexecutor.exchange_account.derive import create_derive_account_value_func
    from tradeexecutor.state.identifier import (
        AssetIdentifier,
        TradingPairIdentifier,
        TradingPairKind,
    )
    from tradeexecutor.state.position import TradingPosition
    from tradeexecutor.state.trade import TradeExecution, TradeType

    # Set up Derive client
    owner_private_key = environment["DERIVE_OWNER_PRIVATE_KEY"]
    session_private_key = environment["DERIVE_SESSION_PRIVATE_KEY"]
    wallet_address = environment.get("DERIVE_WALLET_ADDRESS")

    owner_account = Account.from_key(owner_private_key)

    if not wallet_address:
        wallet_address = fetch_derive_wallet_address(owner_account.address, is_testnet=True)

    client = DeriveApiClient(
        owner_account=owner_account,
        derive_wallet_address=wallet_address,
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
    chain_id = 901
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
    assert actual_value > 0

    # Create state with position tracking EXACT actual value
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

    trade = TradeExecution(
        trade_id=1,
        position_id=1,
        trade_type=TradeType.rebalance,
        pair=exchange_account_pair,
        opened_at=opened_at,
        planned_quantity=actual_value,
        planned_price=1.0,
        planned_reserve=actual_value,
        reserve_currency=exchange_account_pair.quote,
    )
    trade.started_at = opened_at
    trade.mark_broadcasted(opened_at)
    trade.mark_success(
        executed_at=opened_at,
        executed_price=1.0,
        executed_quantity=actual_value,
        executed_reserve=actual_value,
        lp_fees=0.0,
        native_token_price=1.0,
    )
    position.trades[1] = trade
    state.portfolio.open_positions[1] = position

    with state_file.open("wt") as f:
        f.write(state.to_json_safe())

    # Invoke CLI
    cli = get_command(app)
    with patch.dict(os.environ, environment, clear=True):
        with pytest.raises(SystemExit) as e:
            cli.main(args=["sync-exchange-account"])

        assert e.value.code == 0

    # Verify no balance update was created (value unchanged)
    final_state = State.read_json_file(state_file)
    final_position = final_state.portfolio.open_positions[1]

    # No balance updates should be recorded when value matches
    assert len(final_position.balance_updates) == 0

    logger.info("CLI sync-exchange-account no-change test passed")
