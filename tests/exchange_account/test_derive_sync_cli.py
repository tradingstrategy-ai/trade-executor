"""Test exchange account sync with real Derive API.

This test follows the CLI test pattern from tests/cli/test_cli_correct_account_price_missing.py:
- Uses environment variables
- Uses state file persistence
- Tests the sync_positions flow that corrects exchange account balances

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
        "LOG_LEVEL": "disabled",
        # Derive-specific environment variables
        "DERIVE_OWNER_PRIVATE_KEY": os.environ.get("DERIVE_OWNER_PRIVATE_KEY", ""),
        "DERIVE_SESSION_PRIVATE_KEY": os.environ.get("DERIVE_SESSION_PRIVATE_KEY", ""),
        "DERIVE_WALLET_ADDRESS": os.environ.get("DERIVE_WALLET_ADDRESS", ""),
        # PATH needed for subprocess
        "PATH": os.environ.get("PATH", ""),
    }
    return environment


def test_exchange_account_sync_real_derive_api(
    logger: logging.Logger,
    environment: dict,
    state_file: Path,
    strategy_file: Path,
):
    """Integration test with real Derive API following CLI test patterns.

    This test follows the pattern from tests/cli/test_cli_correct_account_price_missing.py:
    - Uses environment variables
    - Uses state file persistence
    - Tests the full sync flow

    The test:
    1. Sets up Derive API client using environment variables
    2. Creates a state file with an exchange account position
    3. Uses real Derive API to fetch account value
    4. Runs sync_positions to detect any balance difference
    5. Verifies the sync model correctly tracks the account value
    """
    from eth_account import Account
    from eth_defi.derive.authentication import DeriveApiClient
    from eth_defi.derive.account import fetch_subaccount_ids
    from eth_defi.derive.onboarding import fetch_derive_wallet_address

    from tradeexecutor.exchange_account.derive import create_derive_account_value_func
    from tradeexecutor.exchange_account.sync_model import ExchangeAccountSyncModel
    from tradeexecutor.state.balance_update import BalanceUpdateCause, BalanceUpdatePositionType
    from tradeexecutor.state.identifier import (
        AssetIdentifier,
        TradingPairIdentifier,
        TradingPairKind,
    )
    from tradeexecutor.state.position import TradingPosition
    from tradeexecutor.state.trade import TradeExecution, TradeType
    from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode
    from tradeexecutor.strategy.strategy_module import read_strategy_module
    from tradeexecutor.strategy.universe_model import UniverseOptions

    # ==========================================================================
    # Step 1: Set up Derive API client using environment variables
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

    logger.info("Derive client set up with subaccount %s", client.subaccount_id)

    # ==========================================================================
    # Step 2: Initialise state file (like `trade-executor init`)
    # ==========================================================================
    state = State()
    with state_file.open("wt") as f:
        f.write(state.to_json_safe())

    assert state_file.exists(), f"State file not created at {state_file}"

    # ==========================================================================
    # Step 3: Load strategy module and create universe
    # ==========================================================================
    strategy_path = Path(environment["STRATEGY_FILE"])
    strategy_module = read_strategy_module(strategy_path)
    execution_context = ExecutionContext(mode=ExecutionMode.unit_testing)
    universe_options = UniverseOptions()

    strategy_universe = strategy_module.create_trading_universe(
        ts=datetime.datetime.utcnow(),
        client=None,
        execution_context=execution_context,
        universe_options=universe_options,
    )

    assert strategy_universe.data_universe.pairs.get_count() == 1
    logger.info("Strategy universe created with %d pairs", strategy_universe.data_universe.pairs.get_count())

    # ==========================================================================
    # Step 4: Create exchange account pair and fetch actual value from Derive
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

    # Fetch actual account value from Derive API
    clients = {client.subaccount_id: client}
    account_value_func = create_derive_account_value_func(clients)

    actual_value = account_value_func(exchange_account_pair)
    assert actual_value > 0, "Derive testnet account should have funds"
    logger.info("Actual Derive account value: %s", actual_value)

    # ==========================================================================
    # Step 5: Create position with tracked value (99% of actual to trigger sync)
    # ==========================================================================
    # Use 99% of actual value so sync will detect the 1% difference
    tracked_value = (actual_value * Decimal("0.99")).quantize(Decimal("0.01"))

    state = State.read_json_file(state_file)
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

    # Save state with position
    with state_file.open("wt") as f:
        f.write(state.to_json_safe())

    logger.info("Position created with tracked value: %s (actual: %s)", tracked_value, actual_value)

    # ==========================================================================
    # Step 6: Run sync with real Derive API
    # ==========================================================================
    sync_model = ExchangeAccountSyncModel(account_value_func)

    # Verify initial position state
    assert position.get_quantity() == tracked_value
    assert position.is_exchange_account()

    # Run sync - should detect difference between tracked and actual value
    events = sync_model.sync_positions(
        timestamp=datetime.datetime.utcnow(),
        state=state,
        strategy_universe=strategy_universe,
        pricing_model=None,
    )

    # ==========================================================================
    # Step 7: Verify sync detected balance change
    # ==========================================================================
    assert len(events) == 1, f"Expected 1 balance update event, got {len(events)}"
    evt = events[0]

    # Expected difference is ~1% of actual value
    expected_diff = actual_value - tracked_value
    assert evt.quantity == expected_diff, f"Expected quantity {expected_diff}, got {evt.quantity}"
    assert evt.old_balance == tracked_value
    assert evt.cause == BalanceUpdateCause.vault_flow
    assert evt.position_type == BalanceUpdatePositionType.open_position
    assert evt.position_id == 1
    assert "derive" in evt.notes.lower()

    # Position should now have actual value
    assert position.get_quantity() == actual_value

    logger.info("Sync detected balance change: %s -> %s (diff: %s)", tracked_value, actual_value, expected_diff)

    # ==========================================================================
    # Step 8: Verify state persists correctly
    # ==========================================================================
    with state_file.open("wt") as f:
        f.write(state.to_json_safe())

    final_state = State.read_json_file(state_file)
    final_position = final_state.portfolio.open_positions[1]

    assert final_position.get_quantity() == actual_value
    assert len(final_position.balance_updates) == 1
    assert len(final_state.sync.accounting.balance_update_refs) == 1

    logger.info("Final state persisted with quantity: %s", final_position.get_quantity())

    # ==========================================================================
    # Step 9: Run sync again - should have no change
    # ==========================================================================
    events_no_change = sync_model.sync_positions(
        timestamp=datetime.datetime.utcnow(),
        state=state,
        strategy_universe=strategy_universe,
        pricing_model=None,
    )

    # No balance update when synced value matches
    assert len(events_no_change) == 0, f"Expected no balance updates, got {len(events_no_change)}"
    assert len(position.balance_updates) == 1

    logger.info("Second sync detected no change - test passed")
