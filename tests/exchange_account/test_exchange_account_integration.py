"""Integration test for exchange account sync with strategy module.

This test verifies the full integration flow similar to CLI commands:
1. Load a strategy module that creates an exchange account universe
2. Create/initialise a state (similar to `trade-executor init`)
3. Create an exchange account position (similar to deposit flow)
4. Run sync_positions to detect balance changes from external exchange
5. Verify balance updates are created and tracked correctly

Note: Exchange account positions represent capital deployed to external perp DEXes
(Derive, Hyperliquid) where the actual trading happens off-chain. The sync model
polls the exchange API to detect PnL changes.

This follows the CLI test pattern from tests/cli/test_close_position.py but without
requiring Anvil since exchange accounts don't need on-chain infrastructure.
"""

import datetime
import os
import tempfile
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock

import pytest
from eth_account import Account

from tradeexecutor.exchange_account.sync_model import ExchangeAccountSyncModel
from tradeexecutor.state.balance_update import BalanceUpdateCause, BalanceUpdatePositionType
from tradeexecutor.state.identifier import (
    AssetIdentifier,
    TradingPairIdentifier,
    TradingPairKind,
)
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution, TradeType
from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode
from tradeexecutor.strategy.strategy_module import read_strategy_module
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradingstrategy.chain import ChainId


@pytest.fixture
def strategy_file() -> Path:
    """Path to the exchange account strategy module."""
    return Path(os.path.dirname(__file__)) / "../../strategies/test_only/exchange_account_strategy.py"


@pytest.fixture
def state_file() -> Path:
    """Create a temporary state file path (similar to CLI tests)."""
    return Path(tempfile.mkdtemp()) / "test-exchange-account-state.json"


@pytest.fixture
def exchange_account_pair():
    """Create an exchange account trading pair.

    Note: We create this directly because DEXPair roundtrip doesn't preserve
    exchange_account kind.
    """
    chain_id = ChainId.ethereum

    usdc = AssetIdentifier(
        chain_id=chain_id.value,
        address="0x0000000000000000000000000000000000000001",
        token_symbol="USDC",
        decimals=6,
    )

    exchange_account = AssetIdentifier(
        chain_id=chain_id.value,
        address="0x0000000000000000000000000000000000000002",
        token_symbol="DERIVE-ACCOUNT",
        decimals=6,
    )

    return TradingPairIdentifier(
        base=exchange_account,
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
            "exchange_subaccount_id": 1,
            "exchange_is_testnet": True,
        },
    )


def test_exchange_account_sync_cli_style(
    strategy_file: Path,
    state_file: Path,
    exchange_account_pair: TradingPairIdentifier,
):
    """End-to-end integration test for exchange account sync following CLI patterns.

    This test simulates the workflow:
    1. Initialise state (similar to `trade-executor init`)
    2. Load strategy module and create universe
    3. Create exchange account position with initial deposit (100k)
    4. Run sync cycles detecting PnL changes:
       - Cycle 1: 100k -> 105k (+5k profit)
       - Cycle 2: 105k -> 103k (-2k loss)
       - Cycle 3: 103k -> 110k (+7k profit)
       - Cycle 4: 110k -> 110k (no change)
    5. Verify balance updates and final state
    """
    # ==========================================================================
    # Step 1: Initialise state (similar to `trade-executor init`)
    # ==========================================================================
    state = State()

    # Save initial state to file (like CLI init command does)
    with state_file.open("wt") as f:
        f.write(state.to_json_safe())

    # Verify state file created
    assert state_file.exists()

    # ==========================================================================
    # Step 2: Load strategy module and create universe
    # ==========================================================================
    strategy_module = read_strategy_module(strategy_file)

    # Verify strategy module loaded correctly
    assert strategy_module is not None
    assert strategy_module.trading_strategy_engine_version == "0.5"

    # Create universe from strategy module
    execution_context = ExecutionContext(mode=ExecutionMode.unit_testing)
    universe_options = UniverseOptions()

    strategy_universe = strategy_module.create_trading_universe(
        ts=datetime.datetime.utcnow(),
        client=None,
        execution_context=execution_context,
        universe_options=universe_options,
    )

    # Verify universe
    assert strategy_universe.data_universe.pairs.get_count() == 1
    assert len(strategy_universe.reserve_assets) == 1
    assert strategy_universe.reserve_assets[0].token_symbol == "USDC"

    # ==========================================================================
    # Step 3: Verify exchange account pair properties
    # ==========================================================================
    assert exchange_account_pair.kind == TradingPairKind.exchange_account
    assert exchange_account_pair.is_exchange_account()
    assert exchange_account_pair.get_exchange_account_protocol() == "derive"
    assert exchange_account_pair.get_exchange_account_id() == 1

    # ==========================================================================
    # Step 4: Create position with initial deposit (simulates deposit to exchange)
    # ==========================================================================
    # Reload state from file (like CLI commands do between invocations)
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

    # Add initial deposit trade (100k USDC -> exchange account)
    trade = TradeExecution(
        trade_id=1,
        position_id=1,
        trade_type=TradeType.rebalance,
        pair=exchange_account_pair,
        opened_at=opened_at,
        planned_quantity=Decimal("100000.0"),
        planned_price=1.0,
        planned_reserve=Decimal("100000.0"),
        reserve_currency=exchange_account_pair.quote,
    )
    trade.started_at = opened_at
    trade.mark_broadcasted(opened_at)
    trade.mark_success(
        executed_at=opened_at,
        executed_price=1.0,
        executed_quantity=Decimal("100000.0"),
        executed_reserve=Decimal("100000.0"),
        lp_fees=0.0,
        native_token_price=1.0,
    )
    position.trades[1] = trade
    state.portfolio.open_positions[1] = position

    # Save state with position
    with state_file.open("wt") as f:
        f.write(state.to_json_safe())

    # Verify position state
    state = State.read_json_file(state_file)
    assert len(state.portfolio.open_positions) == 1
    position = state.portfolio.open_positions[1]
    assert position.get_quantity() == Decimal("100000.0")
    assert position.is_exchange_account()

    # ==========================================================================
    # Step 5: Run sync cycles (simulates periodic sync like main loop does)
    # ==========================================================================
    mock_value_func = Mock()
    sync_model = ExchangeAccountSyncModel(mock_value_func)

    # --- Cycle 1: Profit (100k -> 105k, +5k) ---
    mock_value_func.return_value = Decimal("105000.0")
    events = sync_model.sync_positions(
        timestamp=datetime.datetime(2024, 1, 2),
        state=state,
        strategy_universe=strategy_universe,
        pricing_model=None,
    )

    assert len(events) == 1
    evt = events[0]
    assert evt.quantity == Decimal("5000.0")
    assert evt.old_balance == Decimal("100000.0")
    assert evt.usd_value == 5000.0
    assert evt.cause == BalanceUpdateCause.vault_flow
    assert evt.position_type == BalanceUpdatePositionType.open_position
    assert evt.position_id == 1
    assert "derive" in evt.notes.lower()

    # Position quantity updated
    assert position.get_quantity() == Decimal("105000.0")
    assert len(position.balance_updates) == 1

    # --- Cycle 2: Loss (105k -> 103k, -2k) ---
    mock_value_func.return_value = Decimal("103000.0")
    events = sync_model.sync_positions(
        timestamp=datetime.datetime(2024, 1, 3),
        state=state,
        strategy_universe=strategy_universe,
        pricing_model=None,
    )

    assert len(events) == 1
    assert events[0].quantity == Decimal("-2000.0")
    assert position.get_quantity() == Decimal("103000.0")
    assert len(position.balance_updates) == 2

    # --- Cycle 3: Profit (103k -> 110k, +7k) ---
    mock_value_func.return_value = Decimal("110000.0")
    events = sync_model.sync_positions(
        timestamp=datetime.datetime(2024, 1, 4),
        state=state,
        strategy_universe=strategy_universe,
        pricing_model=None,
    )

    assert len(events) == 1
    assert events[0].quantity == Decimal("7000.0")
    assert position.get_quantity() == Decimal("110000.0")
    assert len(position.balance_updates) == 3

    # --- Cycle 4: No change (110k -> 110k) ---
    mock_value_func.return_value = Decimal("110000.0")
    events = sync_model.sync_positions(
        timestamp=datetime.datetime(2024, 1, 5),
        state=state,
        strategy_universe=strategy_universe,
        pricing_model=None,
    )

    # No balance update when value unchanged
    assert len(events) == 0
    assert len(position.balance_updates) == 3

    # ==========================================================================
    # Step 6: Verify final state (similar to CLI verification)
    # ==========================================================================
    # Save final state
    with state_file.open("wt") as f:
        f.write(state.to_json_safe())

    # Reload and verify
    final_state = State.read_json_file(state_file)

    assert len(final_state.portfolio.open_positions) == 1
    final_position = final_state.portfolio.open_positions[1]

    # Final position value: 100k + 5k - 2k + 7k = 110k
    assert final_position.get_quantity() == Decimal("110000.0")

    # All balance updates persisted
    assert len(final_position.balance_updates) == 3
    assert len(final_state.sync.accounting.balance_update_refs) == 3

    # Verify balance update details
    balance_updates = list(final_position.balance_updates.values())
    quantities = [bu.quantity for bu in balance_updates]
    assert Decimal("5000.0") in quantities   # Profit day 1
    assert Decimal("-2000.0") in quantities  # Loss day 2
    assert Decimal("7000.0") in quantities   # Profit day 3


# =============================================================================
# Real Derive API Test
# =============================================================================


@pytest.fixture(scope="module")
def derive_owner_account():
    """Owner wallet from environment."""
    if not os.environ.get("DERIVE_OWNER_PRIVATE_KEY"):
        pytest.skip("DERIVE_OWNER_PRIVATE_KEY not set")
    return Account.from_key(os.environ["DERIVE_OWNER_PRIVATE_KEY"])


@pytest.fixture(scope="module")
def derive_wallet_address(derive_owner_account):
    """Derive wallet address from env, or auto-derived from owner key."""
    from eth_defi.derive.onboarding import fetch_derive_wallet_address
    env_address = os.environ.get("DERIVE_WALLET_ADDRESS")
    if env_address:
        return env_address
    return fetch_derive_wallet_address(derive_owner_account.address, is_testnet=True)


@pytest.fixture(scope="module")
def derive_authenticated_client(derive_owner_account, derive_wallet_address):
    """Derive API client authenticated with session key."""
    from eth_defi.derive.authentication import DeriveApiClient
    from eth_defi.derive.account import fetch_subaccount_ids

    if not os.environ.get("DERIVE_SESSION_PRIVATE_KEY"):
        pytest.skip("DERIVE_SESSION_PRIVATE_KEY not set")

    client = DeriveApiClient(
        owner_account=derive_owner_account,
        derive_wallet_address=derive_wallet_address,
        is_testnet=True,
        session_key_private=os.environ["DERIVE_SESSION_PRIVATE_KEY"],
    )
    # Resolve subaccount ID from the API
    ids = fetch_subaccount_ids(client)
    if ids:
        client.subaccount_id = ids[0]
    return client


@pytest.fixture
def derive_exchange_account_pair(derive_authenticated_client):
    """Create an exchange account pair for real Derive testnet."""
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
    return TradingPairIdentifier(
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
            "exchange_subaccount_id": derive_authenticated_client.subaccount_id,
            "exchange_is_testnet": True,
        },
    )


@pytest.mark.skipif(
    not os.environ.get("DERIVE_OWNER_PRIVATE_KEY") or not os.environ.get("DERIVE_SESSION_PRIVATE_KEY"),
    reason="Set DERIVE_OWNER_PRIVATE_KEY and DERIVE_SESSION_PRIVATE_KEY for Derive tests",
)
def test_exchange_account_sync_real_derive_api(
    strategy_file: Path,
    derive_authenticated_client,
    derive_exchange_account_pair: TradingPairIdentifier,
):
    """Integration test with real Derive API following CLI patterns.

    This test:
    1. Creates a state file with an exchange account position
    2. Uses real Derive API to fetch account value
    3. Runs sync_positions to detect any balance difference
    4. Verifies the sync model correctly tracks the account value

    Required environment variables:
    - DERIVE_OWNER_PRIVATE_KEY: Owner wallet private key
    - DERIVE_SESSION_PRIVATE_KEY: Session key private key
    """
    from eth_defi.derive.account import fetch_subaccount_ids
    from tradeexecutor.exchange_account.derive import create_derive_account_value_func

    # Skip if no subaccounts
    ids = fetch_subaccount_ids(derive_authenticated_client)
    if not ids:
        pytest.skip("Account has no subaccounts yet")

    # ==========================================================================
    # Step 1: Initialise state file (like `trade-executor init`)
    # ==========================================================================
    state_file = Path(tempfile.mkdtemp()) / "test-derive-sync-state.json"
    state = State()

    with state_file.open("wt") as f:
        f.write(state.to_json_safe())

    # ==========================================================================
    # Step 2: Load strategy module and create universe
    # ==========================================================================
    strategy_module = read_strategy_module(strategy_file)
    execution_context = ExecutionContext(mode=ExecutionMode.unit_testing)
    universe_options = UniverseOptions()

    strategy_universe = strategy_module.create_trading_universe(
        ts=datetime.datetime.utcnow(),
        client=None,
        execution_context=execution_context,
        universe_options=universe_options,
    )

    # ==========================================================================
    # Step 3: Create position with expected value from Derive
    # ==========================================================================
    # First, fetch actual account value to use as baseline
    clients = {derive_authenticated_client.subaccount_id: derive_authenticated_client}
    account_value_func = create_derive_account_value_func(clients)

    actual_value = account_value_func(derive_exchange_account_pair)
    assert actual_value > 0, "Derive testnet account should have funds"

    # Create position with slightly different tracked value to trigger sync
    # Use 99% of actual value as "tracked" to simulate a small PnL change
    tracked_value = (actual_value * Decimal("0.99")).quantize(Decimal("0.01"))

    state = State.read_json_file(state_file)
    opened_at = datetime.datetime(2024, 1, 1)

    position = TradingPosition(
        position_id=1,
        pair=derive_exchange_account_pair,
        opened_at=opened_at,
        last_pricing_at=opened_at,
        last_token_price=1.0,
        last_reserve_price=1.0,
        reserve_currency=derive_exchange_account_pair.quote,
    )

    # Add initial deposit trade with tracked value
    trade = TradeExecution(
        trade_id=1,
        position_id=1,
        trade_type=TradeType.rebalance,
        pair=derive_exchange_account_pair,
        opened_at=opened_at,
        planned_quantity=tracked_value,
        planned_price=1.0,
        planned_reserve=tracked_value,
        reserve_currency=derive_exchange_account_pair.quote,
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

    # ==========================================================================
    # Step 4: Run sync with real Derive API
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
    # Step 5: Verify sync detected balance change
    # ==========================================================================
    # Should have one balance update (1% difference)
    assert len(events) == 1
    evt = events[0]

    # Expected difference is ~1% of actual value
    expected_diff = actual_value - tracked_value
    assert evt.quantity == expected_diff
    assert evt.old_balance == tracked_value
    assert evt.cause == BalanceUpdateCause.vault_flow
    assert evt.position_type == BalanceUpdatePositionType.open_position
    assert evt.position_id == 1
    assert "derive" in evt.notes.lower()

    # Position should now have actual value
    assert position.get_quantity() == actual_value

    # ==========================================================================
    # Step 6: Verify state persists correctly
    # ==========================================================================
    with state_file.open("wt") as f:
        f.write(state.to_json_safe())

    final_state = State.read_json_file(state_file)
    final_position = final_state.portfolio.open_positions[1]

    assert final_position.get_quantity() == actual_value
    assert len(final_position.balance_updates) == 1
    assert len(final_state.sync.accounting.balance_update_refs) == 1

    # ==========================================================================
    # Step 7: Run sync again - should have no change
    # ==========================================================================
    events_no_change = sync_model.sync_positions(
        timestamp=datetime.datetime.utcnow(),
        state=state,
        strategy_universe=strategy_universe,
        pricing_model=None,
    )

    # No balance update when synced value matches
    assert len(events_no_change) == 0
    assert len(position.balance_updates) == 1
