"""Test Derive vault integration on mainnet via CLI commands.

Run init, start, and correct-accounts against a live Derive mainnet vault
on Derive (Lyra) chain.

To run:

.. code-block:: shell

    source .local-test.env && poetry run pytest tests/exchange_account/test_derive_vault_mainnet.py -v --log-cli-level=info

Requires environment variables:
- TRADING_STRATEGY_API_KEY
- DERIVE_INTEGRATION_TEST_ASSET_MANAGER_PRIVATE_KEY
- DERIVE_INTEGRATION_TEST_SESSION_KEY
- DERIVE_INTEGRATION_TEST_VAULT_ADDRESS (Derive wallet address on Derive chain)
- DERIVE_INTEGRATION_TEST_SAFE_ADDRESS (Gnosis Safe address)
"""

import datetime
import os
import secrets
from decimal import Decimal
from pathlib import Path
from unittest import mock

import pytest
from typer.main import get_command

from eth_defi.provider.anvil import AnvilLaunch, launch_anvil

from tradeexecutor.cli.main import app
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeStatus
from tradeexecutor.utils.hex import hexbytes_to_hex_str


#: Default Derive mainnet RPC (public endpoint)
DERIVE_RPC_URL = os.environ.get("JSON_RPC_DERIVE", "https://rpc.derive.xyz")

pytestmark = pytest.mark.skipif(
    not os.environ.get("TRADING_STRATEGY_API_KEY")
    or not os.environ.get("DERIVE_INTEGRATION_TEST_ASSET_MANAGER_PRIVATE_KEY")
    or not os.environ.get("DERIVE_INTEGRATION_TEST_SESSION_KEY")
    or not os.environ.get("DERIVE_INTEGRATION_TEST_VAULT_ADDRESS")
    or not os.environ.get("DERIVE_INTEGRATION_TEST_SAFE_ADDRESS"),
    reason="Set TRADING_STRATEGY_API_KEY, DERIVE_INTEGRATION_TEST_ASSET_MANAGER_PRIVATE_KEY, DERIVE_INTEGRATION_TEST_SESSION_KEY, DERIVE_INTEGRATION_TEST_VAULT_ADDRESS and DERIVE_INTEGRATION_TEST_SAFE_ADDRESS",
)


@pytest.fixture()
def anvil_derive_fork() -> AnvilLaunch:
    """Fork Derive (Lyra) mainnet via Anvil."""
    anvil = launch_anvil(DERIVE_RPC_URL)
    try:
        yield anvil
    finally:
        anvil.close()


@pytest.fixture()
def strategy_file() -> Path:
    p = Path(__file__).resolve().parent / ".." / ".." / "strategies" / "test_only" / "derive_vault_mainnet_test.py"
    assert p.exists(), f"Strategy file missing: {p.resolve()}"
    return p


@pytest.fixture()
def environment(
    anvil_derive_fork: AnvilLaunch,
    strategy_file: Path,
    tmp_path: Path,
    persistent_test_cache_path,
) -> dict:
    """Environment variables passed to CLI commands.

    Maps integration test env vars to the standard Derive env vars
    expected by the CLI.
    """
    state_file = tmp_path / "test_derive_vault_mainnet.json"

    env = {
        "STRATEGY_FILE": strategy_file.as_posix(),
        "PRIVATE_KEY": hexbytes_to_hex_str(secrets.token_bytes(32)),
        "JSON_RPC_DERIVE": anvil_derive_fork.json_rpc_url,
        "STATE_FILE": state_file.as_posix(),
        "ASSET_MANAGEMENT_MODE": "hot_wallet",
        "UNIT_TESTING": "true",
        "LOG_LEVEL": "disabled",
        "TRADING_STRATEGY_API_KEY": os.environ["TRADING_STRATEGY_API_KEY"],
        "CACHE_PATH": persistent_test_cache_path,
        "CHECK_ACCOUNTS": "false",
        "RUN_SINGLE_CYCLE": "true",
        "MIN_GAS_BALANCE": "0.0",
        "SYNC_TREASURY_ON_STARTUP": "false",
        # Map integration test credentials to standard Derive env vars
        "DERIVE_OWNER_PRIVATE_KEY": os.environ["DERIVE_INTEGRATION_TEST_ASSET_MANAGER_PRIVATE_KEY"],
        "DERIVE_SESSION_PRIVATE_KEY": os.environ["DERIVE_INTEGRATION_TEST_SESSION_KEY"],
        "DERIVE_WALLET_ADDRESS": os.environ["DERIVE_INTEGRATION_TEST_VAULT_ADDRESS"],
        "DERIVE_NETWORK": "mainnet",
    }
    return env


@pytest.mark.slow_test_group
def test_derive_vault_mainnet_cli_init_and_start(environment: dict):
    """Run one cycle of init + start against the mainnet Derive vault.

    - Fork Derive chain with Anvil
    - Strategy creates one Derive exchange account position
    - Position is spoofed (trade marked success immediately)
    - Derive API is queried on mainnet for valuation
    """
    with mock.patch.dict("os.environ", environment, clear=True):
        app(["init"], standalone_mode=False)
        app(["start"], standalone_mode=False)

    state_file = environment["STATE_FILE"]
    json_text = open(state_file, "rt").read()
    state = State.from_json(json_text)

    assert len(state.portfolio.open_positions) == 1

    position = list(state.portfolio.open_positions.values())[0]
    assert position.pair.is_exchange_account()
    assert position.pair.get_exchange_account_protocol() == "derive"

    assert len(position.trades) == 1
    trade = list(position.trades.values())[0]
    assert trade.get_status() == TradeStatus.success


@pytest.mark.slow_test_group
def test_derive_vault_mainnet_correct_accounts(environment: dict):
    """Test correct-accounts syncs Derive mainnet vault balance.

    1. Create state with exchange account position at 99% of actual value
    2. Run correct-accounts CLI command
    3. Verify balance was corrected
    """
    state_file = Path(environment["STATE_FILE"])

    # Query actual vault balance and create state with a deliberate mismatch
    actual_value = _create_test_state_with_mainnet_position(state_file)

    cli = get_command(app)
    with mock.patch.dict("os.environ", environment, clear=True):
        with pytest.raises(SystemExit) as e:
            cli.main(args=["correct-accounts"])

        assert e.value.code == 0, f"CLI command failed with exit code {e.value.code}"

    final_state = State.read_json_file(state_file)
    final_position = final_state.portfolio.open_positions[1]
    final_quantity = final_position.get_quantity()

    assert abs(float(final_quantity) - float(actual_value)) < float(actual_value) * 0.02, \
        f"Expected quantity ~{actual_value}, got {final_quantity}"

    assert len(final_position.balance_updates) == 1
    assert len(final_state.sync.accounting.balance_update_refs) == 1


def _create_test_state_with_mainnet_position(state_file: Path) -> Decimal:
    """Create test state with a Derive mainnet exchange account position.

    Creates a position with tracked value at 99% of actual Derive balance
    so that correct-accounts will detect a difference.

    :return: Actual value from Derive API
    """
    from eth_account import Account
    from eth_defi.derive.account import fetch_subaccount_ids
    from eth_defi.derive.authentication import DeriveApiClient

    from tradeexecutor.exchange_account.derive import create_derive_account_value_func
    from tradeexecutor.state.identifier import (
        AssetIdentifier,
        TradingPairIdentifier,
        TradingPairKind,
    )
    from tradeexecutor.state.position import TradingPosition
    from tradeexecutor.state.trade import TradeExecution, TradeType

    owner_private_key = os.environ["DERIVE_INTEGRATION_TEST_ASSET_MANAGER_PRIVATE_KEY"]
    session_private_key = os.environ["DERIVE_INTEGRATION_TEST_SESSION_KEY"]
    derive_wallet_address = os.environ["DERIVE_INTEGRATION_TEST_VAULT_ADDRESS"]

    owner_account = Account.from_key(owner_private_key)

    client = DeriveApiClient(
        owner_account=owner_account,
        derive_wallet_address=derive_wallet_address,
        is_testnet=False,
        session_key_private=session_private_key,
    )

    ids = fetch_subaccount_ids(client)
    if not ids:
        pytest.skip("Account has no subaccounts")
    client.subaccount_id = ids[0]

    clients = {client.subaccount_id: client}
    account_value_func = create_derive_account_value_func(clients)

    # Use Derive mainnet chain ID to match the strategy file
    chain_id = 957

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
            "exchange_is_testnet": False,
        },
    )

    actual_value = account_value_func(exchange_account_pair)
    assert actual_value > 20, f"Expected at least 20 USD in vault, got {actual_value}"

    # Create position with 99% of actual value to trigger correction
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

    with state_file.open("wt") as f:
        f.write(state.to_json_safe())

    return actual_value
