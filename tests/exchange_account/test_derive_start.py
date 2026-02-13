"""Test Derive exchange account strategy via CLI start command.

Run one cycle of the Derive start test strategy on a forked Arbitrum chain
with real Derive testnet connection.

To run:

.. code-block:: shell

    source .local-test.env && poetry run pytest tests/exchange_account/test_derive_start.py -v --log-cli-level=info

Requires environment variables:
- JSON_RPC_ARBITRUM
- TRADING_STRATEGY_API_KEY
- DERIVE_SESSION_PRIVATE_KEY
- DERIVE_WALLET_ADDRESS (or DERIVE_OWNER_PRIVATE_KEY)
"""

import os
import secrets
from pathlib import Path
from unittest import mock

import pytest

from eth_defi.provider.anvil import AnvilLaunch, launch_anvil

from tradeexecutor.cli.main import app
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeStatus


pytestmark = pytest.mark.skipif(
    not os.environ.get("JSON_RPC_ARBITRUM")
    or not os.environ.get("TRADING_STRATEGY_API_KEY")
    or not os.environ.get("DERIVE_SESSION_PRIVATE_KEY"),
    reason="Set JSON_RPC_ARBITRUM, TRADING_STRATEGY_API_KEY and DERIVE_SESSION_PRIVATE_KEY to run this test",
)
from tradeexecutor.utils.hex import hexbytes_to_hex_str


@pytest.fixture()
def anvil_arbitrum_fork() -> AnvilLaunch:
    """Fork Arbitrum mainnet via Anvil."""
    mainnet_rpc = os.environ["JSON_RPC_ARBITRUM"]
    anvil = launch_anvil(mainnet_rpc)
    try:
        yield anvil
    finally:
        anvil.close()


@pytest.fixture()
def strategy_file() -> Path:
    p = Path(__file__).resolve().parent / ".." / ".." / "strategies" / "test_only" / "derive_start_test.py"
    assert p.exists(), f"Strategy file missing: {p.resolve()}"
    return p


@pytest.fixture()
def environment(
    anvil_arbitrum_fork: AnvilLaunch,
    strategy_file: Path,
    tmp_path: Path,
    persistent_test_cache_path,
) -> dict:
    """Environment variables passed to init and start commands."""
    state_file = tmp_path / "test_derive_start.json"

    env = {
        "STRATEGY_FILE": strategy_file.as_posix(),
        "PRIVATE_KEY": hexbytes_to_hex_str(secrets.token_bytes(32)),
        "JSON_RPC_ARBITRUM": anvil_arbitrum_fork.json_rpc_url,
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
        # Derive credentials
        "DERIVE_SESSION_PRIVATE_KEY": os.environ["DERIVE_SESSION_PRIVATE_KEY"],
        "DERIVE_NETWORK": os.environ.get("DERIVE_NETWORK", "testnet"),
    }

    # Pass through optional Derive credentials
    if os.environ.get("DERIVE_OWNER_PRIVATE_KEY"):
        env["DERIVE_OWNER_PRIVATE_KEY"] = os.environ["DERIVE_OWNER_PRIVATE_KEY"]
    if os.environ.get("DERIVE_WALLET_ADDRESS"):
        env["DERIVE_WALLET_ADDRESS"] = os.environ["DERIVE_WALLET_ADDRESS"]

    return env


@pytest.mark.slow_test_group
def test_derive_start_single_cycle(environment: dict):
    """Run one cycle and verify exchange account position is created.

    - Fork Arbitrum with Anvil
    - Strategy creates one Derive exchange account position
    - Position is spoofed (trade marked success immediately)
    - No real on-chain trades happen
    """

    with mock.patch.dict("os.environ", environment, clear=True):
        app(["init"], standalone_mode=False)
        app(["start"], standalone_mode=False)

    # Load the resulting state and verify
    state_file = environment["STATE_FILE"]
    json_text = open(state_file, "rt").read()
    state = State.from_json(json_text)

    # Should have exactly one open position
    assert len(state.portfolio.open_positions) == 1

    position = list(state.portfolio.open_positions.values())[0]
    assert position.pair.is_exchange_account()
    assert position.pair.get_exchange_account_protocol() == "derive"

    # Should have one trade in success state
    assert len(position.trades) == 1
    trade = list(position.trades.values())[0]
    assert trade.get_status() == TradeStatus.success
