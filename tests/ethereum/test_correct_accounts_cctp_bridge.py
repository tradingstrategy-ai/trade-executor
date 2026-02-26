"""Test correct-accounts CLI command with CCTP bridge positions.

Verifies that the correct-accounts CLI auto-creates and corrects
accounting for both:
1. Main reserve (source USDC)
2. CCTP bridge position (destination USDC)

Uses a single Anvil instance with two deployed ERC-20 tokens
to simulate source and destination USDC.

Flow:
1. ``cli init`` creates empty state
2. ``cli correct-accounts`` loads universe, auto-creates missing
   bridge positions, detects on-chain balances, corrects everything
"""

import os
import secrets
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from eth_account import Account
from eth_typing import HexAddress
from web3 import Web3, HTTPProvider
from web3.contract import Contract

from eth_defi.hotwallet import HotWallet
from eth_defi.provider.anvil import AnvilLaunch, launch_anvil
from eth_defi.chain import install_chain_middleware
from eth_defi.token import create_token
from eth_defi.trace import assert_transaction_success_with_explanation
from typer.main import get_command

from tradeexecutor.cli.main import app
from tradeexecutor.state.identifier import TradingPairKind
from tradeexecutor.state.state import State
from tradeexecutor.utils.hex import hexbytes_to_hex_str

# Fixtures below are not directly used in test bodies but are required
# as transitive pytest fixture dependencies (environment → hot_wallet → usdc_source etc.)


# --- Fixtures ---


@pytest.fixture()
def anvil() -> AnvilLaunch:
    """Launch local Anvil node."""
    anvil = launch_anvil()
    try:
        yield anvil
    finally:
        anvil.close()


@pytest.fixture()
def web3(anvil: AnvilLaunch) -> Web3:
    """Set up the Anvil Web3 connection."""
    web3 = Web3(HTTPProvider(anvil.json_rpc_url, request_kwargs={"timeout": 2}))
    web3.middleware_onion.clear()
    install_chain_middleware(web3)
    return web3


@pytest.fixture()
def deployer(web3) -> HexAddress:
    """Deployer account with ETH."""
    return web3.eth.accounts[0]


@pytest.fixture()
def usdc_source(web3, deployer) -> Contract:
    """Mock source chain USDC token (reserve currency)."""
    token = create_token(web3, deployer, "USD Coin Source", "USDC", 100_000_000 * 10**6, decimals=6)
    return token


@pytest.fixture()
def usdc_dest(web3, deployer) -> Contract:
    """Mock destination chain USDC token (bridged currency)."""
    token = create_token(web3, deployer, "USD Coin Bridged", "USDC-BRIDGED", 100_000_000 * 10**6, decimals=6)
    return token


@pytest.fixture()
def hot_wallet(web3, deployer, usdc_source: Contract, usdc_dest: Contract) -> HotWallet:
    """Create hot wallet funded with ETH and both USDC tokens.

    Simulates a wallet that started with 10,000 source USDC
    and bridged 5,000 to the destination chain:

    - 15 ETH for gas
    - 5000 source USDC (remaining after bridge)
    - 5000 destination USDC (bridged)
    - Total portfolio: 10,000 USDC
    """
    private_key = hexbytes_to_hex_str(secrets.token_bytes(32))
    account = Account.from_key(private_key)
    wallet = HotWallet(account)
    wallet.sync_nonce(web3)

    # Fund with ETH
    user_1 = web3.eth.accounts[1]
    tx_hash = web3.eth.send_transaction({"to": wallet.address, "from": user_1, "value": 15 * 10**18})
    assert_transaction_success_with_explanation(web3, tx_hash)

    # Fund with source USDC (5000 remaining after bridging 5000)
    tx_hash = usdc_source.functions.transfer(wallet.address, 5_000 * 10**6).transact({"from": deployer})
    assert_transaction_success_with_explanation(web3, tx_hash)

    # Fund with destination USDC (5000 bridged from source)
    tx_hash = usdc_dest.functions.transfer(wallet.address, 5_000 * 10**6).transact({"from": deployer})
    assert_transaction_success_with_explanation(web3, tx_hash)

    return wallet


@pytest.fixture()
def strategy_file() -> Path:
    """Path to the CCTP bridge strategy module."""
    return Path(os.path.dirname(__file__)) / "../../strategies/test_only/cctp_bridge_strategy.py"


@pytest.fixture()
def state_file() -> Path:
    """Create a temporary state file path."""
    return Path(tempfile.mkdtemp()) / "test-correct-accounts-cctp-bridge.json"


@pytest.fixture()
def environment(
    anvil: AnvilLaunch,
    hot_wallet: HotWallet,
    state_file: Path,
    strategy_file: Path,
    usdc_source: Contract,
    usdc_dest: Contract,
) -> dict:
    """Set up environment vars for correct-accounts CLI command."""
    environment = {
        "EXECUTOR_ID": "test_correct_accounts_cctp_bridge",
        "STRATEGY_FILE": strategy_file.as_posix(),
        "PRIVATE_KEY": hexbytes_to_hex_str(hot_wallet.account.key),
        "JSON_RPC_ANVIL": anvil.json_rpc_url,
        "STATE_FILE": state_file.as_posix(),
        "ASSET_MANAGEMENT_MODE": "hot_wallet",
        "UNIT_TESTING": "true",
        "TRADING_STRATEGY_API_KEY": os.environ.get("TRADING_STRATEGY_API_KEY", ""),
        # Token addresses for the strategy to discover
        "TEST_USDC_SOURCE_ADDRESS": usdc_source.address,
        "TEST_USDC_DEST_ADDRESS": usdc_dest.address,
        # PATH needed for subprocess
        "PATH": os.environ.get("PATH", ""),
    }
    return environment


# --- Tests ---


def test_correct_accounts_cctp_bridge(
    environment: dict,
    state_file: Path,
):
    """Test correct-accounts auto-creates and corrects bridge position.

    Simulates a wallet that started with 10,000 source USDC and bridged
    5,000 to destination. On-chain state: 5,000 source + 5,000 dest.

    Flow:
    1. Run ``cli init`` to create empty state
    2. Run ``cli correct-accounts`` which auto-creates bridge position
       from universe and corrects both reserve and bridge to match
       on-chain balances
    3. Verify reserve = 5,000 (source USDC remaining after bridge)
    4. Verify bridge position = 5,000 (destination USDC)
    5. Verify total portfolio = 10,000
    6. Verify trade counts and balance updates
    """
    cli = get_command(app)

    # Step 1: Init (creates empty state)
    with patch.dict(os.environ, environment, clear=True):
        cli.main(args=["init"], standalone_mode=False)

    # Verify state exists but has no positions
    initial_state = State.read_json_file(state_file)
    assert len(initial_state.portfolio.open_positions) == 0, \
        "State should have no positions after init"

    # Step 2: Run correct-accounts (should auto-create bridge position and correct balances)
    with patch.dict(os.environ, environment, clear=True):
        with pytest.raises(SystemExit) as e:
            cli.main(args=["correct-accounts"])
        assert e.value.code == 0, f"CLI command failed with exit code {e.value.code}"

    # Step 3: Read back corrected state
    final_state = State.read_json_file(state_file)

    # Verify reserve was initialised and corrected to on-chain source balance
    # (5000 remaining after bridging 5000 from original 10000)
    final_reserve = final_state.portfolio.get_default_reserve_position()
    assert final_reserve is not None, "Reserve should exist after correct-accounts"
    assert float(final_reserve.quantity) == pytest.approx(5_000, rel=0.02), \
        f"Reserve expected ~5000, got {final_reserve.quantity}"

    # Verify bridge position was auto-created
    assert len(final_state.portfolio.open_positions) == 1, \
        f"Expected 1 open position, got {len(final_state.portfolio.open_positions)}"
    bridge_position = list(final_state.portfolio.open_positions.values())[0]
    assert bridge_position.pair.kind == TradingPairKind.cctp_bridge, \
        f"Expected cctp_bridge position, got {bridge_position.pair.kind}"

    # Verify bridge position was corrected to on-chain dest balance (5000 bridged)
    final_bridge_qty = bridge_position.get_quantity()
    assert float(final_bridge_qty) == pytest.approx(5_000, rel=0.02), \
        f"Bridge position expected ~5000, got {final_bridge_qty}"

    # Verify total portfolio = reserve + bridge = 10,000
    total_portfolio = float(final_reserve.quantity) + float(final_bridge_qty)
    assert total_portfolio == pytest.approx(10_000, rel=0.02), \
        f"Total portfolio expected ~10000, got {total_portfolio}"

    # Verify bridge position has exactly 1 trade (auto-creation)
    trades = list(bridge_position.trades.values())
    assert len(trades) == 1, \
        f"Expected exactly 1 trade on bridge position, got {len(trades)}"
    assert "Auto-created by correct-accounts for CCTP bridge" in (trades[0].notes or ""), \
        f"Expected auto-creation notes, got: {trades[0].notes}"

    # Verify balance updates were recorded for the correction
    assert len(bridge_position.balance_updates) == 1, \
        f"Expected 1 balance update for bridge position, got {len(bridge_position.balance_updates)}"
    assert len(final_state.sync.accounting.balance_update_refs) >= 1, \
        f"Expected balance update refs, got {len(final_state.sync.accounting.balance_update_refs)}"
