"""Reconcile settled cash moved externally between two test chains.

This is balance-correction coverage, not CCTP execution evidence. Actual burn,
receive, interruption and restart are covered on fixed Lagoon forks by
``tests/lagoon/test_multichain_account_commands.py``. A recoverable CCTP message
must never be written off merely because destination cash has not arrived.
"""

import os
import secrets
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from eth_account import Account
from eth_typing import HexAddress
from web3 import Web3
from web3.contract import Contract

from eth_defi.hotwallet import HotWallet
from eth_defi.provider.anvil import AnvilLaunch, launch_anvil
from eth_defi.token import create_token
from eth_defi.trace import assert_transaction_success_with_explanation
from eth_defi.provider.multi_provider import create_multi_provider_web3
from tradingstrategy.chain import ChainId
from typer.main import get_command

from tradeexecutor.cli.main import app
from tradeexecutor.state.identifier import TradingPairKind
from tradeexecutor.state.state import State
from tradeexecutor.utils.hex import hexbytes_to_hex_str


# --- Fixtures ---


@pytest.fixture()
def anvil_source() -> AnvilLaunch:
    """Source chain (Arbitrum-like) Anvil node."""
    anvil = launch_anvil()
    try:
        yield anvil
    finally:
        anvil.close()


@pytest.fixture()
def anvil_dest() -> AnvilLaunch:
    """Destination chain (Base-like) Anvil node."""
    anvil = launch_anvil()
    try:
        yield anvil
    finally:
        anvil.close()


@pytest.fixture()
def web3_source(anvil_source: AnvilLaunch) -> Web3:
    web3 = create_multi_provider_web3(anvil_source.json_rpc_url)
    return web3


@pytest.fixture()
def web3_dest(anvil_dest: AnvilLaunch) -> Web3:
    web3 = create_multi_provider_web3(anvil_dest.json_rpc_url)
    return web3


@pytest.fixture()
def deployer_source(web3_source) -> HexAddress:
    return web3_source.eth.accounts[0]


@pytest.fixture()
def deployer_dest(web3_dest) -> HexAddress:
    return web3_dest.eth.accounts[0]


@pytest.fixture()
def usdc_source(web3_source, deployer_source) -> Contract:
    return create_token(web3_source, deployer_source, "USD Coin", "USDC", 100_000_000 * 10**6, decimals=6)


@pytest.fixture()
def usdc_dest(web3_dest, deployer_dest) -> Contract:
    return create_token(web3_dest, deployer_dest, "USD Coin Bridged", "USDC-BRIDGED", 100_000_000 * 10**6, decimals=6)


@pytest.fixture()
def hot_wallet(
    web3_source,
    web3_dest,
    deployer_source,
    deployer_dest,
    usdc_source: Contract,
    usdc_dest: Contract,
) -> HotWallet:
    """Create hot wallet funded with 5,000 USDC on each chain."""
    private_key = hexbytes_to_hex_str(secrets.token_bytes(32))
    account = Account.from_key(private_key)
    wallet = HotWallet(account)
    wallet.sync_nonce(web3_source)

    user_1_source = web3_source.eth.accounts[1]
    tx_hash = web3_source.eth.send_transaction({"to": wallet.address, "from": user_1_source, "value": 15 * 10**18})
    assert_transaction_success_with_explanation(web3_source, tx_hash)
    tx_hash = usdc_source.functions.transfer(wallet.address, 5_000 * 10**6).transact({"from": deployer_source})
    assert_transaction_success_with_explanation(web3_source, tx_hash)

    user_1_dest = web3_dest.eth.accounts[1]
    tx_hash = web3_dest.eth.send_transaction({"to": wallet.address, "from": user_1_dest, "value": 15 * 10**18})
    assert_transaction_success_with_explanation(web3_dest, tx_hash)
    tx_hash = usdc_dest.functions.transfer(wallet.address, 5_000 * 10**6).transact({"from": deployer_dest})
    assert_transaction_success_with_explanation(web3_dest, tx_hash)

    return wallet


@pytest.fixture()
def strategy_file() -> Path:
    return Path(os.path.dirname(__file__)) / "../../strategies/test_only/cctp_bridge_strategy.py"


@pytest.fixture()
def state_file() -> Path:
    return Path(tempfile.mkdtemp()) / "test-correct-accounts-cctp-failures.json"


@pytest.fixture()
def environment(
    anvil_source: AnvilLaunch,
    anvil_dest: AnvilLaunch,
    hot_wallet: HotWallet,
    state_file: Path,
    strategy_file: Path,
    usdc_source: Contract,
    usdc_dest: Contract,
) -> dict:
    return {
        "EXECUTOR_ID": "test_correct_accounts_cctp_failures",
        "STRATEGY_FILE": strategy_file.as_posix(),
        "PRIVATE_KEY": hexbytes_to_hex_str(hot_wallet.account.key),
        "JSON_RPC_ARBITRUM": anvil_source.json_rpc_url,
        "JSON_RPC_BASE": anvil_dest.json_rpc_url,
        "STATE_FILE": state_file.as_posix(),
        "ASSET_MANAGEMENT_MODE": "hot_wallet",
        "UNIT_TESTING": "true",
        "TRADING_STRATEGY_API_KEY": os.environ.get("TRADING_STRATEGY_API_KEY", ""),
        "TEST_USDC_SOURCE_ADDRESS": usdc_source.address,
        "TEST_USDC_DEST_ADDRESS": usdc_dest.address,
        "PATH": os.environ.get("PATH", ""),
        "LOG_LEVEL": "disabled",
    }


# --- Tests ---


@pytest.mark.timeout(120)
def test_correct_accounts_settled_cash_return(
    environment: dict,
    state_file: Path,
    web3_source: Web3,
    web3_dest: Web3,
    deployer_source,
    deployer_dest,
    hot_wallet: HotWallet,
    usdc_source: Contract,
    usdc_dest: Contract,
):
    """Correct settled external cash movement without inferring bridge execution.

    1. Establish a settled two-chain cash ledger.
    2. Move tokens on both chains to represent an external return of capital.
    3. Correct balances and verify total equity and per-chain attribution.
    """
    cli = get_command(app)

    # 1. Establish a settled two-chain cash ledger.
    with patch.dict(os.environ, environment, clear=True):
        cli.main(args=["init"], standalone_mode=False)
    with patch.dict(os.environ, environment, clear=True):
        with pytest.raises(SystemExit) as e:
            cli.main(args=["correct-accounts"])
        assert e.value.code == 0

    # Verify baseline: reserve=5,000, bridge=5,000
    baseline = State.read_json_file(state_file)
    assert float(baseline.portfolio.get_default_reserve_position().quantity) == pytest.approx(5_000, rel=0.02)
    bridge_pos = list(baseline.portfolio.open_positions.values())[0]
    assert float(bridge_pos.get_quantity()) == pytest.approx(5_000, rel=0.02)

    # 2. Move tokens on both chains to represent an external return of capital.
    # External transfers are deliberate test actions, not evidence of a burn revert.
    tx_hash = usdc_source.functions.transfer(
        hot_wallet.address, 5_000 * 10**6,
    ).transact({"from": deployer_source})
    assert_transaction_success_with_explanation(web3_source, tx_hash)

    # Remove settled cash from the destination using test-only impersonation.
    web3_dest.provider.make_request("anvil_impersonateAccount", [hot_wallet.address])
    tx_hash = usdc_dest.functions.transfer(
        deployer_dest, 5_000 * 10**6,
    ).transact({"from": hot_wallet.address})
    assert_transaction_success_with_explanation(web3_dest, tx_hash)
    web3_dest.provider.make_request("anvil_stopImpersonatingAccount", [hot_wallet.address])

    # 3. Correct balances and verify total equity and per-chain attribution.
    with patch.dict(os.environ, environment, clear=True):
        with pytest.raises(SystemExit) as e:
            cli.main(args=["correct-accounts"])
        assert e.value.code == 0

    # Phase 4: Verify repaired state
    repaired = State.read_json_file(state_file)

    # Reserve should be 10,000 (all USDC back on source)
    reserve = repaired.portfolio.get_default_reserve_position()
    assert float(reserve.quantity) == pytest.approx(10_000, rel=0.02), \
        f"Reserve expected ~10,000, got {reserve.quantity}"

    # Bridge position should be closed (quantity went to 0)
    assert len(repaired.portfolio.open_positions) == 0, \
        f"Expected 0 open positions, got {len(repaired.portfolio.open_positions)}"
    assert len(repaired.portfolio.closed_positions) >= 1, \
        "Bridge position should have been closed"

    # Total equity = 10,000
    total = repaired.portfolio.calculate_total_equity()
    assert total == pytest.approx(10_000, rel=0.02), \
        f"Total equity expected ~10,000, got {total}"

    # Chain equity: all on source chain
    chain_equity = repaired.portfolio.calculate_total_equity_chain()
    assert chain_equity.get(ChainId.arbitrum, 0) == pytest.approx(10_000, rel=0.02)
    assert chain_equity.get(ChainId.base, 0) == pytest.approx(0, abs=1)
