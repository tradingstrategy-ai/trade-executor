"""Test Lagoon using end-to-end integration with guard smart contracts and Aave test trade."""
import json
import os
import secrets
from decimal import Decimal
from pathlib import Path
from unittest import mock

import pytest

from eth_account import Account
from eth_typing import HexAddress
import flaky
from hexbytes import HexBytes
from web3 import Web3, HTTPProvider

from eth_defi.lagoon.vault import LagoonVault
from eth_defi.provider.anvil import AnvilLaunch, launch_anvil
from eth_defi.chain import install_chain_middleware
from eth_defi.hotwallet import HotWallet
from eth_defi.token import TokenDetails, fetch_erc20_details
from eth_defi.trace import assert_transaction_success_with_explanation
from eth_defi.vault.base import VaultSpec

from tradeexecutor.cli.main import app
from tradeexecutor.monkeypatch.web3 import construct_sign_and_send_raw_middleware
from tradeexecutor.state.state import State
from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client

pytestmark = pytest.mark.skipif(not os.environ.get("JSON_RPC_BASE") or not os.environ.get("TRADING_STRATEGY_API_KEY"), reason="Set JSON_RPC_BASE and TRADING_STRATEGY_API_KEY environment variables to run this test")


@pytest.fixture()
def usdc_whale() -> HexAddress:
    """A random account picked from Polygon that holds a lot of USDC."""
    # https://basescan.org/token/0x833589fcd6edb6e08f4c7c32d4f71b54bda02913#balances
    return HexAddress("0x0B0A5886664376F59C351ba3f598C8A8B4D0A6f3")


@pytest.fixture()
def anvil(usdc_whale) -> AnvilLaunch:
    """Launch Polygon fork."""
    rpc_url = os.environ["JSON_RPC_BASE"]
    anvil = launch_anvil(
        fork_url=rpc_url,
        unlocked_addresses=[usdc_whale],
    )
    try:
        yield anvil
    finally:
        anvil.close()


@pytest.fixture()
def web3(anvil) -> Web3:
    """Set up the Anvil Web3 connection.

    Also perform the Anvil state reset for each test.
    """
    web3 = Web3(HTTPProvider(anvil.json_rpc_url, request_kwargs={"timeout": 10}))
    web3.middleware_onion.clear()
    install_chain_middleware(web3)
    return web3


@pytest.fixture()
def deployer(web3) -> HexAddress:
    """Deployer account.

    - This account will deploy all smart contracts

    - Starts with 10,000 ETH
    """
    return web3.eth.accounts[0]


@pytest.fixture
def usdc(web3) -> TokenDetails:
    details = fetch_erc20_details(web3, "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913")
    return details


@pytest.fixture
def hot_wallet(
    web3,
    deployer,
    usdc: TokenDetails,
    usdc_whale,
) -> HotWallet:
    """Create hot wallet for the signing tests.

    Top is up with some gas money and 500 USDC.
    """
    private_key = HexBytes(secrets.token_bytes(32))
    account = Account.from_key(private_key)
    wallet = HotWallet(account)
    wallet.sync_nonce(web3)
    tx_hash = web3.eth.send_transaction({"to": wallet.address, "from": deployer, "value": 15 * 10 ** 18})
    assert_transaction_success_with_explanation(web3, tx_hash)

    tx_hash = web3.eth.send_transaction({"to": usdc_whale, "from": deployer, "value": 15 * 10 ** 18})
    assert_transaction_success_with_explanation(web3, tx_hash)

    tx_hash = usdc.contract.functions.transfer(wallet.address, 500 * 10 ** 6).transact({"from": usdc_whale})
    assert_transaction_success_with_explanation(web3, tx_hash)

    # Add to the local signer chain
    web3.middleware_onion.add(construct_sign_and_send_raw_middleware(account))

    return wallet


@pytest.fixture()
def strategy_file() -> Path:
    """Where do we load our strategy file."""
    return Path(os.path.dirname(__file__)) / ".." / ".." / "strategies" / "test_only" / "base-ath.py"


@pytest.fixture()
def vault_record_file(tmp_path) -> Path:
    return Path(tmp_path) / "vault-info.txt"


@pytest.fixture()
def state_file(tmp_path) -> Path:
    """The state file for the tests.

    Always start with an empty file.
    """
    path = Path(f"/{tmp_path}/test_enzyme_end_to_end_aave.json")
    if path.exists():
        os.remove(path)
    return path


@pytest.fixture()
def environment(
    anvil: AnvilLaunch,
    hot_wallet: HotWallet,
    state_file: Path,
    strategy_file: Path,
    vault_record_file: Path,
    persistent_test_client: Client,
) -> dict:
    """Passed to init and start commands as environment variables"""
    # Set up the configuration for the live trader
    unit_test_cache_path = persistent_test_client.transport.cache_path
    environment = {
        "CACHE_PATH": unit_test_cache_path,
        "EXECUTOR_ID": "test_lagoon_guard_perform_test_aave_v3",
        "STRATEGY_FILE": strategy_file.as_posix(),
        "PRIVATE_KEY": hot_wallet.account.key.hex(),
        "JSON_RPC_ANVIL": anvil.json_rpc_url,
        "STATE_FILE": state_file.as_posix(),
        "ASSET_MANAGEMENT_MODE": "lagoon",
        "UNIT_TESTING": "true",
        "UNIT_TEST_FORCE_ANVIL": "true",  # check-wallet command legacy hack
        "LOG_LEVEL": "disabled",
        "CONFIRMATION_BLOCK_COUNT": "0",  # Needed for test backend, Anvil
        "TRADING_STRATEGY_API_KEY": os.environ["TRADING_STRATEGY_API_KEY"],
        "VAULT_RECORD_FILE": vault_record_file.as_posix(),
        "FUND_NAME": "Boogeyman I",
        "FUND_SYMBOL": "BOO1",
        "DENOMINATION_ASSET": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
        # 0x625E7708f30cA75bfd92586e17077590C60eb4cD
        "PERFORMANCE_FEE": "2000",
        "MANAGEMENT_FEE": "0",
        "PATH": os.environ["PATH"],  # Needs forge
        "ANY_ASSET": "true",
        "AAVE": "true",
        "UNISWAP_V2": "true",
        "UNISWAP_V3": "true",
        "MULTISIG_OWNERS": "0xa7208b5c92d4862b3f11c0047b57a00Dc304c0f8, 0xbD35322AA7c7842bfE36a8CF49d0F063bf83a100, 0x05835597cAf9e04331dfe1f62C2Ec0C2aDc0d4a2, 0x5C46ab9e42824c51b55DcD3Cf5876f1132F9FbA9",
    }
    return environment


def test_lagoon_guard_perform_test_trade_aave_uniswap_v2(
    environment: dict,
    web3: Web3,
    state_file: Path,
    usdc: TokenDetails,
    hot_wallet: HotWallet,
    vault_record_file: Path,
):
    """Perform a test trades vault via CLI with Uniswap v2 and sAave credit position.

    """

    # Deploy a new vault on the
    with mock.patch.dict('os.environ', environment, clear=True):
        app(["lagoon-deploy-vault"], standalone_mode=False)

    vault_info = json.load(vault_record_file.with_suffix(".json").open("rt"))
    assert "Safe" in vault_info
    assert "Vault" in vault_info
    assert "Block number" in vault_info

    # Deposit some USDC to the vault to start
    usdc_amount = Decimal(500)
    usdc_amount_raw = usdc.convert_to_raw(usdc_amount)
    vault = LagoonVault(web3, VaultSpec(ChainId.base.value, vault_info["Vault"]))
    depositor = hot_wallet.address
    tx_hash = usdc.approve(vault.address, usdc_amount).transact({"from": depositor})
    assert_transaction_success_with_explanation(web3, tx_hash)
    deposit_func = vault.request_deposit(depositor, usdc_amount_raw)
    tx_hash = deposit_func.transact({"from": depositor})
    assert_transaction_success_with_explanation(web3, tx_hash)
    assert usdc.fetch_balance_of(vault.silo_address) == pytest.approx(Decimal(500))

    # Update the environment for the future commands with vault parameters from out deployment
    environment.update({
        "VAULT_ADDRESS": vault_info["Vault"],
        "VAULT_ADAPTER_ADDRESS": vault_info["Trading strategy module"],
    })

    with mock.patch.dict('os.environ', environment, clear=True):
        app(["init"], standalone_mode=False)

    # Test trade KEYCAT
    with mock.patch.dict('os.environ', environment, clear=True):
        app(["perform-test-trade", "--pair", "(base, uniswap-v2, WETH, USDC, 0.003)"], standalone_mode=False)

    # Test trade aave
    with mock.patch.dict('os.environ', environment, clear=True):
        app(["perform-test-trade", "--lending-reserve=(base,aave-v3,USDC)"], standalone_mode=False)

    # Check the resulting state and see we made some trade for trading fee losses
    with state_file.open("rt") as inp:
        state: State = State.from_json(inp.read())
        assert len(list(state.portfolio.frozen_positions)) == 0  # all trades succeed
        trades = list(state.portfolio.get_all_trades())
        if len(trades) != 4:
            for t in trades:
                print(t)
            assert len(trades) == 4  # buy ETH, sell ETH, supply aPolUSDC, unsupply aPolUSDC
        reserve_value = state.portfolio.get_default_reserve_position().get_value()
        assert reserve_value < 500
