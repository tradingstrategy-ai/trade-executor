"""Test Enzyme using end-to-end integration with guard smart contracts and Aave test trade."""
import json
import os
import secrets
from pathlib import Path
from unittest import mock

import pytest

from eth_account import Account
from eth_typing import HexAddress
import flaky
from hexbytes import HexBytes
from web3 import Web3, HTTPProvider

from eth_defi.provider.anvil import AnvilLaunch, launch_anvil
from eth_defi.chain import install_chain_middleware
from eth_defi.enzyme.vault import Vault
from eth_defi.hotwallet import HotWallet
from eth_defi.token import TokenDetails, fetch_erc20_details
from eth_defi.trace import assert_transaction_success_with_explanation

from tradeexecutor.cli.main import app
from tradeexecutor.monkeypatch.web3 import construct_sign_and_send_raw_middleware
from tradeexecutor.state.state import State


CI = os.environ.get("CI") == "true"


pytestmark = pytest.mark.skipif(not os.environ.get("JSON_RPC_POLYGON") or not os.environ.get("TRADING_STRATEGY_API_KEY"), reason="Set POLYGON_JSON_RPC and TRADING_STRATEGY_API_KEY environment variables to run this test")


@pytest.fixture()
def usdc_whale() -> HexAddress:
    """A random account picked from Polygon that holds a lot of USDC."""
    # https://polygonscan.com/token/0x2791bca1f2de4661ed88a30c99a7a9449aa84174#balances
    return HexAddress("0x72A53cDBBcc1b9efa39c834A540550e23463AAcB")


@pytest.fixture()
def anvil(usdc_whale) -> AnvilLaunch:
    """Launch Polygon fork."""
    rpc_url = os.environ["JSON_RPC_POLYGON"]

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
    details = fetch_erc20_details(web3, "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
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
    tx_hash = web3.eth.send_transaction({"to": wallet.address, "from": deployer, "value": 15 * 10**18})
    assert_transaction_success_with_explanation(web3, tx_hash)
    tx_hash = usdc.contract.functions.transfer(wallet.address, 500 * 10**6).transact({"from": usdc_whale})
    assert_transaction_success_with_explanation(web3, tx_hash)

    # Add to the local signer chain
    web3.middleware_onion.add(construct_sign_and_send_raw_middleware(account))

    return wallet


@pytest.fixture()
def strategy_file() -> Path:
    """Where do we load our strategy file."""
    return Path(os.path.dirname(__file__)) / ".." / ".." / "strategies" / "eth-breakout-atr-1h-aave.py"


@pytest.fixture()
def vault_record_file(tmp_path) -> Path:
    return Path(tmp_path) / "vault-info.json"


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
    persistent_test_cache_path,
    ) -> dict:
    """Passed to init and start commands as environment variables"""
    # Set up the configuration for the live trader
    environment = {
        "EXECUTOR_ID": "test_enzyme_guard_perform_test_trade_aave",
        "STRATEGY_FILE": strategy_file.as_posix(),
        "PRIVATE_KEY": hot_wallet.account.key.hex(),
        "JSON_RPC_ANVIL": anvil.json_rpc_url,
        "STATE_FILE": state_file.as_posix(),
        "ASSET_MANAGEMENT_MODE": "enzyme",
        "UNIT_TESTING": "true",
        "UNIT_TEST_FORCE_ANVIL": "true",  # check-wallet command legacy hack
        "LOG_LEVEL": "disabled",
        "CONFIRMATION_BLOCK_COUNT": "0",  # Needed for test backend, Anvil
        "TRADING_STRATEGY_API_KEY": os.environ["TRADING_STRATEGY_API_KEY"],
        "VAULT_RECORD_FILE": vault_record_file.as_posix(),
        "FUND_NAME": "Boogeyman I",
        "FUND_SYMBOL": "BOO1",
        # 0x625E7708f30cA75bfd92586e17077590C60eb4cD
        "WHITELISTED_ASSETS": "0x7ceb23fd6bc0add59e62ac25578270cff1b9f619 0x625E7708f30cA75bfd92586e17077590C60eb4cD",  # WETH, aPolUSDC
        "DENOMINATION_ASSET": "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",   # USDC.e
        "TERMS_OF_SERVICE_ADDRESS": "0xbe1418df0bAd87577de1A41385F19c6e77312780",  # Deployed earlier, but not used
        "OWNER_ADDRESS": "0x238B0435F69355e623d99363d58F7ba49C408491",  # ProtoDAO multisig
        "PATH": os.environ["PATH"],  # Needs forge
        "ONE_DELTA": "true",
        "CACHE_PATH": persistent_test_cache_path,
    }
    return environment


@pytest.mark.skipif(CI, reason="Too flaky to run on CI")
def test_enzyme_guard_perform_test_trade_aave(
    environment: dict,
    web3: Web3,
    state_file: Path,
    usdc: TokenDetails,
    hot_wallet: HotWallet,
    vault_record_file: Path,
):
    """Perform a test trade on Enzyme vault via CLI with Aave credit position.

    """

    # Deploy a new vault on the
    with mock.patch.dict('os.environ', environment, clear=True):
        app(["enzyme-deploy-vault"], standalone_mode=False)

    vault_info = json.load(vault_record_file.open("rt"))

    assert "guard" in vault_info
    assert "terms_of_service" in vault_info
    assert "fund_name" in vault_info
    assert "fund_symbol" in vault_info

    # Deposit some USDC to the vault to start
    deposit_amount = 500 * 10**6
    vault = Vault.fetch(web3, vault_info["vault"])
    tx_hash = usdc.contract.functions.approve(vault_info["comptroller"], deposit_amount).transact({"from": hot_wallet.address})
    assert_transaction_success_with_explanation(web3, tx_hash)
    tx_hash = vault.comptroller.functions.buyShares(deposit_amount, 1).transact({"from": hot_wallet.address})
    assert_transaction_success_with_explanation(web3, tx_hash)
    assert usdc.contract.functions.balanceOf(vault.address).call() == deposit_amount

    # Update the environment for the future commands with vault parameters from out deployment
    github_hack = 10  # For some reason, CI gets the start block wrong, could be Anvil issue?
    environment.update({
        "VAULT_ADDRESS": vault_info["vault"],
        "VAULT_DEPLOYMENT_BLOCK_NUMBER": str(vault_info["block_number"] - github_hack),
        "VAULT_ADAPTER_ADDRESS": vault_info["generic_adapter"],
        "VAULT_PAYMENT_FORWARDER_ADDRESS": vault_info["usdc_payment_forwarder"],
    })

    with mock.patch.dict('os.environ', environment, clear=True):
        app(["init"], standalone_mode=False)

    with mock.patch.dict('os.environ', environment, clear=True):
        app(["check-wallet"], standalone_mode=False)

    with mock.patch.dict('os.environ', environment, clear=True):
        app(["perform-test-trade", "--lending-reserve=(polygon,aave-v3,USDC.e)"], standalone_mode=False)

    with mock.patch.dict('os.environ', environment, clear=True):
        app(["perform-test-trade", "--all-pairs"], standalone_mode=False)

    # Check the resulting state and see we made some trade for trading fee losses
    with state_file.open("rt") as inp:
        state: State = State.from_json(inp.read())
        assert len(list(state.portfolio.get_all_trades())) == 4  # buy ETH, sell ETH, supply aPolUSDC, unsupply aPolUSDC
        reserve_value = state.portfolio.get_default_reserve_position().get_value()
        assert reserve_value < 500

