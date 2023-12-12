"""Gas level warning tests."""

import secrets
from decimal import Decimal

import pytest
from eth_account import Account
from eth_defi.provider.anvil import launch_anvil, AnvilLaunch
from eth_defi.chain import install_chain_middleware
from eth_defi.hotwallet import HotWallet
from eth_defi.trace import assert_transaction_success_with_explanation
from eth_typing import HexAddress
from hexbytes import HexBytes

from web3 import Web3, HTTPProvider

from tradeexecutor.ethereum.wallet import perform_gas_level_checks
from tradeexecutor.strategy.run_state import RunState


@pytest.fixture()
def anvil() -> AnvilLaunch:

    # London hardfork will enable EIP-1559 style gas fees
    anvil = launch_anvil(
        hardfork="london",
        gas_limit=15_000_000,  # Max 5M gas per block, or per transaction in test automining
    )
    try:
        # Make the initial snapshot ("zero state") to which we revert between tests
        # web3 = Web3(HTTPProvider(anvil.json_rpc_url))
        # snapshot_id = make_anvil_custom_rpc_request(web3, "evm_snapshot")
        # assert snapshot_id == "0x0"
        yield anvil
    finally:
        anvil.close()


@pytest.fixture()
def web3(anvil: AnvilLaunch) -> Web3:
    """Set up the Anvil Web3 connection.

    Also perform the Anvil state reset for each test.
    """
    web3 = Web3(HTTPProvider(anvil.json_rpc_url, request_kwargs={"timeout": 2}))
    install_chain_middleware(web3)
    return web3


@pytest.fixture()
def deployer(web3) -> HexAddress:
    """Deployer account.

    - This account will deploy all smart contracts

    - Starts with 10,000 ETH
    """
    return web3.eth.accounts[0]


@pytest.fixture()
def user_1(web3) -> HexAddress:
    """User account.

    Do some account allocation for tests.
    """
    return web3.eth.accounts[1]


@pytest.fixture
def hot_wallet(web3, deployer, user_1) -> HotWallet:
    """Create hot wallet for the signing tests.

    Top is up with some gas money and 500 USDC.
    """
    private_key = HexBytes(secrets.token_bytes(32))
    account = Account.from_key(private_key)
    wallet = HotWallet(account)
    wallet.sync_nonce(web3)
    tx_hash = web3.eth.send_transaction({"to": wallet.address, "from": user_1, "value": 15 * 10**18})
    assert_transaction_success_with_explanation(web3, tx_hash)
    return wallet


def test_hot_wallet_gas_level_warning_set(
    web3: Web3,
    hot_wallet: HotWallet,
    deployer: HexAddress,
):
    """Hot wallet gas warning is set."""

    assert hot_wallet.get_native_currency_balance(web3) == Decimal(15)

    # Warning level 1000 ETH
    run_state = RunState()
    run_state.hot_wallet_gas_warning_level = 1000

    flag = perform_gas_level_checks(
        web3,
        run_state,
        hot_wallet,
    )

    assert flag
    assert run_state.hot_wallet_gas_warning_message == f"Hot wallet address 0x084258cCef54100E18277ce188a665bd53AD27D1, gas is 15.0 tokens, warning level is 1000"


def test_hot_wallet_gas_level_warning_not_set(
    web3: Web3,
    hot_wallet: HotWallet,
    deployer: HexAddress,
):
    """Hot wallet gas warning is set."""

    # Warning level 0.1 ETH
    run_state = RunState()
    run_state.hot_wallet_gas_warning_level = Decimal(0.1)

    flag = perform_gas_level_checks(
        web3,
        run_state,
        hot_wallet,
    )

    assert not flag
    assert run_state.hot_wallet_gas_warning_message is None
