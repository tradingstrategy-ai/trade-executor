"""Hot wallet nonce refresh tests.

The executor keeps the hot wallet nonce in an in-memory counter that is read
from the chain only once at startup. If the same private key is used outside the
executor process, for example a manual Safe multisig transaction or an operator
running a CLI command, the counter falls behind the chain and every transaction
the executor signs afterwards is permanently rejected with ``nonce too low``.

:py:func:`~tradeexecutor.ethereum.nonce.refresh_hot_wallet_nonces` is called at
the start of every scheduled live task to correct this before anything is signed.
"""

import pytest
from eth_defi.hotwallet import HotWallet
from eth_defi.provider.anvil import AnvilLaunch, launch_anvil
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.trace import assert_transaction_success_with_explanation
from tradingstrategy.chain import ChainId
from web3 import Web3

from tradeexecutor.ethereum.nonce import refresh_hot_wallet_nonces
from tradeexecutor.ethereum.web3config import Web3Config

ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"


@pytest.fixture()
def anvil() -> AnvilLaunch:
    """Local test chain."""
    anvil = launch_anvil()
    try:
        yield anvil
    finally:
        anvil.close()


@pytest.fixture()
def web3(anvil: AnvilLaunch) -> Web3:
    """Web3 connection to the local test chain."""
    return create_multi_provider_web3(anvil.json_rpc_url)


@pytest.fixture()
def hot_wallet(web3: Web3) -> HotWallet:
    """Funded hot wallet with an unsynced nonce counter."""
    return HotWallet.create_for_testing(web3)


@pytest.fixture()
def web3config(web3: Web3) -> Web3Config:
    """Web3Config holding the single test chain connection."""
    config = Web3Config()
    config.connections[ChainId.anvil] = web3
    config.set_default_chain(ChainId.anvil)
    return config


def _broadcast_outside_the_counter(web3: Web3, hot_wallet: HotWallet):
    """Spend a nonce without touching the hot wallet's in-memory counter.

    Simulates the private key being used by another process, a manual Safe
    transaction or a CLI command while the executor is running.
    """
    tx = {
        "from": hot_wallet.address,
        "to": ZERO_ADDRESS,
        "value": 1,
        "gas": 100_000,
        "gasPrice": web3.eth.gas_price,
        "chainId": web3.eth.chain_id,
        # Read straight from the chain, bypassing HotWallet.allocate_nonce()
        "nonce": web3.eth.get_transaction_count(hot_wallet.address),
    }
    signed = hot_wallet.account.sign_transaction(tx)
    tx_hash = web3.eth.send_raw_transaction(signed.raw_transaction)
    assert_transaction_success_with_explanation(web3, tx_hash)


def test_refresh_hot_wallet_nonces_adopts_external_nonce(
    web3: Web3,
    web3config: Web3Config,
    hot_wallet: HotWallet,
):
    """Refreshing adopts a nonce burned outside the trade executor.

    This is the failure that took the hyper-ai executor down: a manual Safe
    settlement spent the next nonce, the executor kept its stale counter and
    every NAV update it signed afterwards was rejected as ``nonce too low``.

    1. Initialise the counter from the chain through the refresh helper.
    2. Spend a nonce outside the counter, as an external process would.
    3. Refresh again and confirm the counter adopted the on-chain nonce.
    """

    # 1. Initialise the counter from the chain through the refresh helper.
    # create_for_testing() has already synced it, so clear it first to cover
    # the unsynced startup path as well.
    hot_wallet.current_nonce = None
    result = refresh_hot_wallet_nonces(web3config, {ChainId.anvil: hot_wallet}, context="unit_test")
    assert result == {ChainId.anvil: 0}
    assert hot_wallet.current_nonce == 0

    # 2. Spend a nonce outside the counter, as an external process would
    _broadcast_outside_the_counter(web3, hot_wallet)
    assert hot_wallet.current_nonce == 0, "External use must not touch the in-memory counter"
    assert web3.eth.get_transaction_count(hot_wallet.address) == 1

    # 3. Refresh again and confirm the counter adopted the on-chain nonce
    result = refresh_hot_wallet_nonces(web3config, {ChainId.anvil: hot_wallet}, context="unit_test")
    assert result == {ChainId.anvil: 1}
    assert hot_wallet.current_nonce == 1


def test_refresh_hot_wallet_nonces_never_rewinds_counter(
    web3: Web3,
    web3config: Web3Config,
    hot_wallet: HotWallet,
):
    """Refreshing does not rewind a counter that is ahead of the chain.

    A local counter ahead of the chain is the normal state while our own
    transaction is broadcast but not yet mined. Rewinding it would hand the same
    nonce out twice and cause a collision, so the counter must be left alone.

    1. Initialise the counter from the chain.
    2. Allocate a nonce locally without broadcasting anything.
    3. Refresh and confirm the counter was not lowered back to the chain value.
    """

    # 1. Initialise the counter from the chain
    refresh_hot_wallet_nonces(web3config, {ChainId.anvil: hot_wallet}, context="unit_test")
    assert hot_wallet.current_nonce == 0

    # 2. Allocate a nonce locally without broadcasting anything
    allocated = hot_wallet.allocate_nonce()
    assert allocated == 0
    assert hot_wallet.current_nonce == 1
    assert web3.eth.get_transaction_count(hot_wallet.address) == 0

    # 3. Refresh and confirm the counter was not lowered back to the chain value
    result = refresh_hot_wallet_nonces(web3config, {ChainId.anvil: hot_wallet}, context="unit_test")
    assert result == {ChainId.anvil: 1}
    assert hot_wallet.current_nonce == 1
