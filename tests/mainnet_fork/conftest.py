import os

import pytest
from eth_defi.provider.anvil import fork_network_anvil
from eth_defi.chain import install_chain_middleware
from eth_defi.abi import get_deployed_contract
from eth_typing import HexAddress, HexStr
from web3 import Web3, HTTPProvider
from web3.contract import Contract

from eth_defi.token import USDCE_WHALE
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.cli.log import setup_pytest_logging
from tradeexecutor.testing.pytest_helpers import is_failed_test


@pytest.fixture(scope="module")
def logger(request):
    """Setup test logger."""
    return setup_pytest_logging(request)


@pytest.fixture()
def large_usdc_holder() -> HexAddress:
    """A random account picked from Polygon chain that holds a lot of usdc.

    This account is unlocked on Ganache, so you have access to good usdc stash.

    `To find large holder accounts, use polygonscan <https://polygonscan.com/token/0x2791bca1f2de4661ed88a30c99a7a9449aa84174#balances>`_.
    """
    return USDCE_WHALE[137]


@pytest.fixture()
def anvil_polygon_chain_fork(request, logger, large_usdc_holder) -> str:
    """Create a testable fork of live polygon chain.

    :return: JSON-RPC URL for Web3
    """

    mainnet_rpc = os.environ["JSON_RPC_POLYGON"]

    launch = fork_network_anvil(mainnet_rpc, unlocked_addresses=[large_usdc_holder])
    try:
        yield launch.json_rpc_url
    finally:
        verbose_anvil_exit = is_failed_test(request)
        stdout, stderr = launch.close()

        # Disabled for now as it causes too much noise in the output
        # if verbose_anvil_exit:
        #    print(f"Anvil stdout:\n{stdout.decode('utf-8')}")
        #    print(f"Anvil stderr:\n{stderr.decode('utf-8')}")
        pass


@pytest.fixture
def web3(anvil_polygon_chain_fork: str):
    """Set up a local unit testing blockchain."""
    # https://web3py.readthedocs.io/en/stable/examples.html#contract-unit-tests-in-python
    web3 = Web3(HTTPProvider(anvil_polygon_chain_fork, request_kwargs={"timeout": 5}))
    install_chain_middleware(web3)
    # web3 = create_multi_provider_web3(anvil_polygon_chain_fork)
    return web3


@pytest.fixture
def chain_id(web3) -> int:
    """The test chain id (67)."""
    return web3.eth.chain_id


@pytest.fixture
def usdc_token(web3) -> Contract:
    """usdc with $4B supply."""
    # https://polygonscan.com/address/0x2791bca1f2de4661ed88a30c99a7a9449aa84174
    token = get_deployed_contract(
        web3, "ERC20MockDecimals.json", "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
    )
    return token


@pytest.fixture
def eth_token(web3) -> Contract:
    """eth token."""
    # https://polygonscan.com//address/0x7ceb23fd6bc0add59e62ac25578270cff1b9f619
    # https://tradingstrategy.ai/trading-view/polygon/tokens/0x7ceb23fd6bc0add59e62ac25578270cff1b9f619
    token = get_deployed_contract(
        web3, "ERC20MockDecimals.json", "0x7ceB23fD6bC0adD59E62ac25578270cFf1b9f619"
    )
    return token


@pytest.fixture()
def usdc_asset(usdc_token, chain_id) -> AssetIdentifier:
    return AssetIdentifier(
        chain_id,
        usdc_token.address,
        usdc_token.functions.symbol().call(),
        usdc_token.functions.decimals().call(),
    )


@pytest.fixture
def matic_asset(wmatic_token, chain_id) -> AssetIdentifier:
    return AssetIdentifier(
        chain_id,
        wmatic_token.address,
        wmatic_token.functions.symbol().call(),
        wmatic_token.functions.decimals().call(),
    )


@pytest.fixture
def eth_asset(eth_token, chain_id) -> AssetIdentifier:
    return AssetIdentifier(
        chain_id,
        eth_token.address,
        eth_token.functions.symbol().call(),
        eth_token.functions.decimals().call(),
    )