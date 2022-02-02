"""Test token portfolio updates."""
import pytest
from web3 import Web3

from tradeexecutor.utils.dataclass import EthereumAddress



@pytest.fixture
def tester_provider():
    # https://web3py.readthedocs.io/en/stable/examples.html#contract-unit-tests-in-python
    return EthereumTesterProvider()


@pytest.fixture
def eth_tester(tester_provider):
    # https://web3py.readthedocs.io/en/stable/examples.html#contract-unit-tests-in-python
    return tester_provider.ethereum_tester


@pytest.fixture
def web3(tester_provider):
    """Set up a local unit testing blockchain."""
    # https://web3py.readthedocs.io/en/stable/examples.html#contract-unit-tests-in-python
    return Web3(tester_provider)


@pytest.fixture()
def deployer() -> EthereumAddress:
    """Deploy account."""


@pytest.fixture()
def user_1() -> EthereumAddress:
    """A normal user account."""


@pytest.fixture()
def user_2() -> EthereumAddress:
    """A normal user account."""


@pytest.fixture()
def weth(deployer) -> EthereumAddress:
    """A simulated Weth token."""


@pytest.fixture()
def usdc(deployer) -> EthereumAddress:
    """A simulated USDC token."""
