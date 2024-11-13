"""Sync reserves from on-chain data."""

import datetime
from decimal import Decimal
from typing import List

import pytest
from eth_typing import HexAddress
from web3 import EthereumTesterProvider, Web3
from web3.contract import Contract

from eth_defi.token import create_token
from tradeexecutor.ethereum.wallet import sync_reserves
from tradeexecutor.state.state import State
from tradeexecutor.testing.dummy_wallet import  apply_sync_events
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.identifier import AssetIdentifier


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
def deployer(web3) -> str:
    """Deploy account.

    Do some account allocation for tests.
    """
    return web3.eth.accounts[0]


@pytest.fixture()
def hot_wallet(web3) -> HexAddress:
    """User account.

    Do some account allocation for tests.
    """
    return web3.eth.accounts[1]

@pytest.fixture
def usdc_token(web3, deployer: HexAddress) -> Contract:
    """Mock some assets"""
    token = create_token(web3, deployer, "Fake USDC coin", "USDC", 100_000 * 10**18, 6)
    return token


@pytest.fixture
def usdc(usdc_token, web3) -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(web3.eth.chain_id, usdc_token.address, "USDC", 6)


@pytest.fixture
def weth(web3) -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(web.eth.chain_id, "0x1", "WETH", 18)


@pytest.fixture
def start_ts() -> datetime.datetime:
    """Timestamp of action started"""
    return datetime.datetime(2022, 1, 1, tzinfo=None)


@pytest.fixture
def supported_reserves(usdc) -> List[AssetIdentifier]:
    """Timestamp of action started"""
    return [usdc]


def test_update_reserves_empty(web3, usdc, start_ts, hot_wallet: HexAddress, supported_reserves):
    """Syncing empty reserves does not cause errors."""
    events = sync_reserves(web3, start_ts, hot_wallet, [], supported_reserves)
    assert len(events) == 0


def test_update_reserves_one_deposit(web3, usdc_token, deployer, start_ts, hot_wallet: HexAddress, supported_reserves):
    """Sync reserves from one deposit."""

    # Deposit 500 usd
    usdc_token.functions.transfer(hot_wallet, 500 * 10**6).transact({"from": deployer})
    events = sync_reserves(web3, start_ts, hot_wallet, [], supported_reserves)
    assert len(events) == 1

    evt = events[0]
    assert evt.updated_at == start_ts
    assert evt.new_balance == 500
    assert evt.past_balance == 0


def test_update_reserves_no_change(web3, usdc_token, deployer, start_ts, hot_wallet: HexAddress, supported_reserves):
    """Do not generate deposit events if there has not been changes."""

    state = State()
    portfolio = state.portfolio

    # Deposit 500 usd
    usdc_token.functions.transfer(hot_wallet, 500 * 10**6).transact({"from": deployer})
    events = sync_reserves(web3, start_ts, hot_wallet, [], supported_reserves)
    assert len(events) == 1

    apply_sync_events(state, events)

    events = sync_reserves(web3, start_ts, hot_wallet, portfolio.reserves.values(), supported_reserves)
    assert len(events) == 0


def test_update_reserves_twice(web3, usdc_token, deployer, start_ts, hot_wallet: HexAddress, supported_reserves):
    """Sync reserves from one deposit."""

    state = State()
    portfolio = state.portfolio

    # Deposit 500 usd
    usdc_token.functions.transfer(hot_wallet, 500 * 10**6).transact({"from": deployer})
    events = sync_reserves(web3, start_ts, hot_wallet, [], supported_reserves)
    assert len(events) == 1

    apply_sync_events(state, events)

    address = f"131277322940537-{usdc_token.address.lower()}"

    assert portfolio.reserves[address].quantity == Decimal(500)
    assert portfolio.reserves[address].reserve_token_price == 1.0

    # Deposit 200 usd more
    usdc_token.functions.transfer(hot_wallet, 200 * 10**6).transact({"from": deployer})
    events = sync_reserves(web3, start_ts, hot_wallet, portfolio.reserves.values(), supported_reserves)
    assert len(events) == 1

    evt = events[0]
    assert evt.updated_at == start_ts
    assert evt.new_balance == 700
    assert evt.past_balance == 500

    apply_sync_events(state, events)

    assert portfolio.reserves[address].quantity == Decimal(700)
