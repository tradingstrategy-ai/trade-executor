"""Unit tests for GMX exchange account pair creation and metadata."""

from types import SimpleNamespace

import pytest

from eth_defi.gmx.contracts import get_contract_addresses
from eth_defi.token import USDC_NATIVE_TOKEN

from tradeexecutor.exchange_account.gmx import create_gmx_exchange_account_pair, has_gmx_exchange_account_pairs
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier, TradingPairKind

#: Arbitrum mainnet chain ID
ARBITRUM_CHAIN_ID = 42161


@pytest.fixture()
def usdc() -> AssetIdentifier:
    return AssetIdentifier(
        chain_id=ARBITRUM_CHAIN_ID,
        address=USDC_NATIVE_TOKEN[ARBITRUM_CHAIN_ID],
        token_symbol="USDC",
        decimals=6,
    )


def test_create_gmx_exchange_account_pair(usdc):
    """Verify pair fields, kind, and is_exchange_account()."""
    pair = create_gmx_exchange_account_pair(quote=usdc)

    exchange_router = get_contract_addresses("arbitrum").exchangerouter
    assert pair.kind == TradingPairKind.exchange_account
    assert pair.is_exchange_account()
    assert pair.base.token_symbol == "GMX-ACCOUNT"
    assert pair.base.decimals == 6
    assert pair.base.chain_id == ARBITRUM_CHAIN_ID
    assert pair.base.address == exchange_router.lower()  # AssetIdentifier lowercases
    assert pair.quote.token_symbol == "USDC"
    assert pair.pool_address == exchange_router
    assert pair.exchange_address == exchange_router
    assert pair.exchange_name == "GMX"
    assert pair.fee == 0.0


def test_gmx_pair_protocol_detection(usdc):
    """Verify get_exchange_account_protocol() and other_data."""
    pair = create_gmx_exchange_account_pair(
        quote=usdc,
        is_testnet=True,
    )

    assert pair.get_exchange_account_protocol() == "gmx"

    config = pair.get_exchange_account_config()
    assert config["exchange_protocol"] == "gmx"
    assert config["exchange_is_testnet"] is True

    # Mainnet variant
    pair_mainnet = create_gmx_exchange_account_pair(
        quote=usdc,
        is_testnet=False,
    )
    assert pair_mainnet.get_exchange_account_config()["exchange_is_testnet"] is False


def test_has_gmx_exchange_account_pairs(usdc: AssetIdentifier):
    """Universe-driven GMX detection finds GMX exchange account pairs and nothing else.

    EthereumPairConfigurator uses this detection to auto-wire the GMX value
    functions without any environment variable, so the check must be positive
    exactly when the strategy universe trades GMX through an exchange account
    pair.

    1. Build a universe stub containing a GMX exchange account pair and verify detection is positive.
    2. Build a universe stub with only a spot pair and verify detection is negative.
    3. Verify a universe object without iterate_pairs() (non-trading stub) is negative.
    """
    # 1. Build a universe stub containing a GMX exchange account pair and verify detection is positive.
    gmx_pair = create_gmx_exchange_account_pair(quote=usdc)
    spot_pair = TradingPairIdentifier(
        base=AssetIdentifier(chain_id=ARBITRUM_CHAIN_ID, address="0x0000000000000000000000000000000000000005", token_symbol="WETH", decimals=18),
        quote=usdc,
        pool_address="0x0000000000000000000000000000000000000006",
        exchange_address="0x0000000000000000000000000000000000000007",
        internal_id=2,
        internal_exchange_id=2,
        fee=0.0005,
        kind=TradingPairKind.spot_market_hold,
        exchange_name="uniswap-v3",
    )
    gmx_universe = SimpleNamespace(iterate_pairs=lambda: [spot_pair, gmx_pair])
    assert has_gmx_exchange_account_pairs(gmx_universe) is True

    # 2. Build a universe stub with only a spot pair and verify detection is negative.
    spot_universe = SimpleNamespace(iterate_pairs=lambda: [spot_pair])
    assert has_gmx_exchange_account_pairs(spot_universe) is False

    # 3. Verify a universe object without iterate_pairs() (non-trading stub) is negative.
    assert has_gmx_exchange_account_pairs(SimpleNamespace(reserve_assets=[])) is False
