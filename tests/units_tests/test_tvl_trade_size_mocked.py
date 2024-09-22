from decimal import Decimal

import pytest

from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.size_risk import SizingType
from tradeexecutor.strategy.pricing_model import FixedPricing
from tradeexecutor.strategy.tvl_size_risk import HistoricalUSDTVLSizeRiskModel
from tradeexecutor.testing.synthetic_ethereum_data import generate_random_ethereum_address
from tradeexecutor.testing.synthetic_exchange_data import generate_exchange
from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import Exchange


@pytest.fixture(scope="module")
def mock_chain_id() -> ChainId:
    """Mock a chai id."""
    return ChainId.ethereum


@pytest.fixture(scope="module")
def mock_exchange(mock_chain_id) -> Exchange:
    """Mock an exchange."""
    return generate_exchange(exchange_id=1, chain_id=mock_chain_id, address=generate_random_ethereum_address())


@pytest.fixture(scope="module")
def usdc() -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(ChainId.ethereum.value, generate_random_ethereum_address(), "USDC", 6, 1)


@pytest.fixture(scope="module")
def weth() -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(ChainId.ethereum.value, generate_random_ethereum_address(), "WETH", 18, 2)


@pytest.fixture(scope="module")
def weth_usdc(mock_exchange, usdc, weth) -> TradingPairIdentifier:
    """Mock some assets"""
    return TradingPairIdentifier(
        weth,
        usdc,
        generate_random_ethereum_address(),
        mock_exchange.address,
        internal_id=555,
        internal_exchange_id=mock_exchange.exchange_id,
        fee=0.0030
    )



@pytest.fixture(scope="module")
def pricing_model(mock_exchange, usdc, weth) -> FixedPricing:
    """Mock fixed price"""

    return FixedPricing(
        price=1500,
        lp_fee=0.01,
        tvl=150_000,
    )



def test_tvl_size_uncapped(
    pricing_model,
    weth_usdc
):
    """Do not limit trade sizes."""

    estimator = HistoricalUSDTVLSizeRiskModel(
        pricing_model,
    )

    estimate = estimator.get_acceptable_size_for_buy(
        None,
        weth_usdc,
        10_000,
    )

    assert estimate.asked_size == 10_000
    assert estimate.accepted_size == 10_000


def test_tvl_size_capped_buy_sell_hold(
    pricing_model,
    weth_usdc
):
    """Cap individual trade at 2% of TVL."""
    estimator = HistoricalUSDTVLSizeRiskModel(
        pricing_model,
        per_trade_cap=0.02,
        per_position_cap=0.05,
    )

    estimate = estimator.get_acceptable_size_for_buy(
        None,
        weth_usdc,
        10_000,
    )
    assert estimate.type == SizingType.buy
    assert estimate.asked_size == 10_000
    assert estimate.accepted_size == 3_000

    estimate = estimator.get_acceptable_size_for_sell(
        None,
        weth_usdc,
        Decimal(3),  # 1 ETH = 1500
    )
    assert estimate.type == SizingType.sell
    assert estimate.asked_size == 4500
    assert estimate.accepted_size == 3000
    assert estimate.asked_quantity == 3
    assert estimate.accepted_quantity == 2

    estimate = estimator.get_acceptable_size_for_position(
        None,
        weth_usdc,
        10_000,
    )
    assert estimate.type == SizingType.hold
    assert estimate.asked_size == 10_000
    assert estimate.accepted_size == 3_000