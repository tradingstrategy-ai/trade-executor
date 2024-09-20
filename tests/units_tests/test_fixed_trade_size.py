import pytest

from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.strategy.fixed_size_risk import FixedSizeRiskModel
from tradeexecutor.strategy.pricing_model import FixedPricing
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

    return FixedPricing(1500, 0.01)



def test_fixed_price_size_uncapped(
    pricing_model,
    weth_usdc
):
    """Do not limit trade sizes."""

    estimator = FixedSizeRiskModel(
        pricing_model,
    )

    estimate = estimator.get_acceptable_size_for_buy(
        None,
        weth_usdc,
        10_000,
    )

    assert estimate.asked_size == 10_000
    assert estimate.accepted_size == 10_000


def test_fixed_price_impact_buy_capped(
    pricing_model,
    weth_usdc
):
    """Cap individual trade to a fixed size."""
    estimator = FixedSizeRiskModel(
        pricing_model,
        per_trade_cap=5000,
    )

    estimate = estimator.get_acceptable_size_for_buy(
        None,
        weth_usdc,
        10_000,
    )

    assert estimate.asked_size == 10_000
    assert estimate.accepted_size == 5_000
