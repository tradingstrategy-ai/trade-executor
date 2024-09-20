import pytest

from tradeexecutor.backtest.backtest_price_impact import FixedCappedSizeBacktestPriceImpact
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
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


def test_fixed_price_impact_buy(weth_usdc):
    """Estimate a fixed price impact of 0.25%"""

    estimator = FixedCappedSizeBacktestPriceImpact(
        fixed_price_impact=0.25,
        capped_size=None,
    )

    estimate = estimator.get_acceptable_size_for_buy(
        None,
        weth_usdc,
        acceptable_price_impact=0.50,
        size=10_000,
    )

    assert estimate.estimated_price_impact == pytest.approx(0.25)
    assert estimate.asked_size == pytest.approx(10_000)
    assert estimate.accepted_size == pytest.approx(10_000)


def test_fixed_price_impact_buy_capped():
    pass
