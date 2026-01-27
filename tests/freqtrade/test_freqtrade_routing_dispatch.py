"""Test routing dispatch for Freqtrade pairs.

Tests that Freqtrade pairs are correctly routed through the generic routing system.
"""

import os
from unittest.mock import Mock, patch

import pytest
from web3 import Web3

from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier, TradingPairKind
from tradeexecutor.strategy.freqtrade.universe import create_freqtrade_pair
from tradeexecutor.strategy.generic.default_protocols import default_match_router, default_supported_routers
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.ethereum.ethereum_protocol_adapters import EthereumPairConfigurator, create_freqtrade_adapter
from tradingstrategy.chain import ChainId


@pytest.fixture
def usdc_asset():
    """USDC asset for testing."""
    return AssetIdentifier(
        chain_id=ChainId.polygon.value,
        address="0x2791bca1f2de4661ed88a30c99a7a9449aa84174",
        token_symbol="USDC",
        decimals=6,
    )


@pytest.fixture
def freqtrade_pair(usdc_asset):
    """Create a Freqtrade trading pair."""
    return create_freqtrade_pair(
        freqtrade_id="momentum-bot",
        api_url="http://localhost:8080",
        exchange_name="binance",
        reserve_currency=usdc_asset,
        deposit_method="on_chain_transfer",
        recipient_address="0xabcdef0123456789abcdef0123456789abcdef01",
    )


@pytest.fixture
def mock_strategy_universe(freqtrade_pair, usdc_asset):
    """Create a mock strategy universe with a Freqtrade pair."""
    universe = Mock(spec=TradingStrategyUniverse)

    # Mock data universe with pairs
    data_universe = Mock()
    pairs_mock = Mock()
    pairs_mock.iterate_pairs.return_value = [freqtrade_pair]
    data_universe.pairs = pairs_mock

    # Mock exchange universe
    exchange_universe = Mock()
    exchange_universe.get_exchange_count.return_value = 0
    exchange_universe.exchanges = {}
    data_universe.exchange_universe = exchange_universe

    universe.data_universe = data_universe
    universe.reserve_assets = [usdc_asset]
    universe.get_reserve_asset.return_value = usdc_asset
    universe.get_single_chain.return_value = ChainId.polygon

    return universe


def test_default_match_router_freqtrade(mock_strategy_universe, freqtrade_pair):
    """Test that default_match_router routes Freqtrade pairs correctly."""
    routing_id = default_match_router(mock_strategy_universe, freqtrade_pair)

    assert routing_id.router_name == "freqtrade"
    assert routing_id.exchange_slug is None
    assert routing_id.lending_protocol_slug is None


def test_default_supported_routers_includes_freqtrade(mock_strategy_universe):
    """Test that default_supported_routers includes Freqtrade when Freqtrade pairs exist."""
    routers = default_supported_routers(mock_strategy_universe)

    # Should contain exactly one router for Freqtrade
    freqtrade_routers = [r for r in routers if r.router_name == "freqtrade"]
    assert len(freqtrade_routers) == 1
    assert freqtrade_routers[0].exchange_slug is None


def test_create_freqtrade_adapter_builds_config(mock_strategy_universe, freqtrade_pair):
    """Test that create_freqtrade_adapter builds FreqtradeConfig correctly from pair data."""
    # Set environment variables for credentials
    os.environ["FREQTRADE_MOMENTUM_BOT_USERNAME"] = "testuser"
    os.environ["FREQTRADE_MOMENTUM_BOT_PASSWORD"] = "testpass"

    try:
        web3 = Mock(spec=Web3)
        routing_id = default_match_router(mock_strategy_universe, freqtrade_pair)

        config = create_freqtrade_adapter(web3, mock_strategy_universe, routing_id)

        # Verify ProtocolRoutingConfig structure
        assert config.routing_id == routing_id
        assert config.routing_model is not None
        assert config.pricing_model is not None
        assert config.valuation_model is not None

        # Verify routing model has correct config
        assert "momentum-bot" in config.routing_model.freqtrade_configs
        bot_config = config.routing_model.freqtrade_configs["momentum-bot"]

        assert bot_config.freqtrade_id == "momentum-bot"
        assert bot_config.api_url == "http://localhost:8080"
        assert bot_config.api_username == "testuser"
        assert bot_config.api_password == "testpass"
        assert bot_config.exchange_name == "binance"

    finally:
        # Clean up environment variables
        del os.environ["FREQTRADE_MOMENTUM_BOT_USERNAME"]
        del os.environ["FREQTRADE_MOMENTUM_BOT_PASSWORD"]


def test_create_freqtrade_adapter_missing_credentials(mock_strategy_universe):
    """Test that create_freqtrade_adapter raises error when credentials are missing."""
    web3 = Mock(spec=Web3)

    # Ensure credentials are not in environment
    for key in ["FREQTRADE_MOMENTUM_BOT_USERNAME", "FREQTRADE_MOMENTUM_BOT_PASSWORD"]:
        if key in os.environ:
            del os.environ[key]

    from tradeexecutor.strategy.generic.pair_configurator import ProtocolRoutingId
    routing_id = ProtocolRoutingId(router_name="freqtrade")

    with pytest.raises(ValueError, match="Missing Freqtrade credentials"):
        create_freqtrade_adapter(web3, mock_strategy_universe, routing_id)


def test_ethereum_pair_configurator_routes_freqtrade(mock_strategy_universe, freqtrade_pair):
    """Test that EthereumPairConfigurator correctly routes Freqtrade pairs."""
    # Set environment variables for credentials
    os.environ["FREQTRADE_MOMENTUM_BOT_USERNAME"] = "testuser"
    os.environ["FREQTRADE_MOMENTUM_BOT_PASSWORD"] = "testpass"

    try:
        web3 = Mock(spec=Web3)
        configurator = EthereumPairConfigurator(web3, mock_strategy_universe)

        # Test that match_router returns freqtrade routing
        routing_id = configurator.match_router(freqtrade_pair)
        assert routing_id.router_name == "freqtrade"

        # Test that create_config returns proper config
        config = configurator.create_config(routing_id)
        assert config.routing_model is not None
        assert config.pricing_model is not None
        assert config.valuation_model is not None

    finally:
        # Clean up environment variables
        del os.environ["FREQTRADE_MOMENTUM_BOT_USERNAME"]
        del os.environ["FREQTRADE_MOMENTUM_BOT_PASSWORD"]
