"""Freqtrade routing model tests for multi-method deposit flow."""

import datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from tradeexecutor.state.identifier import (
    TradingPairIdentifier,
    AssetIdentifier,
    TradingPairKind,
)
from tradeexecutor.state.trade import TradeExecution, TradeType
from tradeexecutor.strategy.freqtrade.config import (
    FreqtradeConfig,
    OnChainTransferExchangeConfig,
    AsterExchangeConfig,
    HyperliquidExchangeConfig,
    OrderlyExchangeConfig,
)
from tradeexecutor.strategy.freqtrade.freqtrade_routing import (
    FreqtradeRoutingModel,
    FreqtradeRoutingState,
)
from tradingstrategy.chain import ChainId


@pytest.fixture
def freqtrade_config_on_chain_transfer():
    """Create an on-chain transfer deposit Freqtrade config."""
    return FreqtradeConfig(
        freqtrade_id="momentum-bot",
        api_url="http://localhost:8080",
        api_username="testuser",
        api_password="testpass",
        exchange_name="binance",
        reserve_currency="0xdAC17F958D2ee523a2206206994597C13D831ec7",  # USDT address
        exchange=OnChainTransferExchangeConfig(
            recipient_address="0xabcdef0123456789abcdef0123456789abcdef01",
            fee_tolerance=Decimal("0.5"),
            confirmation_timeout=300,
            poll_interval=5,
        ),
    )


@pytest.fixture
def freqtrade_config_aster():
    """Create an Aster vault deposit Freqtrade config."""
    return FreqtradeConfig(
        freqtrade_id="momentum-bot",
        api_url="http://localhost:8080",
        api_username="testuser",
        api_password="testpass",
        exchange_name="aster",
        reserve_currency="0xdAC17F958D2ee523a2206206994597C13D831ec7",  # USDT address
        exchange=AsterExchangeConfig(
            vault_address="0x1234567890123456789012345678901234567890",
            broker_id=0,
            fee_tolerance=Decimal("0.5"),
            confirmation_timeout=300,
            poll_interval=5,
        ),
    )


@pytest.fixture
def freqtrade_config_hyperliquid():
    """Create a Hyperliquid deposit Freqtrade config."""
    return FreqtradeConfig(
        freqtrade_id="momentum-bot",
        api_url="http://localhost:8080",
        api_username="testuser",
        api_password="testpass",
        exchange_name="hyperliquid",
        reserve_currency="0xaf88d065e77c8cC2239327C5EDb3A432268e5831",  # USDC on Arbitrum
        exchange=HyperliquidExchangeConfig(
            vault_address="0xabcdef0123456789abcdef0123456789abcdef01",
            is_mainnet=True,
            fee_tolerance=Decimal("0.5"),
            confirmation_timeout=300,
            poll_interval=5,
        ),
    )


@pytest.fixture
def freqtrade_config_orderly():
    """Create an Orderly vault deposit Freqtrade config."""
    return FreqtradeConfig(
        freqtrade_id="momentum-bot",
        api_url="http://localhost:8080",
        api_username="testuser",
        api_password="testpass",
        exchange_name="modetrade",
        reserve_currency="0xdAC17F958D2ee523a2206206994597C13D831ec7",  # USDT address
        exchange=OrderlyExchangeConfig(
            vault_address="0x1234567890123456789012345678901234567890",
            orderly_account_id="0xabcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
            broker_id="test_broker",
            token_id="USDT",
            fee_tolerance=Decimal("0.5"),
            confirmation_timeout=300,
            poll_interval=5,
        ),
    )


@pytest.fixture
def freqtrade_pair():
    """Create a Freqtrade trading pair."""
    usdt = AssetIdentifier(
        ChainId.centralised_exchange.value,
        "0x1",
        "USDT",
        6,
        1,
    )
    ft_strategy = AssetIdentifier(
        ChainId.centralised_exchange.value,
        "0x2",
        "FT-MOMENTUM",
        6,
        2,
    )
    return TradingPairIdentifier(
        ft_strategy,
        usdt,
        "freqtrade-momentum/usdt",
        "binance",
        internal_id=1,
        internal_exchange_id=1,
        fee=0.0,
        kind=TradingPairKind.freqtrade,
        other_data={
            "freqtrade_id": "momentum-bot",
            "freqtrade_url": "http://localhost:8080",
            "exchange": "binance",
        },
    )


@pytest.fixture
def mock_trade(freqtrade_pair):
    """Create a mock trade for testing."""
    opened_at = datetime.datetime(2024, 1, 1)
    return TradeExecution(
        trade_id=1,
        position_id=1,
        trade_type=TradeType.rebalance,
        pair=freqtrade_pair,
        opened_at=opened_at,
        planned_quantity=Decimal("1000.0"),
        planned_price=1.0,
        planned_reserve=Decimal("1000.0"),
        reserve_currency=freqtrade_pair.quote,
    )


# Configuration validation tests


def test_on_chain_transfer_missing_tx_builder(
    freqtrade_config_on_chain_transfer, mock_trade
):
    """Test that on-chain transfer deposits require tx_builder."""
    model = FreqtradeRoutingModel(
        freqtrade_configs={"momentum-bot": freqtrade_config_on_chain_transfer}
    )

    routing_state = FreqtradeRoutingState(
        tx_builder=None,
        web3=None,
        freqtrade_clients={"momentum-bot": MagicMock()},
    )

    with pytest.raises(ValueError, match="tx_builder required"):
        model.setup_trades(
            state=None,
            routing_state=routing_state,
            trades=[mock_trade],
        )


def test_on_chain_transfer_missing_recipient(mock_trade):
    """Test that on-chain transfer deposits require recipient_address."""
    config = FreqtradeConfig(
        freqtrade_id="momentum-bot",
        api_url="http://localhost:8080",
        api_username="testuser",
        api_password="testpass",
        exchange_name="binance",
        reserve_currency="0xdAC17F958D2ee523a2206206994597C13D831ec7",
        exchange=OnChainTransferExchangeConfig(
            recipient_address=None,  # Missing
        ),
    )

    model = FreqtradeRoutingModel(freqtrade_configs={"momentum-bot": config})

    mock_tx_builder = MagicMock()
    mock_client = MagicMock()
    mock_client.get_balance.return_value = {"total": 500}

    routing_state = FreqtradeRoutingState(
        tx_builder=mock_tx_builder,
        web3=MagicMock(),
        freqtrade_clients={"momentum-bot": mock_client},
    )

    with pytest.raises(ValueError, match="recipient_address required"):
        model.setup_trades(
            state=None,
            routing_state=routing_state,
            trades=[mock_trade],
        )


def test_aster_missing_tx_builder(freqtrade_config_aster, mock_trade):
    """Test that Aster vault deposits require tx_builder."""
    model = FreqtradeRoutingModel(
        freqtrade_configs={"momentum-bot": freqtrade_config_aster}
    )

    routing_state = FreqtradeRoutingState(
        tx_builder=None,
        web3=None,
        freqtrade_clients={"momentum-bot": MagicMock()},
    )

    with pytest.raises(ValueError, match="tx_builder required"):
        model.setup_trades(
            state=None,
            routing_state=routing_state,
            trades=[mock_trade],
        )


def test_aster_missing_vault_address(mock_trade):
    """Test that Aster vault deposits require vault_address."""
    config = FreqtradeConfig(
        freqtrade_id="momentum-bot",
        api_url="http://localhost:8080",
        api_username="testuser",
        api_password="testpass",
        exchange_name="aster",
        reserve_currency="0xdAC17F958D2ee523a2206206994597C13D831ec7",
        exchange=AsterExchangeConfig(
            vault_address=None,  # Missing
            broker_id=0,
        ),
    )

    model = FreqtradeRoutingModel(freqtrade_configs={"momentum-bot": config})

    mock_tx_builder = MagicMock()
    mock_client = MagicMock()
    mock_client.get_balance.return_value = {"total": 500}

    routing_state = FreqtradeRoutingState(
        tx_builder=mock_tx_builder,
        web3=MagicMock(),
        freqtrade_clients={"momentum-bot": mock_client},
    )

    with pytest.raises(ValueError, match="vault_address required"):
        model.setup_trades(
            state=None,
            routing_state=routing_state,
            trades=[mock_trade],
        )


def test_hyperliquid_missing_tx_builder(freqtrade_config_hyperliquid, mock_trade):
    """Test that Hyperliquid deposits require tx_builder."""
    model = FreqtradeRoutingModel(
        freqtrade_configs={"momentum-bot": freqtrade_config_hyperliquid}
    )

    routing_state = FreqtradeRoutingState(
        tx_builder=None,
        web3=None,
        freqtrade_clients={"momentum-bot": MagicMock()},
    )

    with pytest.raises(ValueError, match="tx_builder required"):
        model.setup_trades(
            state=None,
            routing_state=routing_state,
            trades=[mock_trade],
        )


def test_orderly_missing_tx_builder(freqtrade_config_orderly, mock_trade):
    """Test that Orderly vault deposits require tx_builder."""
    model = FreqtradeRoutingModel(
        freqtrade_configs={"momentum-bot": freqtrade_config_orderly}
    )

    routing_state = FreqtradeRoutingState(
        tx_builder=None,
        web3=None,
        freqtrade_clients={"momentum-bot": MagicMock()},
    )

    with pytest.raises(ValueError, match="tx_builder required"):
        model.setup_trades(
            state=None,
            routing_state=routing_state,
            trades=[mock_trade],
        )


def test_orderly_missing_account_id(mock_trade):
    """Test that Orderly vault deposits require orderly_account_id."""
    config = FreqtradeConfig(
        freqtrade_id="momentum-bot",
        api_url="http://localhost:8080",
        api_username="testuser",
        api_password="testpass",
        exchange_name="modetrade",
        reserve_currency="0xdAC17F958D2ee523a2206206994597C13D831ec7",
        exchange=OrderlyExchangeConfig(
            vault_address="0x1234567890123456789012345678901234567890",
            orderly_account_id=None,  # Missing
            broker_id="test_broker",
        ),
    )

    model = FreqtradeRoutingModel(freqtrade_configs={"momentum-bot": config})

    mock_tx_builder = MagicMock()
    mock_client = MagicMock()
    mock_client.get_balance.return_value = {"total": 500}

    routing_state = FreqtradeRoutingState(
        tx_builder=mock_tx_builder,
        web3=MagicMock(),
        freqtrade_clients={"momentum-bot": mock_client},
    )

    with pytest.raises(ValueError, match="orderly_account_id required"):
        model.setup_trades(
            state=None,
            routing_state=routing_state,
            trades=[mock_trade],
        )


# Transaction structure tests


def test_build_on_chain_transfer_tx(freqtrade_config_on_chain_transfer, mock_trade):
    """Test that on-chain transfer builds single transfer transaction."""
    model = FreqtradeRoutingModel(
        freqtrade_configs={"momentum-bot": freqtrade_config_on_chain_transfer}
    )

    mock_web3 = MagicMock()
    mock_tx_builder = MagicMock()
    mock_tx_builder.web3 = mock_web3
    mock_tx_builder.sign_transaction.return_value = MagicMock()

    mock_client = MagicMock()
    mock_client.get_balance.return_value = {"total": 500}

    routing_state = FreqtradeRoutingState(
        tx_builder=mock_tx_builder,
        web3=mock_web3,
        freqtrade_clients={"momentum-bot": mock_client},
    )

    mock_token = MagicMock()
    mock_token.convert_to_raw.return_value = 1000_000_000
    mock_token.contract.functions.transfer.return_value = MagicMock()

    with patch(
        "tradeexecutor.strategy.freqtrade.freqtrade_routing.fetch_erc20_details",
        return_value=mock_token,
    ):
        model.setup_trades(
            state=None,
            routing_state=routing_state,
            trades=[mock_trade],
        )

    # Verify single transaction (transfer only)
    assert mock_tx_builder.sign_transaction.call_count == 1
    assert len(mock_trade.blockchain_transactions) == 1


def test_build_aster_deposit_tx(freqtrade_config_aster, mock_trade):
    """Test that Aster vault deposit builds approve + deposit transactions."""
    model = FreqtradeRoutingModel(
        freqtrade_configs={"momentum-bot": freqtrade_config_aster}
    )

    mock_web3 = MagicMock()
    mock_tx_builder = MagicMock()
    mock_tx_builder.web3 = mock_web3
    mock_tx_builder.sign_transaction.return_value = MagicMock()

    mock_client = MagicMock()
    mock_client.get_balance.return_value = {"total": 500}

    routing_state = FreqtradeRoutingState(
        tx_builder=mock_tx_builder,
        web3=mock_web3,
        freqtrade_clients={"momentum-bot": mock_client},
    )

    mock_token = MagicMock()
    mock_token.convert_to_raw.return_value = 1000_000_000
    mock_token.contract.functions.approve.return_value = MagicMock()

    mock_vault = MagicMock()
    mock_vault.functions.deposit.return_value = MagicMock()

    with patch(
        "tradeexecutor.strategy.freqtrade.freqtrade_routing.fetch_erc20_details",
        return_value=mock_token,
    ), patch(
        "tradeexecutor.strategy.freqtrade.freqtrade_routing.get_deployed_contract",
        return_value=mock_vault,
    ):
        model.setup_trades(
            state=None,
            routing_state=routing_state,
            trades=[mock_trade],
        )

    # Verify two transactions (approve + deposit)
    assert mock_tx_builder.sign_transaction.call_count == 2
    assert len(mock_trade.blockchain_transactions) == 2


def test_build_hyperliquid_bridge_transfer_tx(
    freqtrade_config_hyperliquid, mock_trade
):
    """Test that Hyperliquid deposit builds single bridge transfer transaction."""
    model = FreqtradeRoutingModel(
        freqtrade_configs={"momentum-bot": freqtrade_config_hyperliquid}
    )

    mock_web3 = MagicMock()
    mock_tx_builder = MagicMock()
    mock_tx_builder.web3 = mock_web3
    mock_tx_builder.sign_transaction.return_value = MagicMock()

    mock_client = MagicMock()
    mock_client.get_balance.return_value = {"total": 500}

    routing_state = FreqtradeRoutingState(
        tx_builder=mock_tx_builder,
        web3=mock_web3,
        freqtrade_clients={"momentum-bot": mock_client},
    )

    mock_token = MagicMock()
    mock_token.convert_to_raw.return_value = 1000_000_000
    mock_token.contract.functions.transfer.return_value = MagicMock()

    with patch(
        "tradeexecutor.strategy.freqtrade.freqtrade_routing.fetch_erc20_details",
        return_value=mock_token,
    ):
        model.setup_trades(
            state=None,
            routing_state=routing_state,
            trades=[mock_trade],
        )

    # Verify vault address stored for settle_trade
    assert mock_trade.other_data.get("hyperliquid_vault_address") is not None
    assert mock_trade.other_data.get("hyperliquid_is_mainnet") is True

    # Verify single transaction (bridge transfer)
    assert mock_tx_builder.sign_transaction.call_count == 1
    assert len(mock_trade.blockchain_transactions) == 1


def test_build_orderly_deposit_tx(freqtrade_config_orderly, mock_trade):
    """Test that Orderly vault deposit builds approve + deposit transactions."""
    model = FreqtradeRoutingModel(
        freqtrade_configs={"momentum-bot": freqtrade_config_orderly}
    )

    mock_web3 = MagicMock()
    mock_web3.keccak.return_value = bytes(32)
    mock_tx_builder = MagicMock()
    mock_tx_builder.web3 = mock_web3
    mock_tx_builder.sign_transaction.return_value = MagicMock()

    mock_client = MagicMock()
    mock_client.get_balance.return_value = {"total": 500}

    routing_state = FreqtradeRoutingState(
        tx_builder=mock_tx_builder,
        web3=mock_web3,
        freqtrade_clients={"momentum-bot": mock_client},
    )

    mock_token = MagicMock()
    mock_token.convert_to_raw.return_value = 1000_000_000
    mock_token.contract.functions.approve.return_value = MagicMock()

    mock_vault = MagicMock()
    mock_vault.functions.deposit.return_value = MagicMock()

    with patch(
        "tradeexecutor.strategy.freqtrade.freqtrade_routing.fetch_erc20_details",
        return_value=mock_token,
    ), patch(
        "tradeexecutor.strategy.freqtrade.freqtrade_routing.get_deployed_contract",
        return_value=mock_vault,
    ):
        model.setup_trades(
            state=None,
            routing_state=routing_state,
            trades=[mock_trade],
        )

    # Verify keccak was called for broker_id and token_id hashing
    assert mock_web3.keccak.call_count >= 1

    # Verify two transactions (approve + deposit)
    assert mock_tx_builder.sign_transaction.call_count == 2
    assert len(mock_trade.blockchain_transactions) == 2
