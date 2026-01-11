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
    FreqtradeDepositMethod,
    AsterDepositConfig,
    HyperliquidDepositConfig,
    OrderlyDepositConfig,
)
from tradeexecutor.strategy.freqtrade.freqtrade_routing import (
    FreqtradeRoutingModel,
    FreqtradeRoutingState,
)
from tradingstrategy.chain import ChainId


@pytest.fixture
def freqtrade_config_aster():
    """Create an Aster vault deposit Freqtrade config."""
    return FreqtradeConfig(
        freqtrade_id="momentum-bot",
        api_url="http://localhost:8080",
        api_username="testuser",
        api_password="testpass",
        exchange="aster",
        reserve_currency="0xdAC17F958D2ee523a2206206994597C13D831ec7",  # USDT address
        deposit=AsterDepositConfig(
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
        exchange="hyperliquid",
        reserve_currency="0xaf88d065e77c8cC2239327C5EDb3A432268e5831",  # USDC on Arbitrum
        deposit=HyperliquidDepositConfig(
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
        exchange="modetrade",
        reserve_currency="0xdAC17F958D2ee523a2206206994597C13D831ec7",  # USDT address
        deposit=OrderlyDepositConfig(
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


class TestFreqtradeRoutingModel:
    """Tests for FreqtradeRoutingModel."""

    def test_init(self, freqtrade_config_aster):
        """Test routing model initialisation."""
        model = FreqtradeRoutingModel(
            freqtrade_configs={"momentum-bot": freqtrade_config_aster}
        )
        assert "momentum-bot" in model.freqtrade_configs
        assert model.freqtrade_configs["momentum-bot"] == freqtrade_config_aster

    def test_create_routing_state_without_execution_details(self, freqtrade_config_aster):
        """Test routing state creation without execution details."""
        model = FreqtradeRoutingModel(
            freqtrade_configs={"momentum-bot": freqtrade_config_aster}
        )

        with patch("tradeexecutor.strategy.freqtrade.freqtrade_routing.FreqtradeClient") as mock_client:
            routing_state = model.create_routing_state(
                universe=None,
                execution_details=None,
            )

        assert isinstance(routing_state, FreqtradeRoutingState)
        assert routing_state.tx_builder is None
        assert routing_state.web3 is None
        assert "momentum-bot" in routing_state.freqtrade_clients

    def test_create_routing_state_with_tx_builder(self, freqtrade_config_aster):
        """Test routing state creation with tx_builder."""
        model = FreqtradeRoutingModel(
            freqtrade_configs={"momentum-bot": freqtrade_config_aster}
        )

        mock_web3 = MagicMock()
        mock_tx_builder = MagicMock()
        mock_tx_builder.web3 = mock_web3

        with patch("tradeexecutor.strategy.freqtrade.freqtrade_routing.FreqtradeClient"):
            routing_state = model.create_routing_state(
                universe=None,
                execution_details={"tx_builder": mock_tx_builder},
            )

        assert routing_state.tx_builder == mock_tx_builder
        assert routing_state.web3 == mock_web3


class TestAsterDeposit:
    """Tests for Aster vault deposit flow."""

    def test_setup_trades_aster_missing_tx_builder(self, freqtrade_config_aster, mock_trade):
        """Test setup_trades raises error when tx_builder is missing."""
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

    def test_setup_trades_aster_missing_vault_address(self, mock_trade):
        """Test setup_trades raises error when vault_address is missing."""
        config = FreqtradeConfig(
            freqtrade_id="momentum-bot",
            api_url="http://localhost:8080",
            api_username="testuser",
            api_password="testpass",
            exchange="aster",
            reserve_currency="0xdAC17F958D2ee523a2206206994597C13D831ec7",
            deposit=AsterDepositConfig(
                vault_address=None,  # Missing
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

    def test_build_aster_deposit_tx(self, freqtrade_config_aster, mock_trade):
        """Test building Aster vault deposit transactions (approve + deposit)."""
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

        # Mock fetch_erc20_details
        mock_token = MagicMock()
        mock_token.convert_to_raw.return_value = 1000_000_000
        mock_token.contract.functions.approve.return_value = MagicMock()

        # Mock get_deployed_contract for vault
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

        # Verify balance was queried
        mock_client.get_balance.assert_called_once()

        # Verify balance before was stored
        assert mock_trade.other_data is not None
        assert mock_trade.other_data.get("balance_before_deposit") == "500"

        # Verify approve and deposit were built
        mock_token.contract.functions.approve.assert_called_once()
        mock_vault.functions.deposit.assert_called_once()

        # Verify two transactions were signed (approve + deposit)
        assert mock_tx_builder.sign_transaction.call_count == 2

        # Verify transactions were attached to trade
        assert mock_trade.blockchain_transactions is not None
        assert len(mock_trade.blockchain_transactions) == 2

    def test_confirm_aster_deposit_success(self, freqtrade_config_aster, mock_trade):
        """Test Aster deposit confirmation when balance increases."""
        model = FreqtradeRoutingModel(
            freqtrade_configs={"momentum-bot": freqtrade_config_aster}
        )

        mock_trade.other_data = {"balance_before_deposit": "500"}

        mock_client = MagicMock()
        mock_client.get_balance.return_value = {"total": 1499.5}

        with patch(
            "tradeexecutor.strategy.freqtrade.freqtrade_routing.FreqtradeClient",
            return_value=mock_client,
        ):
            model.settle_trade(
                web3=MagicMock(),
                state=None,
                trade=mock_trade,
                receipts={},
            )

        assert "Deposit confirmed" in mock_trade.notes


class TestHyperliquidDeposit:
    """Tests for Hyperliquid deposit flow."""

    def test_setup_trades_hyperliquid_missing_tx_builder(self, freqtrade_config_hyperliquid, mock_trade):
        """Test setup_trades raises error when tx_builder is missing."""
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

    def test_build_hyperliquid_bridge_transfer_tx(self, freqtrade_config_hyperliquid, mock_trade):
        """Test building Hyperliquid bridge transfer transaction."""
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

        # Verify transfer was built (not approve - Hyperliquid uses direct transfer)
        mock_token.contract.functions.transfer.assert_called_once()

        # Only one transaction (bridge transfer)
        assert mock_tx_builder.sign_transaction.call_count == 1
        assert len(mock_trade.blockchain_transactions) == 1


class TestOrderlyDeposit:
    """Tests for Orderly vault deposit flow."""

    def test_setup_trades_orderly_missing_tx_builder(self, freqtrade_config_orderly, mock_trade):
        """Test setup_trades raises error when tx_builder is missing."""
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

    def test_setup_trades_orderly_missing_account_id(self, mock_trade):
        """Test setup_trades raises error when orderly_account_id is missing."""
        config = FreqtradeConfig(
            freqtrade_id="momentum-bot",
            api_url="http://localhost:8080",
            api_username="testuser",
            api_password="testpass",
            exchange="modetrade",
            reserve_currency="0xdAC17F958D2ee523a2206206994597C13D831ec7",
            deposit=OrderlyDepositConfig(
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

    def test_build_orderly_deposit_tx(self, freqtrade_config_orderly, mock_trade):
        """Test building Orderly vault deposit transactions."""
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

        # Verify approve and deposit were built
        mock_token.contract.functions.approve.assert_called_once()
        mock_vault.functions.deposit.assert_called_once()

        # Two transactions (approve + deposit)
        assert mock_tx_builder.sign_transaction.call_count == 2
        assert len(mock_trade.blockchain_transactions) == 2


class TestFreqtradeRoutingState:
    """Tests for FreqtradeRoutingState dataclass."""

    def test_routing_state_creation(self):
        """Test routing state can be created."""
        mock_tx_builder = MagicMock()
        mock_web3 = MagicMock()
        mock_client = MagicMock()

        state = FreqtradeRoutingState(
            tx_builder=mock_tx_builder,
            web3=mock_web3,
            freqtrade_clients={"bot1": mock_client},
        )

        assert state.tx_builder == mock_tx_builder
        assert state.web3 == mock_web3
        assert state.freqtrade_clients["bot1"] == mock_client

    def test_routing_state_with_none_values(self):
        """Test routing state with None values."""
        state = FreqtradeRoutingState(
            tx_builder=None,
            web3=None,
            freqtrade_clients={},
        )

        assert state.tx_builder is None
        assert state.web3 is None
        assert state.freqtrade_clients == {}


class TestDepositConfigs:
    """Tests for deposit configuration classes."""

    def test_aster_config(self):
        """Test Aster deposit config."""
        config = AsterDepositConfig(
            vault_address="0x1234567890123456789012345678901234567890",
            broker_id=5,
            fee_tolerance=Decimal("0.5"),
        )

        assert config.method == FreqtradeDepositMethod.aster_vault
        assert config.vault_address == "0x1234567890123456789012345678901234567890"
        assert config.broker_id == 5
        assert config.fee_tolerance == Decimal("0.5")

    def test_hyperliquid_config(self):
        """Test Hyperliquid deposit config."""
        config = HyperliquidDepositConfig(
            vault_address="0xabcdef0123456789abcdef0123456789abcdef01",
            is_mainnet=False,
        )

        assert config.method == FreqtradeDepositMethod.hyperliquid
        assert config.vault_address == "0xabcdef0123456789abcdef0123456789abcdef01"
        assert config.is_mainnet is False

    def test_orderly_config(self):
        """Test Orderly deposit config."""
        config = OrderlyDepositConfig(
            vault_address="0x1234567890123456789012345678901234567890",
            orderly_account_id="0xabcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
            broker_id="my_broker",
            token_id="USDT",
        )

        assert config.method == FreqtradeDepositMethod.orderly_vault
        assert config.vault_address == "0x1234567890123456789012345678901234567890"
        assert config.orderly_account_id == "0xabcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
        assert config.broker_id == "my_broker"
        assert config.token_id == "USDT"
