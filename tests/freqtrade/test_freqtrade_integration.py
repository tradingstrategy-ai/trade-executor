"""Freqtrade position integration tests.

Tests full deposit, valuation, and withdrawal flows with fixture-based mocking.
Covers on-chain transfer and Aster exchange types.
"""

import datetime
from decimal import Decimal
from unittest.mock import patch, MagicMock

import pytest

from tradeexecutor.strategy.freqtrade.config import (
    FreqtradeConfig,
    OnChainTransferExchangeConfig,
    AsterExchangeConfig,
)
from tradeexecutor.strategy.freqtrade.freqtrade_routing import FreqtradeRoutingModel
from tradeexecutor.strategy.freqtrade.freqtrade_pricing import FreqtradePricingModel
from tradeexecutor.strategy.freqtrade.freqtrade_valuation import FreqtradeValuator


# =============================================================================
# On-chain transfer integration tests
# =============================================================================


def test_on_chain_transfer_deposit_flow(
    on_chain_transfer_config,
    mock_routing_state_factory,
    create_buy_trade,
    mock_token,
):
    """Test full on-chain transfer deposit flow with balance confirmation."""
    model = FreqtradeRoutingModel(
        freqtrade_configs={"momentum-bot": on_chain_transfer_config}
    )

    routing_state, mock_client = mock_routing_state_factory(
        "momentum-bot", initial_balance=500.0
    )
    trade = create_buy_trade(Decimal("1000.0"))

    with patch(
        "tradeexecutor.strategy.freqtrade.freqtrade_routing.fetch_erc20_details",
        return_value=mock_token,
    ):
        model.setup_trades(
            state=None,
            routing_state=routing_state,
            trades=[trade],
        )

    # Verify transaction structure
    assert len(trade.blockchain_transactions) == 1
    assert trade.other_data["balance_before_deposit"] == "500.0"

    # Simulate balance increase after deposit
    mock_client.set_balance(1500.0)

    # Use short timeout config for test
    short_timeout_config = FreqtradeConfig(
        freqtrade_id="momentum-bot",
        api_url="http://localhost:8080",
        api_username="testuser",
        api_password="testpass",
        exchange_name="binance",
        reserve_currency="0xdAC17F958D2ee523a2206206994597C13D831ec7",
        exchange=OnChainTransferExchangeConfig(
            recipient_address="0xabcdef0123456789abcdef0123456789abcdef01",
            fee_tolerance=Decimal("1.0"),
            confirmation_timeout=1,
            poll_interval=0.1,
        ),
    )

    with patch(
        "tradeexecutor.strategy.freqtrade.freqtrade_routing.FreqtradeClient"
    ) as MockClient:
        MockClient.return_value.get_balance.return_value = {"total": 1500.0}
        model._confirm_deposit(trade, short_timeout_config, short_timeout_config.exchange)

    assert "confirmed" in trade.notes.lower()


def test_on_chain_transfer_withdrawal_flow(
    on_chain_transfer_config,
    mock_routing_state_factory,
    create_sell_trade,
):
    """Test on-chain transfer withdrawal flow with balance decrease confirmation."""
    model = FreqtradeRoutingModel(
        freqtrade_configs={"momentum-bot": on_chain_transfer_config}
    )

    routing_state, mock_client = mock_routing_state_factory(
        "momentum-bot", initial_balance=1500.0
    )
    trade = create_sell_trade(Decimal("500.0"))

    model.setup_trades(
        state=None,
        routing_state=routing_state,
        trades=[trade],
    )

    # Withdrawals have no on-chain transactions (initiated externally)
    assert len(trade.blockchain_transactions) == 0
    assert trade.other_data["balance_before_withdrawal"] == "1500.0"

    # Simulate balance decrease after withdrawal
    short_timeout_config = FreqtradeConfig(
        freqtrade_id="momentum-bot",
        api_url="http://localhost:8080",
        api_username="testuser",
        api_password="testpass",
        exchange_name="binance",
        reserve_currency="0xdAC17F958D2ee523a2206206994597C13D831ec7",
        exchange=OnChainTransferExchangeConfig(
            recipient_address="0xabcdef0123456789abcdef0123456789abcdef01",
            fee_tolerance=Decimal("1.0"),
            confirmation_timeout=1,
            poll_interval=0.1,
        ),
    )

    with patch(
        "tradeexecutor.strategy.freqtrade.freqtrade_routing.FreqtradeClient"
    ) as MockClient:
        MockClient.return_value.get_balance.return_value = {"total": 1000.0}
        model._confirm_withdrawal(trade, short_timeout_config, short_timeout_config.exchange)

    assert "confirmed" in trade.notes.lower()


def test_on_chain_transfer_deposit_timeout(
    on_chain_transfer_config,
    mock_routing_state_factory,
    create_buy_trade,
    mock_token,
):
    """Test timeout when balance never increases during deposit confirmation."""
    model = FreqtradeRoutingModel(
        freqtrade_configs={"momentum-bot": on_chain_transfer_config}
    )

    routing_state, _ = mock_routing_state_factory("momentum-bot", initial_balance=500.0)
    trade = create_buy_trade(Decimal("1000.0"))

    with patch(
        "tradeexecutor.strategy.freqtrade.freqtrade_routing.fetch_erc20_details",
        return_value=mock_token,
    ):
        model.setup_trades(
            state=None,
            routing_state=routing_state,
            trades=[trade],
        )

    # Configure very short timeout
    short_timeout_config = FreqtradeConfig(
        freqtrade_id="momentum-bot",
        api_url="http://localhost:8080",
        api_username="testuser",
        api_password="testpass",
        exchange_name="binance",
        reserve_currency="0xdAC17F958D2ee523a2206206994597C13D831ec7",
        exchange=OnChainTransferExchangeConfig(
            recipient_address="0xabcdef0123456789abcdef0123456789abcdef01",
            fee_tolerance=Decimal("1.0"),
            confirmation_timeout=0.1,
            poll_interval=0.05,
        ),
    )

    # Balance never increases
    with patch(
        "tradeexecutor.strategy.freqtrade.freqtrade_routing.FreqtradeClient"
    ) as MockClient:
        MockClient.return_value.get_balance.return_value = {"total": 500.0}

        with pytest.raises(Exception, match="not confirmed within"):
            model._confirm_deposit(trade, short_timeout_config, short_timeout_config.exchange)


def test_on_chain_transfer_api_failure_during_confirmation(
    on_chain_transfer_config,
    mock_routing_state_factory,
    create_buy_trade,
    mock_token,
    caplog,
):
    """Test API failures during polling continue with warnings."""
    model = FreqtradeRoutingModel(
        freqtrade_configs={"momentum-bot": on_chain_transfer_config}
    )

    routing_state, _ = mock_routing_state_factory("momentum-bot", initial_balance=500.0)
    trade = create_buy_trade(Decimal("1000.0"))

    with patch(
        "tradeexecutor.strategy.freqtrade.freqtrade_routing.fetch_erc20_details",
        return_value=mock_token,
    ):
        model.setup_trades(
            state=None,
            routing_state=routing_state,
            trades=[trade],
        )

    short_timeout_config = FreqtradeConfig(
        freqtrade_id="momentum-bot",
        api_url="http://localhost:8080",
        api_username="testuser",
        api_password="testpass",
        exchange_name="binance",
        reserve_currency="0xdAC17F958D2ee523a2206206994597C13D831ec7",
        exchange=OnChainTransferExchangeConfig(
            recipient_address="0xabcdef0123456789abcdef0123456789abcdef01",
            fee_tolerance=Decimal("1.0"),
            confirmation_timeout=1,
            poll_interval=0.1,
        ),
    )

    call_count = [0]

    def get_balance_with_failures():
        call_count[0] += 1
        if call_count[0] <= 2:
            raise Exception("Temporary API failure")
        return {"total": 1500.0}

    with patch(
        "tradeexecutor.strategy.freqtrade.freqtrade_routing.FreqtradeClient"
    ) as MockClient:
        MockClient.return_value.get_balance = get_balance_with_failures

        with caplog.at_level("WARNING"):
            model._confirm_deposit(trade, short_timeout_config, short_timeout_config.exchange)

    # Should have logged warnings but eventually succeeded
    assert "Balance check failed" in caplog.text
    assert "confirmed" in trade.notes.lower()


def test_on_chain_transfer_position_state_after_deposit(
    on_chain_transfer_config,
    mock_freqtrade_client_factory,
    create_position_with_deposit,
    freqtrade_pair,
):
    """Test position state and valuation after deposit."""
    position = create_position_with_deposit(Decimal("1000.0"))

    # Verify position state
    assert position.get_quantity() == Decimal("1000.0")
    assert position.is_open()

    # Set up valuation with API balance matching deposit
    mock_client = mock_freqtrade_client_factory(initial_balance=1000.0)
    pricing_model = FreqtradePricingModel({"momentum-bot": mock_client})
    valuator = FreqtradeValuator(pricing_model)

    ts = datetime.datetime(2024, 1, 2)
    update = valuator(ts, position)

    assert update.new_value == pytest.approx(1000.0)
    assert update.quantity == Decimal("1000")


# =============================================================================
# Aster integration tests
# =============================================================================


def test_aster_deposit_flow(
    aster_config,
    mock_routing_state_factory,
    create_buy_trade,
    mock_token,
):
    """Test full Aster deposit flow with approve + deposit transactions."""
    model = FreqtradeRoutingModel(
        freqtrade_configs={"momentum-bot": aster_config}
    )

    routing_state, mock_client = mock_routing_state_factory(
        "momentum-bot", initial_balance=500.0
    )
    trade = create_buy_trade(Decimal("1000.0"))

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
            trades=[trade],
        )

    # Verify two transactions (approve + deposit)
    assert len(trade.blockchain_transactions) == 2
    assert trade.other_data["balance_before_deposit"] == "500.0"

    # Simulate balance increase after deposit
    mock_client.set_balance(1500.0)

    short_timeout_config = FreqtradeConfig(
        freqtrade_id="momentum-bot",
        api_url="http://localhost:8080",
        api_username="testuser",
        api_password="testpass",
        exchange_name="aster",
        reserve_currency="0xdAC17F958D2ee523a2206206994597C13D831ec7",
        exchange=AsterExchangeConfig(
            vault_address="0x1234567890123456789012345678901234567890",
            broker_id=0,
            fee_tolerance=Decimal("1.0"),
            confirmation_timeout=1,
            poll_interval=0.1,
        ),
    )

    with patch(
        "tradeexecutor.strategy.freqtrade.freqtrade_routing.FreqtradeClient"
    ) as MockClient:
        MockClient.return_value.get_balance.return_value = {"total": 1500.0}
        model._confirm_deposit(trade, short_timeout_config, short_timeout_config.exchange)

    assert "confirmed" in trade.notes.lower()


def test_aster_deposit_with_custom_broker_id(
    mock_routing_state_factory,
    create_buy_trade,
    mock_token,
):
    """Test Aster deposit with custom broker_id."""
    custom_broker_config = FreqtradeConfig(
        freqtrade_id="momentum-bot",
        api_url="http://localhost:8080",
        api_username="testuser",
        api_password="testpass",
        exchange_name="aster",
        reserve_currency="0xdAC17F958D2ee523a2206206994597C13D831ec7",
        exchange=AsterExchangeConfig(
            vault_address="0x1234567890123456789012345678901234567890",
            broker_id=42,
            fee_tolerance=Decimal("1.0"),
            confirmation_timeout=300,
            poll_interval=5,
        ),
    )

    model = FreqtradeRoutingModel(
        freqtrade_configs={"momentum-bot": custom_broker_config}
    )

    routing_state, _ = mock_routing_state_factory("momentum-bot", initial_balance=500.0)
    trade = create_buy_trade(Decimal("1000.0"))

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
            trades=[trade],
        )

    # Verify vault.deposit was called with broker_id=42
    mock_vault.functions.deposit.assert_called_once()
    call_args = mock_vault.functions.deposit.call_args[0]
    assert call_args[2] == 42  # broker_id is third argument


def test_aster_deposit_timeout(
    aster_config,
    mock_routing_state_factory,
    create_buy_trade,
    mock_token,
):
    """Test timeout when balance never increases during Aster deposit confirmation."""
    model = FreqtradeRoutingModel(
        freqtrade_configs={"momentum-bot": aster_config}
    )

    routing_state, _ = mock_routing_state_factory("momentum-bot", initial_balance=500.0)
    trade = create_buy_trade(Decimal("1000.0"))

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
            trades=[trade],
        )

    short_timeout_config = FreqtradeConfig(
        freqtrade_id="momentum-bot",
        api_url="http://localhost:8080",
        api_username="testuser",
        api_password="testpass",
        exchange_name="aster",
        reserve_currency="0xdAC17F958D2ee523a2206206994597C13D831ec7",
        exchange=AsterExchangeConfig(
            vault_address="0x1234567890123456789012345678901234567890",
            broker_id=0,
            fee_tolerance=Decimal("1.0"),
            confirmation_timeout=0.1,
            poll_interval=0.05,
        ),
    )

    # Balance never increases
    with patch(
        "tradeexecutor.strategy.freqtrade.freqtrade_routing.FreqtradeClient"
    ) as MockClient:
        MockClient.return_value.get_balance.return_value = {"total": 500.0}

        with pytest.raises(Exception, match="not confirmed within"):
            model._confirm_deposit(trade, short_timeout_config, short_timeout_config.exchange)


def test_aster_withdrawal_not_implemented(
    aster_config,
    mock_routing_state_factory,
    create_sell_trade,
):
    """Test that Aster withdrawal raises NotImplementedError."""
    model = FreqtradeRoutingModel(
        freqtrade_configs={"momentum-bot": aster_config}
    )

    routing_state, _ = mock_routing_state_factory("momentum-bot", initial_balance=1500.0)
    trade = create_sell_trade(Decimal("500.0"))

    with pytest.raises(NotImplementedError, match="signature infrastructure"):
        model.setup_trades(
            state=None,
            routing_state=routing_state,
            trades=[trade],
        )


def test_aster_position_state_after_deposit(
    mock_freqtrade_client_factory,
    create_position_with_deposit,
):
    """Test position state after Aster deposit."""
    position = create_position_with_deposit(Decimal("1000.0"))

    # Verify position state
    assert position.get_quantity() == Decimal("1000.0")
    assert position.is_open()

    # Set up valuation
    mock_client = mock_freqtrade_client_factory(initial_balance=1000.0)
    pricing_model = FreqtradePricingModel({"momentum-bot": mock_client})
    valuator = FreqtradeValuator(pricing_model)

    ts = datetime.datetime(2024, 1, 2)
    update = valuator(ts, position)

    assert update.new_value == pytest.approx(1000.0)
    assert update.quantity == Decimal("1000")


def test_aster_valuation_with_profit(
    mock_freqtrade_client_factory,
    create_position_with_deposit,
    caplog,
):
    """Test valuation when API reports profit (balance drift warning)."""
    position = create_position_with_deposit(Decimal("1000.0"))

    # API returns 10% profit
    mock_client = mock_freqtrade_client_factory(initial_balance=1100.0)
    pricing_model = FreqtradePricingModel({"momentum-bot": mock_client})
    valuator = FreqtradeValuator(pricing_model)

    ts = datetime.datetime(2024, 1, 2)

    with caplog.at_level("WARNING"):
        update = valuator(ts, position)

    # Valuation should reflect API balance
    assert update.new_value == pytest.approx(1100.0)
    assert update.quantity == Decimal("1100")

    # Balance drift warning should be logged (>1% difference)
    assert "Balance drift detected" in caplog.text
