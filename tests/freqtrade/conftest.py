"""Shared fixtures for Freqtrade integration tests."""

import datetime
from decimal import Decimal
from unittest.mock import Mock, MagicMock

import pytest

from tradeexecutor.state.identifier import (
    TradingPairIdentifier,
    AssetIdentifier,
    TradingPairKind,
)
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.trade import TradeExecution, TradeType
from tradeexecutor.strategy.freqtrade.config import (
    FreqtradeConfig,
    OnChainTransferExchangeConfig,
    AsterExchangeConfig,
)
from tradeexecutor.strategy.freqtrade.freqtrade_client import FreqtradeClient
from tradeexecutor.strategy.freqtrade.freqtrade_routing import FreqtradeRoutingState
from tradingstrategy.chain import ChainId


@pytest.fixture
def mock_freqtrade_client_factory():
    """Factory for creating configurable mock FreqtradeClient instances.

    Returns a callable that creates mock clients with controllable balance.
    The returned client has a `set_balance` method to change balance mid-test.
    """
    def _create(initial_balance: float = 1000.0, fail_after: int | None = None):
        client = Mock(spec=FreqtradeClient)
        state = {"balance": initial_balance, "call_count": 0, "fail_after": fail_after}

        def get_balance():
            state["call_count"] += 1
            if state["fail_after"] and state["call_count"] > state["fail_after"]:
                raise Exception("API unreachable")
            return {
                "total": state["balance"],
                "free": state["balance"],
                "used": 0,
            }

        def set_balance(new_balance: float):
            state["balance"] = new_balance

        client.get_balance = get_balance
        client.set_balance = set_balance
        client._state = state
        return client

    return _create


@pytest.fixture
def mock_web3():
    """Mock Web3 instance with keccak and checksum address support."""
    web3 = MagicMock()
    web3.keccak.return_value = bytes(32)
    web3.to_checksum_address = lambda addr: addr
    return web3


@pytest.fixture
def mock_tx_builder(mock_web3):
    """Mock transaction builder that returns trackable tx objects."""
    tx_builder = MagicMock()
    tx_builder.web3 = mock_web3

    tx_counter = [0]

    def sign_transaction(contract, call, gas_limit=None, notes=None):
        tx_counter[0] += 1
        tx = MagicMock()
        tx.tx_id = tx_counter[0]
        tx.notes = notes
        tx.gas_limit = gas_limit
        return tx

    tx_builder.sign_transaction = sign_transaction
    return tx_builder


@pytest.fixture
def mock_token():
    """Mock ERC20 token with convert_to_raw, transfer, approve functions."""
    token = MagicMock()
    token.convert_to_raw = lambda amount: int(amount * 10**6)
    token.contract.functions.transfer.return_value = MagicMock()
    token.contract.functions.approve.return_value = MagicMock()
    return token


@pytest.fixture
def freqtrade_pair():
    """Reusable TradingPairIdentifier for Freqtrade positions."""
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
def on_chain_transfer_config():
    """FreqtradeConfig with OnChainTransferExchangeConfig."""
    return FreqtradeConfig(
        freqtrade_id="momentum-bot",
        api_url="http://localhost:8080",
        api_username="testuser",
        api_password="testpass",
        exchange_name="binance",
        reserve_currency="0xdAC17F958D2ee523a2206206994597C13D831ec7",
        exchange=OnChainTransferExchangeConfig(
            recipient_address="0xabcdef0123456789abcdef0123456789abcdef01",
            fee_tolerance=Decimal("1.0"),
            confirmation_timeout=300,
            poll_interval=5,
        ),
    )


@pytest.fixture
def aster_config():
    """FreqtradeConfig with AsterExchangeConfig."""
    return FreqtradeConfig(
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
            confirmation_timeout=300,
            poll_interval=5,
        ),
    )


@pytest.fixture
def mock_routing_state_factory(mock_freqtrade_client_factory, mock_tx_builder, mock_web3):
    """Factory for creating routing state with all mocks wired together."""
    def _create(freqtrade_id: str = "momentum-bot", initial_balance: float = 1000.0):
        client = mock_freqtrade_client_factory(initial_balance)
        return FreqtradeRoutingState(
            tx_builder=mock_tx_builder,
            web3=mock_web3,
            freqtrade_clients={freqtrade_id: client},
        ), client

    return _create


@pytest.fixture
def create_buy_trade(freqtrade_pair):
    """Factory for creating buy (deposit) trades."""
    def _create(amount: Decimal = Decimal("1000.0"), trade_id: int = 1, position_id: int = 1):
        opened_at = datetime.datetime(2024, 1, 1)
        return TradeExecution(
            trade_id=trade_id,
            position_id=position_id,
            trade_type=TradeType.rebalance,
            pair=freqtrade_pair,
            opened_at=opened_at,
            planned_quantity=amount,
            planned_price=1.0,
            planned_reserve=amount,
            reserve_currency=freqtrade_pair.quote,
        )

    return _create


@pytest.fixture
def create_sell_trade(freqtrade_pair):
    """Factory for creating sell (withdrawal) trades."""
    def _create(amount: Decimal = Decimal("500.0"), trade_id: int = 2, position_id: int = 1):
        opened_at = datetime.datetime(2024, 1, 2)
        trade = TradeExecution(
            trade_id=trade_id,
            position_id=position_id,
            trade_type=TradeType.rebalance,
            pair=freqtrade_pair,
            opened_at=opened_at,
            planned_quantity=-amount,
            planned_price=1.0,
            planned_reserve=amount,
            reserve_currency=freqtrade_pair.quote,
        )
        return trade

    return _create


@pytest.fixture
def create_position_with_deposit(freqtrade_pair):
    """Factory for creating a position with a completed deposit trade."""
    def _create(deposit_amount: Decimal = Decimal("1000.0")):
        opened_at = datetime.datetime(2024, 1, 1)
        position = TradingPosition(
            position_id=1,
            pair=freqtrade_pair,
            opened_at=opened_at,
            last_pricing_at=opened_at,
            last_token_price=1.0,
            last_reserve_price=1.0,
            reserve_currency=freqtrade_pair.quote,
        )

        trade = TradeExecution(
            trade_id=1,
            position_id=1,
            trade_type=TradeType.rebalance,
            pair=freqtrade_pair,
            opened_at=opened_at,
            planned_quantity=deposit_amount,
            planned_price=1.0,
            planned_reserve=deposit_amount,
            reserve_currency=freqtrade_pair.quote,
        )
        trade.started_at = opened_at
        trade.mark_broadcasted(opened_at)
        trade.mark_success(
            executed_at=opened_at,
            executed_price=1.0,
            executed_quantity=deposit_amount,
            executed_reserve=deposit_amount,
            lp_fees=0.0,
            native_token_price=1.0,
        )
        position.trades[1] = trade
        return position

    return _create
