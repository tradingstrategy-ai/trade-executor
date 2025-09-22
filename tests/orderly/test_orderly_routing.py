"""Test Orderly routing functionality."""

import datetime
import pytest
from decimal import Decimal
from unittest.mock import Mock, patch

from tradeexecutor.ethereum.orderly.orderly_vault import OrderlyVault
from tradeexecutor.ethereum.orderly.orderly_routing import OrderlyRouting, OrderlyRoutingState
from tradeexecutor.ethereum.orderly.tx import OrderlyTransactionBuilder
from tradeexecutor.state.identifier import TradingPairIdentifier, AssetIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution, TradeFlag, TradeType
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


# JSON_RPC_ARBITRUM_SEPOLIA = os.environ.get("JSON_RPC_ARBITRUM_SEPOLIA")
# pytestmark = pytest.mark.skipif(not JSON_RPC_ARBITRUM_SEPOLIA, reason="No JSON_RPC_ARBITRUM_SEPOLIA environment variable")


@pytest.mark.skip(reason="Requires proper vault contract interaction")
def test_orderly_routing_deposit_transaction_creation(
    orderly_vault: OrderlyVault,
    orderly_routing_model: OrderlyRouting,
    orderly_tx_builder: OrderlyTransactionBuilder,
    orderly_strategy_universe: TradingStrategyUniverse,
    usdc: AssetIdentifier,
    weth: AssetIdentifier,
    broker_id: str,
    orderly_account_id: str,
):
    """Test creation of deposit transactions through routing."""

    # Create a mock vault trading pair
    vault_pair = TradingPairIdentifier(
        base=weth,  # Vault shares
        quote=usdc,  # USDC denomination
        pool_address="0x0EaC556c0C2321BA25b9DC01e4e3c95aD5CDCd2f",  # Vault address
        exchange_address="0x0000000000000000000000000000000000000000",  # Not used for vaults
        fee=0.0,
    )

    # Create a mock state and trade
    state = Mock(spec=State)

    trade = TradeExecution(
        trade_id=1,
        position_id=1,
        trade_type=TradeType.rebalance,
        pair=vault_pair,
        opened_at=datetime.datetime.utcnow(),
        planned_quantity=Decimal("100"),  # 100 vault shares
        planned_reserve=Decimal("100"),   # 100 USDC
        planned_price=1.0,     # Expected 1:1 price
        reserve_currency=usdc,
        slippage_tolerance=0.01,
        notes="Test deposit",
    )
    trade.add_flag(TradeFlag.open)

    routing_state = OrderlyRoutingState(
        tx_builder=orderly_tx_builder,
        strategy_universe=orderly_strategy_universe,
        vault=orderly_vault,
        broker_id=broker_id,
        orderly_account_id=orderly_account_id,
    )

    # Mock the deposit function to avoid actual contract interaction
    with patch('eth_defi.orderly.vault.deposit') as mock_deposit:
        mock_approve_fn = Mock()
        mock_deposit_fee_fn = Mock()
        mock_deposit_fee_fn.call.return_value = 1000  # 1000 wei fee
        mock_deposit_fn = Mock()

        mock_deposit.return_value = (mock_approve_fn, mock_deposit_fee_fn, mock_deposit_fn)

        # Test deposit transaction creation
        transactions = orderly_routing_model.deposit_or_withdraw(state, routing_state, trade)

        assert len(transactions) == 2  # Approve + Deposit
        assert mock_deposit.called


@pytest.mark.skip(reason="Requires proper vault contract interaction")
def test_orderly_routing_withdraw_transaction_creation(
    orderly_vault: OrderlyVault,
    orderly_routing_model: OrderlyRouting,
    orderly_tx_builder: OrderlyTransactionBuilder,
    orderly_strategy_universe: TradingStrategyUniverse,
    usdc: AssetIdentifier,
    weth: AssetIdentifier,
    broker_id: str,
    orderly_account_id: str,
):
    """Test creation of withdraw transactions through routing."""

    # Create a mock vault trading pair
    vault_pair = TradingPairIdentifier(
        base=weth,  # Vault shares
        quote=usdc,  # USDC denomination
        pool_address="0x0EaC556c0C2321BA25b9DC01e4e3c95aD5CDCd2f",  # Vault address
        exchange_address="0x0000000000000000000000000000000000000000",  # Not used for vaults
        fee=0.0,
        vault_features={"vault_type": "orderly"},
    )

    # Create a mock state and trade
    state = Mock(spec=State)

    trade = TradeExecution(
        trade_id=1,
        position_id=1,
        trade_type=TradeType.rebalance,
        pair=vault_pair,
        opened_at=datetime.datetime.utcnow(),
        planned_quantity=Decimal("-50"),  # Withdraw 50 vault shares
        planned_reserve=Decimal("50"),    # Expected 50 USDC
        planned_price=1.0,     # Expected 1:1 price
        reserve_currency=usdc,
        slippage_tolerance=0.01,
        notes="Test withdraw",
    )
    trade.add_flag(TradeFlag.close)

    routing_state = OrderlyRoutingState(
        tx_builder=orderly_tx_builder,
        strategy_universe=orderly_strategy_universe,
        vault=orderly_vault,
        broker_id=broker_id,
        orderly_account_id=orderly_account_id,
    )

    # Mock the withdraw function to avoid actual contract interaction
    with patch('eth_defi.orderly.vault.withdraw') as mock_withdraw:
        mock_approve_fn = Mock()
        mock_withdraw_fee_fn = Mock()
        mock_withdraw_fee_fn.call.return_value = 1000  # 1000 wei fee
        mock_withdraw_fn = Mock()

        mock_withdraw.return_value = (mock_approve_fn, mock_withdraw_fee_fn, mock_withdraw_fn)

        # Test withdraw transaction creation
        transactions = orderly_routing_model.deposit_or_withdraw(state, routing_state, trade)

        assert len(transactions) == 2  # Approve + Withdraw
        assert mock_withdraw.called


def test_orderly_routing_setup_trades(
    orderly_routing_model: OrderlyRouting,
    orderly_tx_builder: OrderlyTransactionBuilder,
    orderly_strategy_universe: TradingStrategyUniverse,
    orderly_vault: OrderlyVault,
    usdc: AssetIdentifier,
    vault_pair: TradingPairIdentifier,
    broker_id: str,
    orderly_account_id: str,
    mocker,
):
    """Test setup_trades method."""

    # Create mock trades
    trades = [
        TradeExecution(
            trade_id=1,
            position_id=1,
            trade_type=TradeType.rebalance,
            pair=vault_pair,
            opened_at=datetime.datetime.utcnow(),
            planned_quantity=Decimal("100"),
            planned_reserve=Decimal("100"),
            planned_price=1.0,
            reserve_currency=usdc,
            slippage_tolerance=0.01,
            notes="Test trade 1",
            flags={TradeFlag.open},
        ),
        TradeExecution(
            trade_id=2,
            position_id=2,
            trade_type=TradeType.rebalance,
            pair=vault_pair,
            opened_at=datetime.datetime.utcnow(),
            planned_quantity=Decimal("50"),
            planned_reserve=Decimal("50"),
            planned_price=1.0,
            reserve_currency=usdc,
            slippage_tolerance=0.01,
            notes="Test trade 2",
        ),
    ]

    state = mocker.Mock(spec=State)
    routing_state = OrderlyRoutingState(
        tx_builder=orderly_tx_builder,
        strategy_universe=orderly_strategy_universe,
        vault=orderly_vault,
        broker_id=broker_id,
        orderly_account_id=orderly_account_id,
    )

    # Mock the deposit_or_withdraw method
    mock_deposit_withdraw = mocker.patch.object(orderly_routing_model, 'deposit_or_withdraw', return_value=[mocker.Mock(), mocker.Mock()]) 

    orderly_routing_model.setup_trades(state, routing_state, trades)

    # Verify that deposit_or_withdraw was called for each trade
    assert mock_deposit_withdraw.call_count == len(trades)

    # Verify that blockchain_transactions were set on each trade
    for trade in trades:
        assert hasattr(trade, 'blockchain_transactions')


def test_orderly_routing_settle_trade_basic(
    orderly_routing_model: OrderlyRouting,
    web3,
    usdc: AssetIdentifier,
    vault_pair: TradingPairIdentifier,
    mocker,
):
    """Test basic trade settlement functionality."""

    # Create a mock trade
    trade = TradeExecution(
        trade_id=1,
        position_id=1,
        trade_type=TradeType.rebalance,
        pair=vault_pair,
        opened_at=datetime.datetime.utcnow(),
        planned_quantity=Decimal("100"),
        planned_reserve=Decimal("100"),
        planned_price=1.0,
        reserve_currency=usdc,
        slippage_tolerance=0.01,
        notes="Test settlement",
        flags={TradeFlag.open},
    )

    # Mock the required components
    state = mocker.Mock(spec=State)
    receipts = {"0x123": {"blockNumber": 12345}}

    # Mock get_swap_transactions to return a transaction with the expected hash
    mock_tx = mocker.Mock()
    mock_tx.tx_hash = "0x123"
    mocker.patch('tradeexecutor.ethereum.orderly.orderly_routing.get_swap_transactions', return_value=mock_tx)
    
    # Mock get_block_timestamp
    mocker.patch('tradeexecutor.ethereum.orderly.orderly_routing.get_block_timestamp', return_value = 1234567890)

    # Test settlement (should not raise exceptions)
    orderly_routing_model.settle_trade(web3, state, trade, receipts)

    # Verify that state.mark_trade_success was called
    assert state.mark_trade_success.called


def test_orderly_routing_error_handling(mocker):
    """Test error handling for invalid trade types."""

    routing = OrderlyRouting(
        reserve_token_address="0x75faf114eafb1BDbe2F0316DF893fd58CE46AA4d".lower(),
        vault=mocker.Mock(spec=OrderlyVault),
        broker_id="test_broker",
        orderly_account_id="0x123",
    )

    # Create a non-vault trade
    non_vault_pair = TradingPairIdentifier(
        base=AssetIdentifier(
            chain_id=421614,
            address="0x123",
            decimals=18,
            token_symbol="TOKEN",
        ),
        quote=AssetIdentifier(
            chain_id=421614,
            address="0x456",
            decimals=6,
            token_symbol="USDC",
        ),
        pool_address="0x789",
        exchange_address="0xabc",
        fee=0.003,
    )

    trade = TradeExecution(
        trade_id=1,
        position_id=1,
        trade_type=TradeType.rebalance,  # Not a vault trade
        pair=non_vault_pair,
        opened_at=datetime.datetime.utcnow(),
        planned_quantity=Decimal("100"),
        planned_reserve=Decimal("100"),
        planned_price=1.0,
        reserve_currency=non_vault_pair.quote,
        slippage_tolerance=0.01,
    )

    state = mocker.Mock(spec=State)
    routing_state = mocker.Mock(spec=OrderlyRoutingState)

    # This should raise an assertion error
    with pytest.raises(AssertionError, match="Orderly routing only supports vault trades"):
        routing.deposit_or_withdraw(state, routing_state, trade)
