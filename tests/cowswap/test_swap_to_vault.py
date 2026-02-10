"""Test opening a vault position via CoW Swap with mocked external calls.

We mock all CoW Swap API and on-chain interactions since they cannot be tested
without a live environment. The test verifies that:

- A tracked position is created in state
- Reserves are correctly deducted
- The trade is marked as successful with correct amounts
- State is synced to the store
"""

import datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest
from hexbytes import HexBytes

from tradeexecutor.ethereum.cowswap.swap_to_vault import (
    _extract_executed_amounts,
    open_vault_position_cowswap,
)
from tradeexecutor.state.identifier import (
    AssetIdentifier,
    TradingPairIdentifier,
    TradingPairKind,
)
from tradeexecutor.state.reserve import ReservePosition
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeStatus
from tradeexecutor.state.types import USDollarAmount


CHAIN_ID = 8453  # Base


@pytest.fixture()
def usdc() -> AssetIdentifier:
    return AssetIdentifier(
        chain_id=CHAIN_ID,
        address="0x833589fcd6edb6e08f4c7c32d4f71b54bda02913",
        decimals=6,
        token_symbol="USDC",
    )


@pytest.fixture()
def vault_share_token() -> AssetIdentifier:
    return AssetIdentifier(
        chain_id=CHAIN_ID,
        address="0x45aa96f0b3188d47a1dafdbefce1db6b37f58216",
        decimals=6,
        token_symbol="ipUSDC",
    )


@pytest.fixture()
def vault_pair(vault_share_token, usdc) -> TradingPairIdentifier:
    return TradingPairIdentifier(
        base=vault_share_token,
        quote=usdc,
        pool_address="0x45aa96f0b3188d47a1dafdbefce1db6b37f58216",
        exchange_address="0x0000000000000000000000000000000000000000",
        internal_id=1,
        kind=TradingPairKind.vault,
        exchange_name="IPOR USDC Lending Optimizer",
        fee=0,
    )


@pytest.fixture()
def state_with_reserves(usdc) -> State:
    """Create a state with $10,000 USDC in reserves."""
    state = State()
    ts = datetime.datetime(2025, 1, 1)

    state.portfolio.reserves[usdc.address] = ReservePosition(
        asset=usdc,
        quantity=Decimal(10_000),
        last_sync_at=ts,
        reserve_token_price=USDollarAmount(1.0),
        last_pricing_at=ts,
        initial_deposit=Decimal(10_000),
    )

    return state


@pytest.fixture()
def mock_strategy_universe(vault_pair, usdc):
    """Mock strategy universe that returns the vault pair."""
    universe = MagicMock()
    universe.get_pair_by_vault_name.return_value = vault_pair
    universe.get_reserve_asset.return_value = usdc
    return universe


@pytest.fixture()
def mock_store():
    store = MagicMock()
    return store


@pytest.fixture()
def mock_lagoon_vault():
    vault = MagicMock()
    vault.safe_address = "0xAD1241Ba37ab07fFc5d38e006747F8b92BB217D5"
    vault.vault_address = "0x45aa96f0b3188d47a1dafdbefce1db6b37f58216"
    return vault


@pytest.fixture()
def mock_hot_wallet():
    hw = MagicMock()
    hw.address = "0x1234567890123456789012345678901234567890"
    return hw


@pytest.fixture()
def mock_web3():
    web3 = MagicMock()
    web3.eth.chain_id = CHAIN_ID
    return web3


@pytest.fixture()
def mock_usdc_token_details():
    """Mock TokenDetails for USDC."""
    token = MagicMock()
    token.symbol = "USDC"
    token.address = "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913"
    token.decimals = 6
    token.convert_to_decimals.side_effect = lambda raw: Decimal(raw) / Decimal(10**6)
    return token


@pytest.fixture()
def mock_share_token_details():
    """Mock TokenDetails for vault share token."""
    token = MagicMock()
    token.symbol = "ipUSDC"
    token.address = "0x45aa96f0b3188d47a1dafdbefce1db6b37f58216"
    token.decimals = 6
    token.convert_to_decimals.side_effect = lambda raw: Decimal(raw) / Decimal(10**6)
    return token


@pytest.fixture()
def mock_quote(mock_share_token_details, mock_usdc_token_details):
    """Mock CoW Swap quote returning ~97.09 shares for 100 USDC."""
    quote = MagicMock()
    quote.get_buy_amount.return_value = Decimal("97.087378")
    quote.pformat.return_value = "mock quote"
    return quote


@pytest.fixture()
def mock_cowswap_result():
    """Mock a successful CoW Swap result."""
    result = MagicMock()
    result.get_status.return_value = "traded"
    result.order_uid = HexBytes(b"\x01" * 32)
    result.order = {
        "sellAmount": "100000000",
        "buyAmount": "97087378",
        "uid": "0x" + "01" * 32,
    }
    result.final_status_reply = {
        "type": "traded",
        "value": [
            {
                "solver": "test_solver",
                "executedAmounts": {
                    "sell": "100000000",
                    "buy": "97087378",
                },
            }
        ],
    }
    return result


def test_extract_executed_amounts_from_status(
    mock_cowswap_result,
    mock_usdc_token_details,
    mock_share_token_details,
):
    """Test extracting executed amounts from status reply."""
    sell_amount, buy_amount = _extract_executed_amounts(
        mock_cowswap_result,
        sell_token=mock_usdc_token_details,
        buy_token=mock_share_token_details,
    )

    assert sell_amount == pytest.approx(Decimal(100))
    assert buy_amount == pytest.approx(Decimal("97.087378"))


def test_extract_executed_amounts_fallback(
    mock_usdc_token_details,
    mock_share_token_details,
):
    """Test fallback to order data when status reply has no executedAmounts."""
    result = MagicMock()
    result.final_status_reply = {"type": "traded"}
    result.order = {
        "sellAmount": "50000000",
        "buyAmount": "48500000",
    }

    sell_amount, buy_amount = _extract_executed_amounts(
        result,
        sell_token=mock_usdc_token_details,
        buy_token=mock_share_token_details,
    )

    assert sell_amount == pytest.approx(Decimal(50))
    assert buy_amount == pytest.approx(Decimal("48.5"))


@patch("tradeexecutor.ethereum.cowswap.swap_to_vault.execute_presigned_cowswap_order")
@patch("tradeexecutor.ethereum.cowswap.swap_to_vault.presign_and_broadcast")
@patch("tradeexecutor.ethereum.cowswap.swap_to_vault._broadcast_tx")
@patch("tradeexecutor.ethereum.cowswap.swap_to_vault.approve_cow_swap")
@patch("tradeexecutor.ethereum.cowswap.swap_to_vault.fetch_quote")
@patch("tradeexecutor.ethereum.cowswap.swap_to_vault.fetch_erc20_details")
def test_open_vault_position_cowswap(
    mock_fetch_erc20,
    mock_fetch_quote,
    mock_approve,
    mock_broadcast,
    mock_presign,
    mock_execute,
    state_with_reserves,
    mock_strategy_universe,
    mock_store,
    mock_lagoon_vault,
    mock_hot_wallet,
    mock_web3,
    mock_usdc_token_details,
    mock_share_token_details,
    mock_quote,
    mock_cowswap_result,
    vault_pair,
    usdc,
):
    """Test the full open_vault_position_cowswap flow with mocked CoW Swap."""

    # Set up mocks
    mock_fetch_erc20.side_effect = lambda web3, addr: (
        mock_usdc_token_details if addr == usdc.address else mock_share_token_details
    )
    mock_fetch_quote.return_value = mock_quote
    mock_approve.return_value = MagicMock()
    mock_broadcast.return_value = MagicMock(hash=HexBytes(b"\xaa" * 32))
    mock_presign.return_value = {
        "uid": "0x" + "bb" * 32,
        "sellAmount": "100000000",
        "buyAmount": "97087378",
    }
    mock_execute.return_value = mock_cowswap_result

    console_context = {
        "strategy_universe": mock_strategy_universe,
        "pricing_model": MagicMock(),
        "vault": mock_lagoon_vault,
        "state": state_with_reserves,
        "web3": mock_web3,
        "store": mock_store,
        "hot_wallet": mock_hot_wallet,
    }

    # Execute
    trade = open_vault_position_cowswap(
        console_context,
        vault_name="IPOR USDC Lending Optimizer",
        amount_usd=100.0,
        max_slippage=0.01,
    )

    # Verify trade was created and marked successful
    assert trade.get_status() == TradeStatus.success
    assert trade.executed_reserve == pytest.approx(Decimal(100))
    assert trade.executed_quantity == pytest.approx(Decimal("97.087378"))
    assert trade.executed_price == pytest.approx(float(Decimal(100) / Decimal("97.087378")))

    # Verify position was created
    assert len(state_with_reserves.portfolio.open_positions) == 1
    position = list(state_with_reserves.portfolio.open_positions.values())[0]
    assert position.pair == vault_pair
    assert position.get_quantity() == pytest.approx(Decimal("97.087378"))

    # Verify reserves were deducted
    reserve_position = state_with_reserves.portfolio.reserves[usdc.address]
    assert reserve_position.quantity == pytest.approx(Decimal(10_000) - Decimal(100))

    # Verify state was synced
    mock_store.sync.assert_called_once_with(state_with_reserves)

    # Verify CoW Swap interactions happened in order
    mock_fetch_quote.assert_called_once()
    mock_approve.assert_called_once()
    mock_presign.assert_called_once()
    mock_execute.assert_called_once()
    mock_hot_wallet.sync_nonce.assert_called_once_with(mock_web3)


@patch("tradeexecutor.ethereum.cowswap.swap_to_vault.execute_presigned_cowswap_order")
@patch("tradeexecutor.ethereum.cowswap.swap_to_vault.presign_and_broadcast")
@patch("tradeexecutor.ethereum.cowswap.swap_to_vault._broadcast_tx")
@patch("tradeexecutor.ethereum.cowswap.swap_to_vault.approve_cow_swap")
@patch("tradeexecutor.ethereum.cowswap.swap_to_vault.fetch_quote")
@patch("tradeexecutor.ethereum.cowswap.swap_to_vault.fetch_erc20_details")
def test_open_vault_position_cowswap_insufficient_reserves(
    mock_fetch_erc20,
    mock_fetch_quote,
    mock_approve,
    mock_broadcast,
    mock_presign,
    mock_execute,
    state_with_reserves,
    mock_strategy_universe,
    mock_store,
    mock_lagoon_vault,
    mock_hot_wallet,
    mock_web3,
    mock_usdc_token_details,
    mock_share_token_details,
    mock_quote,
    usdc,
):
    """Test that opening a position fails when reserves are insufficient."""

    mock_fetch_erc20.side_effect = lambda web3, addr: (
        mock_usdc_token_details if addr == usdc.address else mock_share_token_details
    )
    mock_fetch_quote.return_value = mock_quote

    console_context = {
        "strategy_universe": mock_strategy_universe,
        "pricing_model": MagicMock(),
        "vault": mock_lagoon_vault,
        "state": state_with_reserves,
        "web3": mock_web3,
        "store": mock_store,
        "hot_wallet": mock_hot_wallet,
    }

    # Try to open a position worth more than reserves ($10,000)
    with pytest.raises(Exception, match="Not enough"):
        open_vault_position_cowswap(
            console_context,
            vault_name="IPOR USDC Lending Optimizer",
            amount_usd=20_000.0,
            max_slippage=0.01,
        )

    # No CoW Swap calls should have been made
    mock_approve.assert_not_called()
    mock_presign.assert_not_called()
    mock_execute.assert_not_called()

    # State should not have been synced
    mock_store.sync.assert_not_called()


@patch("tradeexecutor.ethereum.cowswap.swap_to_vault.execute_presigned_cowswap_order")
@patch("tradeexecutor.ethereum.cowswap.swap_to_vault.presign_and_broadcast")
@patch("tradeexecutor.ethereum.cowswap.swap_to_vault._broadcast_tx")
@patch("tradeexecutor.ethereum.cowswap.swap_to_vault.approve_cow_swap")
@patch("tradeexecutor.ethereum.cowswap.swap_to_vault.fetch_quote")
@patch("tradeexecutor.ethereum.cowswap.swap_to_vault.fetch_erc20_details")
def test_open_vault_position_cowswap_custom_notes(
    mock_fetch_erc20,
    mock_fetch_quote,
    mock_approve,
    mock_broadcast,
    mock_presign,
    mock_execute,
    state_with_reserves,
    mock_strategy_universe,
    mock_store,
    mock_lagoon_vault,
    mock_hot_wallet,
    mock_web3,
    mock_usdc_token_details,
    mock_share_token_details,
    mock_quote,
    mock_cowswap_result,
    usdc,
):
    """Test that custom notes are attached to the trade."""

    mock_fetch_erc20.side_effect = lambda web3, addr: (
        mock_usdc_token_details if addr == usdc.address else mock_share_token_details
    )
    mock_fetch_quote.return_value = mock_quote
    mock_approve.return_value = MagicMock()
    mock_broadcast.return_value = MagicMock(hash=HexBytes(b"\xaa" * 32))
    mock_presign.return_value = {
        "uid": "0x" + "bb" * 32,
        "sellAmount": "100000000",
        "buyAmount": "97087378",
    }
    mock_execute.return_value = mock_cowswap_result

    console_context = {
        "strategy_universe": mock_strategy_universe,
        "pricing_model": MagicMock(),
        "vault": mock_lagoon_vault,
        "state": state_with_reserves,
        "web3": mock_web3,
        "store": mock_store,
        "hot_wallet": mock_hot_wallet,
    }

    trade = open_vault_position_cowswap(
        console_context,
        vault_name="IPOR USDC Lending Optimizer",
        amount_usd=50.0,
        notes="Manual rebalance from console",
    )

    assert trade.notes.strip() == "Manual rebalance from console"
    assert trade.get_status() == TradeStatus.success
