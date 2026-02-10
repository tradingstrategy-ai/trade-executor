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

from tradeexecutor.ethereum.cowswap.swap_to_vault import open_vault_position_cowswap
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
    vault_pair,
    usdc,
):
    """Test the full open_vault_position_cowswap happy path with mocked CoW Swap."""

    # Mock token details
    mock_usdc = MagicMock()
    mock_usdc.symbol = "USDC"
    mock_usdc.address = usdc.address
    mock_usdc.decimals = 6
    mock_usdc.convert_to_decimals.side_effect = lambda raw: Decimal(raw) / Decimal(10**6)

    mock_share = MagicMock()
    mock_share.symbol = "ipUSDC"
    mock_share.address = vault_pair.base.address
    mock_share.decimals = 6
    mock_share.convert_to_decimals.side_effect = lambda raw: Decimal(raw) / Decimal(10**6)

    mock_fetch_erc20.side_effect = lambda web3, addr: (
        mock_usdc if addr == usdc.address else mock_share
    )

    # Mock quote: ~97.09 shares for 100 USDC
    mock_quote = MagicMock()
    mock_quote.get_buy_amount.return_value = Decimal("97.087378")
    mock_quote.pformat.return_value = "mock quote"
    mock_fetch_quote.return_value = mock_quote

    # Mock CoW Swap on-chain interactions
    mock_approve.return_value = MagicMock()
    mock_broadcast.return_value = MagicMock(hash=HexBytes(b"\xaa" * 32))
    mock_presign.return_value = {
        "uid": "0x" + "bb" * 32,
        "sellAmount": "100000000",
        "buyAmount": "97087378",
    }

    # Mock successful CoW Swap result
    mock_result = MagicMock()
    mock_result.get_status.return_value = "traded"
    mock_result.order_uid = HexBytes(b"\x01" * 32)
    mock_result.order = {"sellAmount": "100000000", "buyAmount": "97087378"}
    mock_result.final_status_reply = {
        "type": "traded",
        "value": [{
            "solver": "test_solver",
            "executedAmounts": {"sell": "100000000", "buy": "97087378"},
        }],
    }
    mock_execute.return_value = mock_result

    # Mock console context objects
    mock_store = MagicMock()
    mock_hot_wallet = MagicMock()
    mock_hot_wallet.address = "0x1234567890123456789012345678901234567890"
    mock_web3 = MagicMock()
    mock_web3.eth.chain_id = CHAIN_ID

    mock_universe = MagicMock()
    mock_universe.get_pair_by_vault_name.return_value = vault_pair
    mock_universe.get_reserve_asset.return_value = usdc

    console_context = {
        "strategy_universe": mock_universe,
        "pricing_model": MagicMock(),
        "vault": MagicMock(),
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

    # Verify CoW Swap interactions happened
    mock_fetch_quote.assert_called_once()
    mock_approve.assert_called_once()
    mock_presign.assert_called_once()
    mock_execute.assert_called_once()
