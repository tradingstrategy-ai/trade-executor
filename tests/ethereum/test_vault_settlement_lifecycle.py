"""Test the vault_settlement_pending lifecycle: accounting, routing, and settlement retry.

Verifies that:

- ``get_vault_settlement_pending_value()`` correctly sums planned_reserve of pending buy trades
- ``calculate_total_equity()`` includes pending vault settlement value
- ``settle_trade()`` marks async vault trades as ``vault_settlement_pending``
- ``check_and_resolve_vault_settlements()`` resolves claimable trades
- ``check_and_resolve_vault_settlements()`` handles reclaimable trades (mark failed)
- Serialisation round-trip preserves all ``other_data`` fields
"""

import datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from eth_defi.vault.deposit_redeem import AsyncVaultRequestStatus, DepositRedeemEventAnalysis
from hexbytes import HexBytes
from tradeexecutor.ethereum.vault.settlement_retry import check_and_resolve_vault_settlements
from tradeexecutor.state.blockhain_transaction import BlockchainTransaction
from tradeexecutor.state.identifier import (
    AssetIdentifier,
    TradingPairIdentifier,
    TradingPairKind,
)
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution, TradeStatus, TradeType
from tradeexecutor.ethereum.vault.settlement_estimate import refresh_vault_settlement_estimate


USDC_ARBITRUM_ADDRESS = "0xaf88d065e77c8cC2239327C5EDb3A432268e5831"
OLP_ARBITRUM_ADDRESS = "0x20d419a8e12c45f88fda7c5760bb6923cee27f98"


@pytest.fixture()
def usdc_arbitrum() -> AssetIdentifier:
    """USDC on Arbitrum."""
    return AssetIdentifier(
        chain_id=42161,
        address=USDC_ARBITRUM_ADDRESS,
        token_symbol="USDC",
        decimals=6,
    )


@pytest.fixture()
def olp_arbitrum() -> AssetIdentifier:
    """OLP share token on Arbitrum."""
    return AssetIdentifier(
        chain_id=42161,
        address=OLP_ARBITRUM_ADDRESS,
        token_symbol="oLP",
        decimals=18,
    )


@pytest.fixture()
def vault_pair(usdc_arbitrum: AssetIdentifier, olp_arbitrum: AssetIdentifier) -> TradingPairIdentifier:
    """Ostium vault trading pair."""
    return TradingPairIdentifier(
        base=olp_arbitrum,
        quote=usdc_arbitrum,
        pool_address=OLP_ARBITRUM_ADDRESS,
        exchange_address=OLP_ARBITRUM_ADDRESS,
        fee=0,
        kind=TradingPairKind.vault,
    )


def _create_vault_buy_trade(
    state: State,
    vault_pair: TradingPairIdentifier,
    reserve_asset: AssetIdentifier,
    reserve_amount: Decimal,
    ts: datetime.datetime,
) -> TradeExecution:
    """Helper to create a vault deposit (buy) trade."""
    _, trade, _ = state.create_trade(
        strategy_cycle_at=ts,
        pair=vault_pair,
        quantity=None,
        reserve=reserve_amount,
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=reserve_asset,
        reserve_currency_price=1.0,
    )
    return trade


def _create_vault_sell_trade(
    state: State,
    vault_pair: TradingPairIdentifier,
    reserve_asset: AssetIdentifier,
    quantity: Decimal,
    ts: datetime.datetime,
) -> TradeExecution:
    """Helper to create a vault redeem (sell) trade."""
    _, trade, _ = state.create_trade(
        strategy_cycle_at=ts,
        pair=vault_pair,
        quantity=-quantity,
        reserve=None,
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=reserve_asset,
        reserve_currency_price=1.0,
    )
    return trade


def test_vault_settlement_pending_value_in_equity(
    vault_pair: TradingPairIdentifier,
    usdc_arbitrum: AssetIdentifier,
):
    """Verify pending vault deposit value is included in portfolio equity.

    1. Create a state with reserves and a vault deposit (buy) trade
    2. Advance trade to vault_settlement_pending
    3. Verify reserve cash was debited and get_vault_settlement_pending_value() returns the planned_reserve
    4. Verify calculate_total_equity() includes the pending value
    """
    state = State()
    ts = datetime.datetime(2025, 1, 1)

    # 1. Create state with reserves and vault buy trade
    state.portfolio.initialise_reserves(usdc_arbitrum)
    reserve = state.portfolio.get_default_reserve_position()
    reserve.quantity = Decimal(10_000)
    reserve.reserve_token_price = 1.0

    trade = _create_vault_buy_trade(state, vault_pair, usdc_arbitrum, Decimal(500), ts)

    # 2. Advance to vault_settlement_pending
    state.start_execution(ts, trade)
    trade.mark_broadcasted(ts)

    # Add fake blockchain transactions
    approve_tx = BlockchainTransaction()
    approve_tx.tx_hash = "0xapprove"
    request_tx = BlockchainTransaction()
    request_tx.tx_hash = "0xrequest123"
    trade.blockchain_transactions = [approve_tx, request_tx]

    ticket_data = {
        "vault_address": OLP_ARBITRUM_ADDRESS,
        "vault_owner": "0xTestOwner",
        "vault_to": "0xTestOwner",
        "vault_raw_amount": 500_000_000,
        "vault_request_tx_hash": "0xrequest123",
        "vault_settlement_id": 42,
    }
    state.mark_vault_settlement_pending(ts, trade, ticket_data)

    # 3. Verify reserve cash was debited and get_vault_settlement_pending_value() returns the planned_reserve.
    assert trade.get_status() == TradeStatus.vault_settlement_pending
    assert trade.reserve_currency_allocated == pytest.approx(Decimal(500))
    assert reserve.quantity == pytest.approx(Decimal(9_500))
    pending_value = state.portfolio.get_vault_settlement_pending_value()
    assert pending_value == pytest.approx(500.0)

    # 4. Verify total equity includes pending value
    # Reserves: 10,000 - 500 (allocated) = 9,500 cash
    # Position equity: 0 (no successful trades)
    # Vault pending: 500
    total_equity = state.portfolio.calculate_total_equity()
    assert total_equity == pytest.approx(10_000.0)


def test_vault_settlement_estimate_refresh_uses_generic_ticket_method(
    vault_pair: TradingPairIdentifier,
    usdc_arbitrum: AssetIdentifier,
) -> None:
    """Check generic settlement refresh stores display ETA from an adapter ticket method.

    1. Create a settlement-pending vault deposit trade.
    2. Mock a deposit manager with a ticket-based settlement ETA method.
    3. Refresh the trade estimate and verify the persisted ISO timestamp.
    """
    state = State()
    ts = datetime.datetime(2025, 1, 1)

    # 1. Create a settlement-pending vault deposit trade.
    state.portfolio.initialise_reserves(usdc_arbitrum)
    reserve = state.portfolio.get_default_reserve_position()
    reserve.quantity = Decimal(10_000)
    reserve.reserve_token_price = 1.0
    trade = _create_vault_buy_trade(state, vault_pair, usdc_arbitrum, Decimal(500), ts)
    state.start_execution(ts, trade)
    trade.mark_broadcasted(ts)
    state.mark_vault_settlement_pending(
        ts,
        trade,
        {
            "vault_address": OLP_ARBITRUM_ADDRESS,
            "vault_owner": "0xTestOwner",
            "vault_to": "0xTestOwner",
            "vault_raw_amount": 500_000_000,
            "vault_request_tx_hash": "0xrequest123",
            "vault_settlement_id": 42,
        },
    )

    # 2. Mock a deposit manager with a ticket-based settlement ETA method.
    ticket = object()
    deposit_manager = MagicMock()
    deposit_manager.get_deposit_ticket_delay_over.return_value = datetime.datetime(2026, 6, 19, 16, 33, 48)

    # 3. Refresh the trade estimate and verify the persisted ISO timestamp.
    refresh_vault_settlement_estimate(trade, deposit_manager, ticket, "deposit")
    assert trade.other_data["vault_settlement_estimated_at"] == "2026-06-19T16:33:48"
    deposit_manager.get_deposit_ticket_delay_over.assert_called_once_with(ticket)


def test_vault_settlement_pending_sell_not_counted(
    vault_pair: TradingPairIdentifier,
    usdc_arbitrum: AssetIdentifier,
):
    """Verify pending vault sell trades are NOT counted in pending value.

    1. Create a vault sell (redeem) trade
    2. Mark as vault_settlement_pending
    3. Verify get_vault_settlement_pending_value() returns 0
    """
    state = State()
    ts = datetime.datetime(2025, 1, 1)

    # 1. Create state with reserves
    state.portfolio.initialise_reserves(usdc_arbitrum)
    reserve = state.portfolio.get_default_reserve_position()
    reserve.quantity = Decimal(10_000)
    reserve.reserve_token_price = 1.0

    trade = _create_vault_sell_trade(state, vault_pair, usdc_arbitrum, Decimal(100), ts)

    # 2. Mark as vault_settlement_pending
    state.start_execution(ts, trade)
    trade.mark_broadcasted(ts)

    request_tx = BlockchainTransaction()
    request_tx.tx_hash = "0xrequest_withdraw"
    trade.blockchain_transactions = [request_tx]

    ticket_data = {
        "vault_address": OLP_ARBITRUM_ADDRESS,
        "vault_owner": "0xTestOwner",
        "vault_to": "0xTestOwner",
        "vault_raw_amount": 100 * 10**18,
        "vault_request_tx_hash": "0xrequest_withdraw",
        "vault_settlement_id": 43,
    }
    state.mark_vault_settlement_pending(ts, trade, ticket_data)

    # 3. Verify sell trades are not counted
    assert trade.get_status() == TradeStatus.vault_settlement_pending
    pending_value = state.portfolio.get_vault_settlement_pending_value()
    assert pending_value == pytest.approx(0.0)


def test_vault_settlement_pending_metadata_stored(
    vault_pair: TradingPairIdentifier,
    usdc_arbitrum: AssetIdentifier,
):
    """Verify vault settlement metadata is stored correctly in other_data.

    1. Create a vault buy trade and mark as pending
    2. Verify all expected metadata fields are present
    3. Verify metadata survives dict round-trip (JSON serialisation proxy)
    """
    state = State()
    ts = datetime.datetime(2025, 1, 1)

    state.portfolio.initialise_reserves(usdc_arbitrum)
    reserve = state.portfolio.get_default_reserve_position()
    reserve.quantity = Decimal(10_000)
    reserve.reserve_token_price = 1.0

    # 1. Create and mark pending
    trade = _create_vault_buy_trade(state, vault_pair, usdc_arbitrum, Decimal(100), ts)
    state.start_execution(ts, trade)
    trade.mark_broadcasted(ts)
    trade.blockchain_transactions = [BlockchainTransaction()]

    ticket_data = {
        "vault_address": OLP_ARBITRUM_ADDRESS,
        "vault_owner": "0xTestOwner",
        "vault_to": "0xTestOwner",
        "vault_raw_amount": 100_000_000,
        "vault_request_tx_hash": "0xrequest123",
        "vault_settlement_id": 42,
    }
    state.mark_vault_settlement_pending(ts, trade, ticket_data)

    # 2. Verify metadata
    assert trade.other_data["vault_async_flow"] is True
    assert trade.other_data["vault_chain_id"] == 42161
    assert trade.other_data["vault_direction"] == "deposit"
    assert trade.other_data["vault_address"] == OLP_ARBITRUM_ADDRESS
    assert trade.other_data["vault_settlement_id"] == 42
    assert trade.other_data["vault_raw_amount"] == 100_000_000

    # 3. Verify round-trip (JSON compat check)
    import json
    serialised = json.dumps(trade.other_data)
    deserialised = json.loads(serialised)
    assert deserialised["vault_settlement_id"] == 42
    assert deserialised["vault_chain_id"] == 42161


def test_vault_settlement_retry_claimable(
    vault_pair: TradingPairIdentifier,
    usdc_arbitrum: AssetIdentifier,
):
    """Verify settlement retry resolves a claimable vault trade.

    1. Create a vault buy trade in vault_settlement_pending state
    2. Mock the deposit manager to return CLAIMABLE status
    3. Mock the claim transaction and analysis
    4. Call check_and_resolve_vault_settlements
    5. Verify trade is marked as success with correct amounts
    """
    from unittest.mock import MagicMock, patch
    from eth_defi.vault.deposit_redeem import AsyncVaultRequestStatus, DepositRedeemEventAnalysis
    state = State()
    ts = datetime.datetime(2025, 1, 1)

    state.portfolio.initialise_reserves(usdc_arbitrum)
    reserve = state.portfolio.get_default_reserve_position()
    reserve.quantity = Decimal(10_000)
    reserve.reserve_token_price = 1.0

    # 1. Create trade in vault_settlement_pending state
    trade = _create_vault_buy_trade(state, vault_pair, usdc_arbitrum, Decimal(100), ts)
    state.start_execution(ts, trade)
    trade.mark_broadcasted(ts)

    approve_tx = BlockchainTransaction()
    approve_tx.tx_hash = "0xapprove"
    request_tx = BlockchainTransaction()
    request_tx.tx_hash = "0xrequest"
    trade.blockchain_transactions = [approve_tx, request_tx]

    trade.other_data["vault_async_flow"] = True
    trade.other_data["vault_chain_id"] = 42161
    trade.other_data["vault_direction"] = "deposit"
    trade.other_data["vault_owner_address"] = "0xTestOwner"
    trade.other_data["vault_raw_amount"] = 100_000_000
    trade.other_data["vault_address"] = OLP_ARBITRUM_ADDRESS
    trade.other_data["vault_request_tx_hash"] = "0xrequest"
    trade.other_data["vault_settlement_id"] = 42
    trade.other_data["vault_request_tx_count"] = 2
    trade.vault_settlement_pending_at = ts

    assert trade.get_status() == TradeStatus.vault_settlement_pending

    # 2. Mock deposit manager
    mock_deposit_manager = MagicMock()
    mock_deposit_manager.get_deposit_request_status.return_value = AsyncVaultRequestStatus.claimable
    mock_deposit_manager.reconstruct_deposit_ticket.return_value = MagicMock()

    # Mock finish_deposit returns a contract function
    mock_func = MagicMock()
    mock_deposit_manager.finish_deposit.return_value = mock_func

    # Mock analysis
    mock_analysis = MagicMock(spec=DepositRedeemEventAnalysis)
    mock_analysis.denomination_amount = Decimal(100)
    mock_analysis.share_count = Decimal("96.15")
    mock_deposit_manager.analyse_deposit.return_value = mock_analysis

    # 3. Mock vault and web3
    mock_vault = MagicMock()
    mock_vault.get_deposit_manager.return_value = mock_deposit_manager
    mock_vault.vault_contract = MagicMock()

    mock_web3 = MagicMock()
    mock_receipt = {"status": 1, "transactionHash": b"\x00" * 32, "blockNumber": 100}
    mock_web3.eth.send_raw_transaction.return_value = b"\x00" * 32

    mock_execution_model = MagicMock()
    mock_execution_model.web3 = mock_web3

    # Mock tx_builder
    mock_tx = BlockchainTransaction()
    mock_tx.tx_hash = "0x" + "ab" * 32
    mock_tx.signed_bytes = b"\xab" * 32
    mock_tx.other = {}
    mock_execution_model.tx_builder.sign_transaction.return_value = mock_tx

    # 4. Patch get_vault_for_pair
    with (
        patch("tradeexecutor.ethereum.vault.settlement_retry.get_vault_for_pair", return_value=mock_vault),
        patch("tradeexecutor.ethereum.vault.settlement_retry.wait_for_transaction_receipt_robust", return_value=mock_receipt),
    ):
        resolved = check_and_resolve_vault_settlements(
            state=state,
            execution_model=mock_execution_model,
        )

    # 5. Verify trade marked as success
    assert len(resolved) == 1
    assert trade.get_status() == TradeStatus.success
    assert trade.vault_settlement_pending_at is None
    assert trade.executed_reserve == pytest.approx(Decimal(100))
    assert trade.executed_quantity == pytest.approx(Decimal("96.15"))


def test_vault_settlement_retry_repairs_legacy_missing_reserve_debit(
    vault_pair: TradingPairIdentifier,
    usdc_arbitrum: AssetIdentifier,
):
    """Verify settlement retry repairs older pending deposit state with missing reserve debit.

    1. Create a vault deposit that is pending settlement.
    2. Simulate a legacy/corrupt state where reserve cash was put back and allocation was lost.
    3. Mock the deposit manager to return CLAIMABLE status and claim analysis.
    4. Resolve the settlement retry.
    5. Verify the reserve debit exists after claim, so cash is not overstated.

    The legacy mutation is mocked because current ``mark_vault_settlement_pending()``
    rejects this state; the test reproduces a persisted state from before that guard.
    """
    state = State()
    ts = datetime.datetime(2025, 1, 1)

    # 1. Create a vault deposit that is pending settlement.
    state.portfolio.initialise_reserves(usdc_arbitrum)
    reserve = state.portfolio.get_default_reserve_position()
    reserve.quantity = Decimal(10_000)
    reserve.reserve_token_price = 1.0
    trade = _create_vault_buy_trade(state, vault_pair, usdc_arbitrum, Decimal(100), ts)
    state.start_execution(ts, trade)
    trade.mark_broadcasted(ts)

    approve_tx = BlockchainTransaction()
    approve_tx.tx_hash = "0xapprove"
    request_tx = BlockchainTransaction()
    request_tx.tx_hash = "0xrequest"
    trade.blockchain_transactions = [approve_tx, request_tx]
    state.mark_vault_settlement_pending(
        ts,
        trade,
        {
            "vault_async_flow": True,
            "vault_chain_id": 42161,
            "vault_direction": "deposit",
            "vault_owner_address": "0xTestOwner",
            "vault_raw_amount": 100_000_000,
            "vault_address": OLP_ARBITRUM_ADDRESS,
            "vault_request_tx_hash": "0xrequest",
            "vault_settlement_id": 42,
            "vault_request_tx_count": 2,
        },
    )

    # 2. Simulate a legacy/corrupt state where reserve cash was put back and allocation was lost.
    reserve.quantity += Decimal(100)
    trade.reserve_currency_allocated = None
    assert reserve.quantity == pytest.approx(Decimal(10_000))

    # 3. Mock the deposit manager to return CLAIMABLE status and claim analysis.
    mock_deposit_manager = MagicMock()
    mock_deposit_manager.get_deposit_request_status.return_value = AsyncVaultRequestStatus.claimable
    mock_deposit_manager.reconstruct_deposit_ticket.return_value = MagicMock()
    mock_deposit_manager.finish_deposit.return_value = MagicMock()
    mock_analysis = MagicMock(spec=DepositRedeemEventAnalysis)
    mock_analysis.denomination_amount = Decimal(100)
    mock_analysis.share_count = Decimal("96.15")
    mock_deposit_manager.analyse_deposit.return_value = mock_analysis

    mock_vault = MagicMock()
    mock_vault.get_deposit_manager.return_value = mock_deposit_manager
    mock_vault.vault_contract = MagicMock()

    mock_web3 = MagicMock()
    mock_receipt = {"status": 1, "transactionHash": b"\x00" * 32, "blockNumber": 100}
    mock_web3.eth.send_raw_transaction.return_value = b"\x00" * 32

    mock_execution_model = MagicMock()
    mock_execution_model.web3 = mock_web3
    mock_tx = BlockchainTransaction()
    mock_tx.tx_hash = "0x" + "ab" * 32
    mock_tx.signed_bytes = b"\xab" * 32
    mock_tx.other = {}
    mock_execution_model.tx_builder.sign_transaction.return_value = mock_tx

    # 4. Resolve the settlement retry.
    with (
        patch("tradeexecutor.ethereum.vault.settlement_retry.get_vault_for_pair", return_value=mock_vault),
        patch("tradeexecutor.ethereum.vault.settlement_retry.wait_for_transaction_receipt_robust", return_value=mock_receipt),
    ):
        resolved = check_and_resolve_vault_settlements(
            state=state,
            execution_model=mock_execution_model,
        )

    # 5. Verify the reserve debit exists after claim, so cash is not overstated.
    assert len(resolved) == 1
    assert trade.get_status() == TradeStatus.success
    assert trade.reserve_currency_allocated == pytest.approx(Decimal(0))
    assert reserve.quantity == pytest.approx(Decimal(9_900))
    assert trade.executed_reserve == pytest.approx(Decimal(100))
    assert trade.executed_quantity == pytest.approx(Decimal("96.15"))


def test_vault_settlement_retry_repairs_legacy_missing_bridge_allocation(
    usdc_arbitrum: AssetIdentifier,
):
    """Verify settlement retry repairs older satellite vault deposits with missing bridge allocation.

    1. Create a CCTP bridge position from Arbitrum to Base.
    2. Create a Base vault deposit funded from the bridge position.
    3. Simulate a legacy/corrupt state where bridge allocation was lost.
    4. Resolve the settlement retry.
    5. Verify partial refunds return to bridge capital and home reserves are not credited.

    The legacy mutation is mocked because current ``mark_vault_settlement_pending()``
    rejects this state; the test reproduces a persisted state from before that guard.
    """
    ts = datetime.datetime(2025, 1, 1)
    state = State()
    usdc_base = AssetIdentifier(
        chain_id=8453,
        address="0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
        token_symbol="USDC",
        decimals=6,
    )
    base_vault_asset = AssetIdentifier(
        chain_id=8453,
        address="0x1111111111111111111111111111111111111111",
        token_symbol="vUSDC",
        decimals=18,
    )
    cctp_pair = TradingPairIdentifier(
        base=usdc_base,
        quote=usdc_arbitrum,
        pool_address="0x28b5a0e9C621a5BadaA536219b3a228C8168cf5d",
        exchange_address="0x28b5a0e9C621a5BadaA536219b3a228C8168cf5d",
        fee=0,
        kind=TradingPairKind.cctp_bridge,
        other_data={"bridge_protocol": "cctp"},
    )
    base_vault_pair = TradingPairIdentifier(
        base=base_vault_asset,
        quote=usdc_base,
        pool_address=base_vault_asset.address,
        exchange_address=base_vault_asset.address,
        fee=0,
        kind=TradingPairKind.vault,
    )

    # 1. Create a CCTP bridge position from Arbitrum to Base.
    state.portfolio.initialise_reserves(usdc_arbitrum)
    reserve = state.portfolio.get_default_reserve_position()
    reserve.quantity = Decimal(10_000)
    reserve.reserve_token_price = 1.0
    _, bridge_trade, _ = state.create_trade(
        strategy_cycle_at=ts,
        pair=cctp_pair,
        quantity=None,
        reserve=Decimal(100),
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc_arbitrum,
        reserve_currency_price=1.0,
    )
    state.start_execution(ts, bridge_trade)
    bridge_trade.mark_broadcasted(ts)
    state.mark_trade_success(
        executed_at=ts,
        trade=bridge_trade,
        executed_price=1.0,
        executed_amount=Decimal(100),
        executed_reserve=Decimal(100),
        lp_fees=0,
        native_token_price=0,
    )
    bridge_position = state.portfolio.get_bridge_position_for_chain(8453)
    assert bridge_position is not None
    assert reserve.quantity == pytest.approx(Decimal(9_900))

    # 2. Create a Base vault deposit funded from the bridge position.
    trade = _create_vault_buy_trade(state, base_vault_pair, usdc_arbitrum, Decimal(100), ts)
    state.start_execution(ts, trade)
    trade.mark_broadcasted(ts)
    request_tx = BlockchainTransaction()
    request_tx.tx_hash = "0xrequest"
    trade.blockchain_transactions = [request_tx]
    state.mark_vault_settlement_pending(
        ts,
        trade,
        {
            "vault_async_flow": True,
            "vault_chain_id": 8453,
            "vault_direction": "deposit",
            "vault_owner_address": "0xTestOwner",
            "vault_raw_amount": 100_000_000,
            "vault_address": base_vault_asset.address,
            "vault_request_tx_hash": "0xrequest",
            "vault_settlement_id": 42,
            "vault_request_tx_count": 1,
        },
    )
    assert bridge_position.bridge_capital_allocated == pytest.approx(Decimal(100))

    # 3. Simulate a legacy/corrupt state where bridge allocation was lost.
    bridge_position.bridge_capital_allocated = Decimal(0)
    trade.bridge_currency_allocated = None

    mock_deposit_manager = MagicMock()
    mock_deposit_manager.get_deposit_request_status.return_value = AsyncVaultRequestStatus.claimable
    mock_deposit_manager.reconstruct_deposit_ticket.return_value = MagicMock()
    mock_deposit_manager.finish_deposit.return_value = MagicMock()
    mock_analysis = MagicMock(spec=DepositRedeemEventAnalysis)
    mock_analysis.denomination_amount = Decimal(80)
    mock_analysis.share_count = Decimal("76.92")
    mock_deposit_manager.analyse_deposit.return_value = mock_analysis
    mock_vault = MagicMock()
    mock_vault.get_deposit_manager.return_value = mock_deposit_manager
    mock_vault.vault_contract = MagicMock()
    mock_web3 = MagicMock()
    mock_receipt = {"status": 1, "transactionHash": b"\x00" * 32, "blockNumber": 100}
    mock_web3.eth.send_raw_transaction.return_value = b"\x00" * 32
    mock_execution_model = MagicMock()
    mock_execution_model.web3 = mock_web3
    mock_tx = BlockchainTransaction()
    mock_tx.tx_hash = "0x" + "ab" * 32
    mock_tx.signed_bytes = b"\xab" * 32
    mock_tx.other = {}
    mock_execution_model.tx_builder.sign_transaction.return_value = mock_tx

    # 4. Resolve the settlement retry.
    with (
        patch("tradeexecutor.ethereum.vault.settlement_retry.get_vault_for_pair", return_value=mock_vault),
        patch("tradeexecutor.ethereum.vault.settlement_retry.wait_for_transaction_receipt_robust", return_value=mock_receipt),
    ):
        resolved = check_and_resolve_vault_settlements(
            state=state,
            execution_model=mock_execution_model,
        )

    # 5. Verify partial refunds return to bridge capital and home reserves are not credited.
    assert len(resolved) == 1
    assert trade.get_status() == TradeStatus.success
    assert reserve.quantity == pytest.approx(Decimal(9_900))
    assert bridge_position.bridge_capital_allocated == pytest.approx(Decimal(80))
    assert trade.bridge_currency_allocated == pytest.approx(Decimal(80))
    assert trade.executed_reserve == pytest.approx(Decimal(80))
    assert trade.executed_quantity == pytest.approx(Decimal("76.92"))


def test_vault_settlement_retry_rebroadcast_uses_robust_receipt_wait(
    vault_pair: TradingPairIdentifier,
    usdc_arbitrum: AssetIdentifier,
):
    """Verify a stored vault claim transaction is rebroadcast using the robust receipt wait.

    1. Create a vault buy trade with an existing post-request claim transaction
    2. Mock the first receipt lookup to miss the transaction
    3. Mock the robust receipt wait and claim analysis
    4. Call check_and_resolve_vault_settlements
    5. Verify the stored transaction is rebroadcast and the trade succeeds
    """
    state = State()
    ts = datetime.datetime(2025, 1, 1)

    state.portfolio.initialise_reserves(usdc_arbitrum)
    reserve = state.portfolio.get_default_reserve_position()
    reserve.quantity = Decimal(10_000)
    reserve.reserve_token_price = 1.0

    # 1. Create a vault buy trade with an existing post-request claim transaction
    trade = _create_vault_buy_trade(state, vault_pair, usdc_arbitrum, Decimal(100), ts)
    state.start_execution(ts, trade)
    trade.mark_broadcasted(ts)

    approve_tx = BlockchainTransaction()
    approve_tx.tx_hash = "0xapprove"
    request_tx = BlockchainTransaction()
    request_tx.tx_hash = "0xrequest"
    claim_tx = BlockchainTransaction()
    claim_tx.tx_hash = "0x" + "ef" * 32
    claim_tx.signed_bytes = b"\xef" * 32
    claim_tx.other["vault_settlement_action"] = "claim"
    trade.blockchain_transactions = [approve_tx, request_tx, claim_tx]

    trade.other_data["vault_async_flow"] = True
    trade.other_data["vault_chain_id"] = 42161
    trade.other_data["vault_direction"] = "deposit"
    trade.other_data["vault_owner_address"] = "0xTestOwner"
    trade.other_data["vault_raw_amount"] = 100_000_000
    trade.other_data["vault_address"] = OLP_ARBITRUM_ADDRESS
    trade.other_data["vault_request_tx_hash"] = "0xrequest"
    trade.other_data["vault_settlement_id"] = 42
    trade.other_data["vault_request_tx_count"] = 2
    trade.vault_settlement_pending_at = ts

    mock_deposit_manager = MagicMock()
    mock_deposit_manager.reconstruct_deposit_ticket.return_value = MagicMock()

    mock_analysis = MagicMock(spec=DepositRedeemEventAnalysis)
    mock_analysis.denomination_amount = Decimal(100)
    mock_analysis.share_count = Decimal("96.15")
    mock_deposit_manager.analyse_deposit.return_value = mock_analysis

    mock_vault = MagicMock()
    mock_vault.get_deposit_manager.return_value = mock_deposit_manager

    mock_web3 = MagicMock()
    mock_web3.eth.get_transaction_receipt.side_effect = RuntimeError("missing receipt")
    mock_receipt = {"status": 1, "transactionHash": b"\x00" * 32, "blockNumber": 100}

    mock_execution_model = MagicMock()
    mock_execution_model.web3 = mock_web3

    # 2. Mock the first receipt lookup to miss the transaction
    # 3. Mock the robust receipt wait and claim analysis
    with (
        patch("tradeexecutor.ethereum.vault.settlement_retry.get_vault_for_pair", return_value=mock_vault),
        patch("tradeexecutor.ethereum.vault.settlement_retry.wait_for_transaction_receipt_robust", return_value=mock_receipt) as wait_receipt,
    ):
        # 4. Call check_and_resolve_vault_settlements
        resolved = check_and_resolve_vault_settlements(
            state=state,
            execution_model=mock_execution_model,
        )

    # 5. Verify the stored transaction is rebroadcast and the trade succeeds
    assert len(resolved) == 1
    assert trade.get_status() == TradeStatus.success
    mock_web3.eth.send_raw_transaction.assert_called_once_with(HexBytes(claim_tx.signed_bytes))
    wait_receipt.assert_called_once()


def test_vault_settlement_retry_recovers_claim_using_robust_receipt_wait(
    vault_pair: TradingPairIdentifier,
    usdc_arbitrum: AssetIdentifier,
):
    """Verify an already-completed vault claim is recovered using the robust receipt wait.

    1. Create a vault buy trade whose request status is already consumed
    2. Mock completed claim discovery for the consumed request
    3. Mock the robust receipt wait and recovered transaction creation
    4. Call check_and_resolve_vault_settlements
    5. Verify the recovered transaction is stored and the trade succeeds
    """
    from tradeexecutor.ethereum.vault.settlement_retry import check_and_resolve_vault_settlements

    state = State()
    ts = datetime.datetime(2025, 1, 1)

    state.portfolio.initialise_reserves(usdc_arbitrum)
    reserve = state.portfolio.get_default_reserve_position()
    reserve.quantity = Decimal(10_000)
    reserve.reserve_token_price = 1.0

    # 1. Create a vault buy trade whose request status is already consumed
    trade = _create_vault_buy_trade(state, vault_pair, usdc_arbitrum, Decimal(100), ts)
    state.start_execution(ts, trade)
    trade.mark_broadcasted(ts)

    approve_tx = BlockchainTransaction()
    approve_tx.tx_hash = "0xapprove"
    request_tx = BlockchainTransaction()
    request_tx.tx_hash = "0xrequest"
    trade.blockchain_transactions = [approve_tx, request_tx]

    trade.other_data["vault_async_flow"] = True
    trade.other_data["vault_chain_id"] = 42161
    trade.other_data["vault_direction"] = "deposit"
    trade.other_data["vault_owner_address"] = "0xTestOwner"
    trade.other_data["vault_raw_amount"] = 100_000_000
    trade.other_data["vault_address"] = OLP_ARBITRUM_ADDRESS
    trade.other_data["vault_request_tx_hash"] = "0xrequest"
    trade.other_data["vault_settlement_id"] = 42
    trade.other_data["vault_request_tx_count"] = 2
    trade.vault_settlement_pending_at = ts

    mock_deposit_manager = MagicMock()
    mock_deposit_manager.get_deposit_request_status.return_value = AsyncVaultRequestStatus.none
    mock_deposit_manager.reconstruct_deposit_ticket.return_value = MagicMock()
    mock_deposit_manager.finish_deposit.return_value = MagicMock()

    mock_analysis = MagicMock(spec=DepositRedeemEventAnalysis)
    mock_analysis.denomination_amount = Decimal(100)
    mock_analysis.share_count = Decimal("96.15")
    mock_deposit_manager.analyse_deposit.return_value = mock_analysis

    mock_vault = MagicMock()
    mock_vault.get_deposit_manager.return_value = mock_deposit_manager

    mock_web3 = MagicMock()
    mock_execution_model = MagicMock()
    mock_execution_model.web3 = mock_web3

    recovered_tx_hash = HexBytes("0x" + "12" * 32)
    recovered_tx = BlockchainTransaction()
    recovered_tx.tx_hash = recovered_tx_hash.hex()
    recovered_tx.other["vault_settlement_action"] = "claim"
    mock_receipt = {"status": 1, "transactionHash": recovered_tx_hash, "blockNumber": 100}

    # 2. Mock completed claim discovery for the consumed request
    # 3. Mock the robust receipt wait and recovered transaction creation
    with (
        patch("tradeexecutor.ethereum.vault.settlement_retry.get_vault_for_pair", return_value=mock_vault),
        patch("tradeexecutor.ethereum.vault.settlement_retry._find_already_completed_claim_tx_hash", return_value=recovered_tx_hash),
        patch("tradeexecutor.ethereum.vault.settlement_retry._create_recovered_claim_transaction", return_value=recovered_tx),
        patch("tradeexecutor.ethereum.vault.settlement_retry.wait_for_transaction_receipt_robust", return_value=mock_receipt) as wait_receipt,
    ):
        # 4. Call check_and_resolve_vault_settlements
        resolved = check_and_resolve_vault_settlements(
            state=state,
            execution_model=mock_execution_model,
        )

    # 5. Verify the recovered transaction is stored and the trade succeeds
    assert len(resolved) == 1
    assert trade.get_status() == TradeStatus.success
    assert trade.blockchain_transactions[-1] is recovered_tx
    wait_receipt.assert_called_once()


def test_vault_settlement_retry_pending_skipped(
    vault_pair: TradingPairIdentifier,
    usdc_arbitrum: AssetIdentifier,
):
    """Verify that trades still pending settlement are skipped (not resolved).

    1. Create a vault buy trade in vault_settlement_pending state
    2. Mock the deposit manager to return PENDING status
    3. Call check_and_resolve_vault_settlements
    4. Verify trade remains in vault_settlement_pending status
    """
    from unittest.mock import MagicMock, patch
    from eth_defi.vault.deposit_redeem import AsyncVaultRequestStatus
    from tradeexecutor.ethereum.vault.settlement_retry import check_and_resolve_vault_settlements

    state = State()
    ts = datetime.datetime(2025, 1, 1)

    state.portfolio.initialise_reserves(usdc_arbitrum)
    reserve = state.portfolio.get_default_reserve_position()
    reserve.quantity = Decimal(10_000)
    reserve.reserve_token_price = 1.0

    # 1. Create trade
    trade = _create_vault_buy_trade(state, vault_pair, usdc_arbitrum, Decimal(100), ts)
    state.start_execution(ts, trade)
    trade.mark_broadcasted(ts)

    approve_tx = BlockchainTransaction()
    approve_tx.tx_hash = "0xapprove"
    request_tx = BlockchainTransaction()
    request_tx.tx_hash = "0xrequest"
    trade.blockchain_transactions = [approve_tx, request_tx]

    trade.other_data["vault_async_flow"] = True
    trade.other_data["vault_chain_id"] = 42161
    trade.other_data["vault_direction"] = "deposit"
    trade.other_data["vault_owner_address"] = "0xTestOwner"
    trade.other_data["vault_raw_amount"] = 100_000_000
    trade.other_data["vault_address"] = OLP_ARBITRUM_ADDRESS
    trade.other_data["vault_request_tx_hash"] = "0xrequest"
    trade.other_data["vault_settlement_id"] = 42
    trade.other_data["vault_request_tx_count"] = 2
    trade.vault_settlement_pending_at = ts

    # 2. Mock deposit manager returning PENDING
    mock_deposit_manager = MagicMock()
    mock_deposit_manager.get_deposit_request_status.return_value = AsyncVaultRequestStatus.pending
    mock_deposit_manager.reconstruct_deposit_ticket.return_value = MagicMock()

    mock_vault = MagicMock()
    mock_vault.get_deposit_manager.return_value = mock_deposit_manager

    mock_execution_model = MagicMock()
    mock_execution_model.web3 = MagicMock()

    # 3. Call retry
    with patch("tradeexecutor.ethereum.vault.settlement_retry.get_vault_for_pair", return_value=mock_vault):
        resolved = check_and_resolve_vault_settlements(
            state=state,
            execution_model=mock_execution_model,
        )

    # 4. Verify trade still pending
    assert len(resolved) == 0
    assert trade.get_status() == TradeStatus.vault_settlement_pending
    assert trade.vault_settlement_pending_at == ts


def test_vault_settlement_retry_reclaimable(
    vault_pair: TradingPairIdentifier,
    usdc_arbitrum: AssetIdentifier,
):
    """Verify reclaimable vault trades are marked as failed with reserves restored.

    1. Create a vault buy trade in vault_settlement_pending state
    2. Mock the deposit manager to return RECLAIMABLE status
    3. Mock the reclaim transaction
    4. Call check_and_resolve_vault_settlements
    5. Verify trade is marked as failed
    6. Verify reserves are restored
    """
    from unittest.mock import MagicMock, patch
    from eth_defi.vault.deposit_redeem import AsyncVaultRequestStatus
    from tradeexecutor.ethereum.vault.settlement_retry import check_and_resolve_vault_settlements

    state = State()
    ts = datetime.datetime(2025, 1, 1)

    state.portfolio.initialise_reserves(usdc_arbitrum)
    reserve = state.portfolio.get_default_reserve_position()
    reserve.quantity = Decimal(10_000)
    reserve.reserve_token_price = 1.0

    # 1. Create trade
    trade = _create_vault_buy_trade(state, vault_pair, usdc_arbitrum, Decimal(100), ts)
    state.start_execution(ts, trade)
    trade.mark_broadcasted(ts)

    approve_tx = BlockchainTransaction()
    approve_tx.tx_hash = "0xapprove"
    request_tx = BlockchainTransaction()
    request_tx.tx_hash = "0xrequest"
    trade.blockchain_transactions = [approve_tx, request_tx]

    trade.other_data["vault_async_flow"] = True
    trade.other_data["vault_chain_id"] = 42161
    trade.other_data["vault_direction"] = "deposit"
    trade.other_data["vault_owner_address"] = "0xTestOwner"
    trade.other_data["vault_raw_amount"] = 100_000_000
    trade.other_data["vault_address"] = OLP_ARBITRUM_ADDRESS
    trade.other_data["vault_request_tx_hash"] = "0xrequest"
    trade.other_data["vault_settlement_id"] = 42
    trade.other_data["vault_request_tx_count"] = 2
    trade.vault_settlement_pending_at = ts

    # Reserve was 10,000 but 100 allocated to trade
    cash_before = float(state.portfolio.get_cash())

    # 2. Mock deposit manager returning RECLAIMABLE
    mock_deposit_manager = MagicMock()
    mock_deposit_manager.get_deposit_request_status.return_value = AsyncVaultRequestStatus.reclaimable
    mock_deposit_manager.reconstruct_deposit_ticket.return_value = MagicMock()

    mock_reclaim_func = MagicMock()
    mock_deposit_manager.reclaim_deposit.return_value = mock_reclaim_func

    mock_vault = MagicMock()
    mock_vault.get_deposit_manager.return_value = mock_deposit_manager
    mock_vault.vault_contract = MagicMock()

    mock_web3 = MagicMock()
    mock_receipt = {"status": 1, "transactionHash": b"\x00" * 32, "blockNumber": 100}
    mock_web3.eth.send_raw_transaction.return_value = b"\x00" * 32

    mock_execution_model = MagicMock()
    mock_execution_model.web3 = mock_web3

    mock_tx = BlockchainTransaction()
    mock_tx.tx_hash = "0x" + "cd" * 32
    mock_tx.signed_bytes = b"\xcd" * 32
    mock_tx.other = {}
    mock_execution_model.tx_builder.sign_transaction.return_value = mock_tx

    # 4. Call retry
    with (
        patch("tradeexecutor.ethereum.vault.settlement_retry.get_vault_for_pair", return_value=mock_vault),
        patch("tradeexecutor.ethereum.vault.settlement_retry.wait_for_transaction_receipt_robust", return_value=mock_receipt),
    ):
        resolved = check_and_resolve_vault_settlements(
            state=state,
            execution_model=mock_execution_model,
        )

    # 5. Verify trade failed
    assert len(resolved) == 1
    assert trade.get_status() == TradeStatus.failed
    assert trade.vault_settlement_pending_at is None

    # 6. Verify reserves restored (mark_trade_failed calls adjust_reserves for buys)
    cash_after = float(state.portfolio.get_cash())
    assert cash_after == pytest.approx(cash_before + 100.0)


def test_is_swap_function_includes_async_selectors():
    """Verify that requestDeposit and requestWithdraw are recognised as swap functions.

    1. Import is_swap_function
    2. Verify requestDeposit returns True
    3. Verify requestWithdraw returns True
    4. Verify unknown functions return False
    """
    from tradeexecutor.ethereum.swap import is_swap_function

    # 2. Async deposit request
    assert is_swap_function("requestDeposit") is True

    # 3. Async withdraw request
    assert is_swap_function("requestWithdraw") is True

    # 4. Unknown
    assert is_swap_function("unknownFunction") is False
