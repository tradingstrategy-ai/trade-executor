"""Test the cctp_in_transit lifecycle: valuation, executor halt, and startup retry.

Verifies that:

- ``get_in_transit_value()`` correctly sums planned_reserve of in-transit trades
- ``calculate_total_equity()`` includes in-transit value
- ``calculate_total_equity_chain()`` attributes in-transit value to the
  destination chain
- The sequential executor halts and expires remaining planned trades when
  a CCTP bridge trade enters cctp_in_transit status
- Orphan positions created by expired trades are cleaned up
- ``check_and_retry_cctp_in_transit()`` resolves in-transit trades on startup
"""

import datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from tradeexecutor.ethereum.cctp.planner import inject_cctp_bridge_trades
from tradeexecutor.state.blockhain_transaction import BlockchainTransaction
from tradeexecutor.state.identifier import (
    AssetIdentifier,
    TradingPairIdentifier,
    TradingPairKind,
)
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution, TradeStatus, TradeType
from tradeexecutor.strategy.execution_model import ExecutionHaltableIssue


USDC_ARBITRUM_ADDRESS = "0xaf88d065e77c8cC2239327C5EDb3A432268e5831"
USDC_BASE_ADDRESS = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"


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
def usdc_base() -> AssetIdentifier:
    """USDC on Base."""
    return AssetIdentifier(
        chain_id=8453,
        address=USDC_BASE_ADDRESS,
        token_symbol="USDC",
        decimals=6,
    )


@pytest.fixture()
def cctp_pair(usdc_arbitrum: AssetIdentifier, usdc_base: AssetIdentifier) -> TradingPairIdentifier:
    """CCTP bridge pair: Arbitrum -> Base."""
    return TradingPairIdentifier(
        base=usdc_base,
        quote=usdc_arbitrum,
        pool_address="0x28b5a0e9C621a5BadaA536219b3a228C8168cf5d",
        exchange_address="0x28b5a0e9C621a5BadaA536219b3a228C8168cf5d",
        fee=0,
        kind=TradingPairKind.cctp_bridge,
        other_data={"bridge_protocol": "cctp"},
    )


def _create_bridge_trade(
    state: State,
    cctp_pair: TradingPairIdentifier,
    reserve_asset: AssetIdentifier,
    reserve_amount: Decimal,
    ts: datetime.datetime,
) -> TradeExecution:
    """Helper to create a bridge-out (buy) trade."""
    _, trade, _ = state.create_trade(
        strategy_cycle_at=ts,
        pair=cctp_pair,
        quantity=None,
        reserve=reserve_amount,
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=reserve_asset,
        reserve_currency_price=1.0,
    )
    return trade


def test_in_transit_value_included_in_equity(
    cctp_pair: TradingPairIdentifier,
    usdc_arbitrum: AssetIdentifier,
):
    """Verify that in-transit value is included in portfolio equity calculations.

    1. Create a state with reserves and a bridge-out (buy) trade
    2. Advance trade to broadcasted, then mark it cctp_in_transit
    3. Verify get_in_transit_value() returns the planned_reserve amount
    4. Verify calculate_total_equity() includes the in-transit value
    5. Verify calculate_total_equity_chain() attributes in-transit value
       to the destination chain
    """
    state = State()
    ts = datetime.datetime(2025, 1, 1)

    # 1. Create a state with reserves and a bridge-out (buy) trade
    state.portfolio.initialise_reserves(usdc_arbitrum)
    reserve = state.portfolio.get_default_reserve_position()
    reserve.quantity = Decimal(10_000)
    reserve.reserve_token_price = 1.0

    trade = _create_bridge_trade(state, cctp_pair, usdc_arbitrum, Decimal(500), ts)

    # 2. Advance trade to broadcasted, then mark it cctp_in_transit
    state.start_execution(ts, trade)
    trade.mark_broadcasted(ts)

    # Add fake blockchain transactions for mark_bridge_in_transit
    approve_tx = BlockchainTransaction()
    approve_tx.tx_hash = "0xapprove"
    burn_tx = BlockchainTransaction()
    burn_tx.tx_hash = "0xburn123"
    trade.blockchain_transactions = [approve_tx, burn_tx]

    state.mark_bridge_in_transit(ts, trade)
    assert trade.get_status() == TradeStatus.cctp_in_transit

    # 3. Verify get_in_transit_value() returns the planned_reserve amount
    in_transit = state.portfolio.get_in_transit_value()
    assert in_transit == pytest.approx(500.0)

    # 4. Verify calculate_total_equity() includes the in-transit value
    # Reserves started at 10,000 but 500 was allocated to the trade via
    # start_execution, so cash is 9,500. Position equity for the bridge
    # position is 0 (no successful trades yet). In-transit = 500.
    total_equity = state.portfolio.calculate_total_equity()
    assert total_equity == pytest.approx(10_000.0)

    # 5. Verify calculate_total_equity_chain() attributes in-transit value
    #    to the destination chain
    chain_equity = state.portfolio.calculate_total_equity_chain()
    from tradingstrategy.chain import ChainId
    # In-transit value should appear under destination chain (Base = 8453)
    assert chain_equity.get(ChainId.base, 0) == pytest.approx(500.0)
    # Reserves remain on Arbitrum
    assert chain_equity.get(ChainId.arbitrum, 0) == pytest.approx(9_500.0)


def test_executor_halt_expires_remaining_trades(
    cctp_pair: TradingPairIdentifier,
    usdc_arbitrum: AssetIdentifier,
):
    """Verify that a cctp_in_transit trade halts the batch and expires remaining trades.

    We mock _execute_trade_batch to simulate the first trade entering
    cctp_in_transit status, then verify the sequential executor:

    1. Create state with reserves and three bridge trades
    2. Mock _execute_trade_batch so the first trade enters cctp_in_transit
    3. Create a second (planned) trade that should be expired
    4. Create a third (planned) trade on a fresh position that should
       be expired and the orphan position cleaned up
    5. Run _execute_trades_sequentially and catch ExecutionHaltableIssue
    6. Verify the first trade is still in cctp_in_transit
    7. Verify remaining planned trades are expired
    8. Verify orphan positions with only expired trades are removed
    """
    from tradeexecutor.ethereum.execution import EthereumExecution

    state = State()
    ts = datetime.datetime(2025, 1, 1)

    # 1. Create state with reserves
    state.portfolio.initialise_reserves(usdc_arbitrum)
    reserve = state.portfolio.get_default_reserve_position()
    reserve.quantity = Decimal(10_000)
    reserve.reserve_token_price = 1.0

    # Create the bridge trade (first in sequence)
    bridge_trade = _create_bridge_trade(state, cctp_pair, usdc_arbitrum, Decimal(500), ts)

    # Create a second planned trade (to be expired)
    second_trade = _create_bridge_trade(state, cctp_pair, usdc_arbitrum, Decimal(200), ts)

    # Create a third planned trade on a fresh pair (orphan position candidate)
    # Use the same pair for simplicity — it ends up on the same position
    # so create a distinct pair to get a distinct position
    other_pair = TradingPairIdentifier(
        base=AssetIdentifier(
            chain_id=8453,
            address="0x0000000000000000000000000000000000000099",
            token_symbol="USDC2",
            decimals=6,
        ),
        quote=usdc_arbitrum,
        pool_address="0x0000000000000000000000000000000000000088",
        exchange_address="0x0000000000000000000000000000000000000088",
        fee=0,
        kind=TradingPairKind.cctp_bridge,
        other_data={"bridge_protocol": "cctp"},
    )
    _, third_trade, _ = state.create_trade(
        strategy_cycle_at=ts,
        pair=other_pair,
        quantity=None,
        reserve=Decimal(100),
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc_arbitrum,
        reserve_currency_price=1.0,
    )
    third_position_id = third_trade.position_id

    # Record the position ID of the third trade before execution
    assert third_position_id in state.portfolio.open_positions

    trades = [bridge_trade, second_trade, third_trade]

    # 2. Mock _execute_trade_batch so the first trade enters cctp_in_transit
    def mock_execute_batch(routing_model, batch_state, batch_trades, rebroadcast):
        for t in batch_trades:
            if t.trade_id == bridge_trade.trade_id:
                # Simulate burn confirmed but attestation timed out
                t.mark_broadcasted(ts)
                approve_tx = BlockchainTransaction()
                approve_tx.tx_hash = "0xapprove"
                burn_tx = BlockchainTransaction()
                burn_tx.tx_hash = "0xburn"
                t.blockchain_transactions = [approve_tx, burn_tx]
                batch_state.mark_bridge_in_transit(ts, t)

    # Build a mock EthereumExecution using MagicMock with spec
    # because web3 is a read-only property on EthereumExecution
    executor = MagicMock(spec=EthereumExecution)
    executor.max_slippage = None
    executor._execute_trade_batch = mock_execute_batch
    # Bind the real method to our mock so logic executes
    executor._execute_trades_sequentially = EthereumExecution._execute_trades_sequentially.__get__(executor)
    executor._log_trade_outcome = EthereumExecution._log_trade_outcome

    mock_routing_model = MagicMock()
    mock_routing_state = MagicMock()

    # 5. Run _execute_trades_sequentially and catch ExecutionHaltableIssue
    with pytest.raises(ExecutionHaltableIssue, match="CCTP bridge in transit"):
        executor._execute_trades_sequentially(
            ts=ts,
            state=state,
            trades=trades,
            routing_model=mock_routing_model,
            routing_state=mock_routing_state,
            check_balances=False,
            rebroadcast=False,
            triggered=False,
        )

    # 6. Verify the first trade is still in cctp_in_transit
    assert bridge_trade.get_status() == TradeStatus.cctp_in_transit

    # 7. Verify remaining planned trades are expired
    assert second_trade.get_status() == TradeStatus.expired
    assert third_trade.get_status() == TradeStatus.expired

    # 8. Verify orphan position (third_trade's position) is removed
    #    since it had zero quantity and all trades expired
    assert third_position_id not in state.portfolio.open_positions


def test_startup_retry_resolves_in_transit(
    cctp_pair: TradingPairIdentifier,
    usdc_arbitrum: AssetIdentifier,
):
    """Verify that check_and_retry_cctp_in_transit resolves an in-transit trade.

    1. Create state with a bridge-out trade in cctp_in_transit status
    2. Mock attestation to return successfully
    3. Mock receiveMessage broadcast and receipt to succeed
    4. Call check_and_retry_cctp_in_transit
    5. Verify the trade transitions to success
    6. Verify cctp_in_transit_at is cleared
    7. Verify the trade is in the returned resolved list
    """
    from tradeexecutor.ethereum.cctp.retry import check_and_retry_cctp_in_transit

    state = State()
    ts = datetime.datetime(2025, 1, 1)

    # 1. Create state with a bridge-out trade in cctp_in_transit status
    state.portfolio.initialise_reserves(usdc_arbitrum)
    reserve = state.portfolio.get_default_reserve_position()
    reserve.quantity = Decimal(10_000)
    reserve.reserve_token_price = 1.0

    trade = _create_bridge_trade(state, cctp_pair, usdc_arbitrum, Decimal(500), ts)
    state.start_execution(ts, trade)
    trade.mark_broadcasted(ts)

    # Add fake blockchain transactions
    approve_tx = BlockchainTransaction()
    approve_tx.tx_hash = "0xapprove"
    burn_tx = BlockchainTransaction()
    burn_tx.tx_hash = "0xburn_abc"
    trade.blockchain_transactions = [approve_tx, burn_tx]

    state.mark_bridge_in_transit(ts, trade)
    assert trade.get_status() == TradeStatus.cctp_in_transit

    # Build mock dependencies
    mock_execution_model = MagicMock()
    mock_execution_model.tx_builder.hot_wallet.account = MagicMock()

    mock_web3config = MagicMock()
    mock_dest_web3 = MagicMock()
    mock_web3config.get_connection.return_value = mock_dest_web3

    # 2. Mock attestation to return successfully
    mock_attestation = MagicMock()
    mock_attestation.message = b"\x01" * 32
    mock_attestation.attestation = b"\x02" * 32

    # 3. Mock receiveMessage broadcast and receipt to succeed
    mock_receive_tx = BlockchainTransaction()
    mock_receive_tx.tx_hash = "0xcccc000000000000000000000000000000000000000000000000000000000099"
    mock_receive_tx.signed_bytes = "0xcafe"

    mock_dest_web3.eth.wait_for_transaction_receipt.return_value = {"status": 1}

    # 4. Call check_and_retry_cctp_in_transit with all mocks
    #    The retry module imports these inside the function body, so
    #    we patch them at their origin packages.
    with (
        patch("eth_defi.cctp.attestation.fetch_attestation", return_value=mock_attestation),
        patch("eth_defi.cctp.transfer._resolve_cctp_domain", return_value=3),
        patch("eth_defi.cctp.transfer.get_message_transmitter_v2", return_value=MagicMock()),
        patch("eth_defi.cctp.receive.prepare_receive_message", return_value=MagicMock()),
        patch("eth_defi.hotwallet.HotWallet") as MockHW,
        patch("tradeexecutor.ethereum.tx.HotWalletTransactionBuilder") as MockTxBuilder,
    ):
        mock_hw_instance = MagicMock()
        MockHW.return_value = mock_hw_instance

        mock_builder_instance = MagicMock()
        mock_builder_instance.sign_transaction.return_value = mock_receive_tx
        MockTxBuilder.return_value = mock_builder_instance

        resolved = check_and_retry_cctp_in_transit(
            state=state,
            execution_model=mock_execution_model,
            web3config=mock_web3config,
            attestation_timeout=5.0,
        )

    # 5. Verify the trade transitions to success
    assert trade.get_status() == TradeStatus.success

    # 6. Verify cctp_in_transit_at is cleared
    assert trade.cctp_in_transit_at is None

    # 7. Verify the trade is in the returned resolved list
    assert len(resolved) == 1
    assert resolved[0].trade_id == trade.trade_id


@pytest.mark.timeout(300)
def test_idle_sweep_skips_in_transit_bridge_back(
    cctp_pair: TradingPairIdentifier,
    usdc_arbitrum: AssetIdentifier,
):
    """An in-transit sweep bridge-back is neither double-swept nor NAV-distorting.

    The idle-capital sweep (issue #1562) bridges settled satellite capital back to
    the hub. In live trading that bridge-back sits ``cctp_in_transit`` for a cycle
    or more, so the next planner cycle must not see the same capital as still-idle
    and sweep it again. It must not, because ``mark_bridge_in_transit`` locks the
    burned amount via ``bridge_capital_allocated``, driving available bridge
    capital to zero; and total NAV must be conserved across the burn window via
    the in-transit value term.

    1. Establish a settled satellite bridge position holding 10_000 idle capital.
    2. Emit a closing bridge-back for the full balance and mark it in transit.
    3. Assert available bridge capital is now zero and total equity is unchanged
       (the in-transit value replaces the position equity).
    4. Run the planner again on a quiet cycle with the sweep enabled and assert it
       injects no further bridge trade.
    """
    state = State()
    ts = datetime.datetime(2025, 1, 1)

    # 1. Establish a settled satellite bridge position holding 10_000 idle capital.
    state.portfolio.initialise_reserves(usdc_arbitrum)
    reserve = state.portfolio.get_default_reserve_position()
    reserve.quantity = Decimal(10_000)
    reserve.reserve_token_price = 1.0
    bridge_out = _create_bridge_trade(state, cctp_pair, usdc_arbitrum, Decimal(10_000), ts)
    state.start_execution(ts, bridge_out)
    bridge_out.mark_broadcasted(ts)
    state.mark_trade_success(
        ts,
        bridge_out,
        executed_price=1.0,
        executed_amount=bridge_out.planned_quantity,
        executed_reserve=bridge_out.planned_reserve,
        lp_fees=0,
        native_token_price=0,
    )
    bridge_pos = state.portfolio.get_bridge_position_for_chain(8453)
    assert bridge_pos.get_available_bridge_capital() == Decimal(10_000)
    equity_before = state.portfolio.calculate_total_equity()

    # 2. Emit a closing bridge-back for the full balance and mark it in transit.
    _, bridge_back, _ = state.create_trade(
        strategy_cycle_at=ts,
        pair=cctp_pair,
        quantity=Decimal(-10_000),
        reserve=None,
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc_arbitrum,
        reserve_currency_price=1.0,
        position=bridge_pos,
        closing=True,
    )
    state.start_execution(ts, bridge_back)
    bridge_back.mark_broadcasted(ts)
    approve_tx = BlockchainTransaction()
    approve_tx.tx_hash = "0xapprove"
    burn_tx = BlockchainTransaction()
    burn_tx.tx_hash = "0xburn"
    bridge_back.blockchain_transactions = [approve_tx, burn_tx]
    state.mark_bridge_in_transit(ts, bridge_back)
    assert bridge_back.get_status() == TradeStatus.cctp_in_transit

    # 3. Available capital is now locked to zero and NAV is conserved.
    assert bridge_pos.get_available_bridge_capital() == Decimal(0)
    assert state.portfolio.calculate_total_equity() == pytest.approx(equity_before)

    # 4. Quiet cycle: the sweep sees no available capital and injects nothing.
    universe = MagicMock()
    universe.iterate_pairs.return_value = [cctp_pair]
    result = inject_cctp_bridge_trades(
        state=state,
        trades=[],
        strategy_universe=universe,
        primary_chain_id=42161,
        ts=ts,
        reserve_asset=usdc_arbitrum,
    )
    assert [t for t in result if t.pair.is_cctp_bridge()] == []
