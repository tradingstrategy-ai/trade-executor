"""Test CCTP bridge multi-phase settlement logic.

Verifies that ``CctpBridgeRouting.settle_trade()`` correctly handles:

- Attestation timeout (trade marked in-transit)
- ``receiveMessage`` revert (trade marked in-transit)
- Full success flow (burn + attestation + receive all succeed)

All web3 and eth_defi calls are mocked — these tests exercise state
transitions, not actual blockchain interaction.
"""

import datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from tradeexecutor.ethereum.cctp.routing import CctpBridgeRouting
from tradeexecutor.ethereum.web3config import Web3Config
from tradeexecutor.state.blockhain_transaction import (
    BlockchainTransaction,
    BlockchainTransactionType,
)
from tradeexecutor.state.identifier import (
    AssetIdentifier,
    TradingPairIdentifier,
    TradingPairKind,
)
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution, TradeStatus, TradeType


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


def _create_broadcasted_bridge_trade(
    state: State,
    cctp_pair: TradingPairIdentifier,
    reserve_asset: AssetIdentifier,
    reserve_amount: Decimal,
    ts: datetime.datetime,
) -> TradeExecution:
    """Create a bridge-out (buy) trade in broadcasted state with fake burn tx.

    Returns a trade that has:

    - Two ``BlockchainTransaction`` objects (approve + burn)
    - The burn tx has ``function_selector='depositForBurn'`` and ``type=hot_wallet``
    - The trade is in broadcasted status
    """
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

    approve_tx = BlockchainTransaction()
    approve_tx.tx_hash = "0xaaaa000000000000000000000000000000000000000000000000000000000001"
    approve_tx.function_selector = "approve"
    approve_tx.type = BlockchainTransactionType.hot_wallet
    approve_tx.signed_bytes = "0xdead"

    burn_tx = BlockchainTransaction()
    burn_tx.tx_hash = "0xbbbb000000000000000000000000000000000000000000000000000000000002"
    burn_tx.function_selector = "depositForBurn"
    burn_tx.type = BlockchainTransactionType.hot_wallet
    burn_tx.signed_bytes = "0xbeef"

    trade.blockchain_transactions = [approve_tx, burn_tx]

    state.start_execution(ts, trade)
    trade.mark_broadcasted(ts)

    return trade


@pytest.fixture()
def bridge_state_and_trade(
    cctp_pair: TradingPairIdentifier,
    usdc_arbitrum: AssetIdentifier,
) -> tuple[State, TradeExecution]:
    """State with reserves and a broadcasted bridge-out trade."""
    state = State()
    ts = datetime.datetime(2025, 1, 1)

    state.portfolio.initialise_reserves(usdc_arbitrum)
    reserve = state.portfolio.get_default_reserve_position()
    reserve.quantity = Decimal(10_000)
    reserve.reserve_token_price = 1.0

    trade = _create_broadcasted_bridge_trade(
        state, cctp_pair, usdc_arbitrum, Decimal(500), ts,
    )
    return state, trade


def _make_routing(web3config: Web3Config | None = None) -> CctpBridgeRouting:
    """Build a CctpBridgeRouting with a mock web3config and fake hot wallet."""
    if web3config is None:
        web3config = MagicMock(spec=Web3Config)

    routing = CctpBridgeRouting(
        web3config=web3config,
        attestation_timeout=5.0,
    )

    mock_hw = MagicMock()
    mock_hw.account = MagicMock()
    routing._hot_wallet = mock_hw

    return routing


def test_settle_trade_attestation_timeout(
    bridge_state_and_trade: tuple[State, TradeExecution],
):
    """Verify that an attestation timeout marks the trade as cctp_in_transit.

    1. Build a CctpBridgeRouting with mocked dependencies
    2. Mock fetch_attestation to raise TimeoutError
    3. Mock fetch_block_timestamp and _resolve_cctp_domain
    4. Call settle_trade with a successful burn receipt
    5. Assert the trade is marked cctp_in_transit
    6. Assert metadata (chain IDs, burn tx hash) is stored
    """
    state, trade = bridge_state_and_trade
    routing = _make_routing()

    # 1. Receipts dict keyed by HexBytes of burn tx hash
    from hexbytes import HexBytes
    receipts = {
        HexBytes("0xbbbb000000000000000000000000000000000000000000000000000000000002"): {"status": 1, "blockNumber": 100},
    }

    mock_ts = datetime.datetime(2025, 1, 1, 0, 5)

    # 2. Mock fetch_attestation to raise TimeoutError
    # 3. Mock fetch_block_timestamp and _resolve_cctp_domain
    with (
        patch("tradeexecutor.ethereum.cctp.routing.fetch_block_timestamp", return_value=mock_ts),
        patch("eth_defi.cctp.attestation.fetch_attestation", side_effect=TimeoutError("timed out")),
        patch("eth_defi.cctp.transfer._resolve_cctp_domain", return_value=3),
    ):
        mock_web3 = MagicMock()
        # 4. Call settle_trade with a successful burn receipt
        routing.settle_trade(mock_web3, state, trade, receipts)

    # 5. Assert the trade is marked cctp_in_transit
    assert trade.get_status() == TradeStatus.cctp_in_transit
    assert trade.cctp_in_transit_at == mock_ts

    # 6. Assert metadata (chain IDs, burn tx hash) is stored
    assert trade.other_data["cctp_source_chain_id"] == 42161
    assert trade.other_data["cctp_dest_chain_id"] == 8453
    assert trade.other_data["cctp_burn_tx_hash"] == "0xbbbb000000000000000000000000000000000000000000000000000000000002"


def test_settle_trade_receive_revert(
    bridge_state_and_trade: tuple[State, TradeExecution],
):
    """Verify that a receiveMessage revert marks the trade as cctp_in_transit.

    1. Build routing with mocked web3config that returns a destination web3
    2. Mock fetch_attestation to return a successful attestation
    3. Mock the destination chain tx building and broadcast
    4. Set the receive receipt status to 0 (revert)
    5. Call settle_trade
    6. Assert the trade is marked cctp_in_transit (not success, not failed)
    7. Assert a receiveMessage tx was appended to blockchain_transactions
    """
    state, trade = bridge_state_and_trade
    assert len(trade.blockchain_transactions) == 2

    mock_dest_web3 = MagicMock()
    mock_dest_web3.eth.chain_id = 8453

    mock_web3config = MagicMock(spec=Web3Config)
    mock_web3config.get_connection.return_value = mock_dest_web3

    routing = _make_routing(web3config=mock_web3config)

    from hexbytes import HexBytes
    receipts = {
        HexBytes("0xbbbb000000000000000000000000000000000000000000000000000000000002"): {"status": 1, "blockNumber": 100},
    }

    mock_ts = datetime.datetime(2025, 1, 1, 0, 5)

    mock_attestation = MagicMock()
    mock_attestation.message = b"\x01" * 32
    mock_attestation.attestation = b"\x02" * 32

    # 3. Mock the destination chain tx building and broadcast
    mock_receive_tx = BlockchainTransaction()
    mock_receive_tx.tx_hash = "0xcccc000000000000000000000000000000000000000000000000000000000003"
    mock_receive_tx.function_selector = "receiveMessage"
    mock_receive_tx.type = BlockchainTransactionType.hot_wallet
    mock_receive_tx.signed_bytes = "0xcafe"

    # 4. Set the receive receipt status to 0 (revert)
    mock_dest_web3.eth.wait_for_transaction_receipt.return_value = {
        "status": 0, "blockNumber": 200, "blockHash": b"\x00" * 32,
        "gasUsed": 100_000, "effectiveGasPrice": 1_000_000_000,
    }

    with (
        patch("tradeexecutor.ethereum.cctp.routing.fetch_block_timestamp", return_value=mock_ts),
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

        mock_web3 = MagicMock()

        # 5. Call settle_trade
        routing.settle_trade(mock_web3, state, trade, receipts)

    # 6. Assert the trade is marked cctp_in_transit (not success, not failed)
    assert trade.get_status() == TradeStatus.cctp_in_transit
    assert trade.cctp_in_transit_at == mock_ts

    # 7. Assert a receiveMessage tx was appended to blockchain_transactions
    assert len(trade.blockchain_transactions) == 3
    assert trade.blockchain_transactions[2].tx_hash == "0xcccc000000000000000000000000000000000000000000000000000000000003"


def test_settle_trade_success_flow(
    bridge_state_and_trade: tuple[State, TradeExecution],
):
    """Verify the full success flow: burn confirmed, attestation received, receiveMessage succeeds.

    1. Build routing with mocked web3config
    2. Mock fetch_attestation to return a successful attestation
    3. Mock the destination chain tx building and broadcast
    4. Set the receive receipt status to 1 (success)
    5. Call settle_trade
    6. Assert the trade is marked success with 1:1 price
    7. Assert a receiveMessage tx was appended and the trade has 3 txs total
    """
    state, trade = bridge_state_and_trade

    mock_dest_web3 = MagicMock()
    mock_dest_web3.eth.chain_id = 8453

    mock_web3config = MagicMock(spec=Web3Config)
    mock_web3config.get_connection.return_value = mock_dest_web3

    routing = _make_routing(web3config=mock_web3config)

    from hexbytes import HexBytes
    receipts = {
        HexBytes("0xbbbb000000000000000000000000000000000000000000000000000000000002"): {"status": 1, "blockNumber": 100},
    }

    mock_ts = datetime.datetime(2025, 1, 1, 0, 5)

    mock_attestation = MagicMock()
    mock_attestation.message = b"\x01" * 32
    mock_attestation.attestation = b"\x02" * 32

    mock_receive_tx = BlockchainTransaction()
    mock_receive_tx.tx_hash = "0xdddd000000000000000000000000000000000000000000000000000000000004"
    mock_receive_tx.function_selector = "receiveMessage"
    mock_receive_tx.type = BlockchainTransactionType.hot_wallet
    mock_receive_tx.signed_bytes = "0xcafe"

    # 4. Set the receive receipt status to 1 (success)
    mock_dest_web3.eth.wait_for_transaction_receipt.return_value = {
        "status": 1, "blockNumber": 200, "blockHash": b"\x00" * 32,
        "gasUsed": 100_000, "effectiveGasPrice": 1_000_000_000,
    }

    with (
        patch("tradeexecutor.ethereum.cctp.routing.fetch_block_timestamp", return_value=mock_ts),
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

        mock_web3 = MagicMock()

        # 5. Call settle_trade
        routing.settle_trade(mock_web3, state, trade, receipts)

    # 6. Assert the trade is marked success with 1:1 price
    assert trade.get_status() == TradeStatus.success
    assert trade.executed_price == pytest.approx(1.0)
    assert trade.executed_quantity == trade.planned_quantity
    assert trade.executed_reserve == trade.planned_reserve

    # 7. Assert a receiveMessage tx was appended and the trade has 3 txs total
    assert len(trade.blockchain_transactions) == 3
    assert trade.blockchain_transactions[2].tx_hash == "0xdddd000000000000000000000000000000000000000000000000000000000004"
