"""Tests for HyperCore rebalance cost reconstruction."""

import datetime
from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace

import pytest

from tradeexecutor.analysis.hypercore_rebalance_cost import (
    build_hypercore_rebalance_cost_report,
    get_trade_gas_cost_usd,
)
from tradeexecutor.ethereum.vault.hypercore_vault import create_hypercore_vault_pair
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.trade import TradeExecution, TradeType


def test_reconstructs_gas_only_when_every_receipt_and_price_are_available():
    """Historic gas reconstruction uses complete receipt data only.

    1. Create a legacy trade with two complete HyperEVM receipts.
    2. Value both receipts with a supplied HYPE/USD price.
    3. Remove one receipt field and verify that no partial cost is reported.
    """
    # 1. Create a legacy trade with two complete HyperEVM receipts.
    timestamp = datetime.datetime(2026, 7, 1, 12, 0)
    # A minimal receipt object isolates the historic gas arithmetic from
    # unrelated BlockchainTransaction serialisation fields.
    complete_transaction = SimpleNamespace(
        realised_gas_units_consumed=100_000,
        realised_gas_price=2_000_000_000,
        included_at=timestamp,
    )
    trade = SimpleNamespace(
        blockchain_transactions=[complete_transaction, complete_transaction],
        get_cost_of_gas_usd=lambda: None,
    )

    # 2. Value both receipts with a supplied HYPE/USD price.
    assert get_trade_gas_cost_usd(trade, lambda _: 20.0) == pytest.approx(0.008)

    # 3. Remove one receipt field and verify that no partial cost is reported.
    incomplete_transaction = SimpleNamespace(
        realised_gas_units_consumed=None,
        realised_gas_price=2_000_000_000,
        included_at=timestamp,
    )
    trade.blockchain_transactions = [complete_transaction, incomplete_transaction]
    assert get_trade_gas_cost_usd(trade, lambda _: 20.0) is None


def test_rebalance_cost_report_and_state_fields():
    """Cost fields round-trip and reports exclude unusable HyperCore history.

    1. Round-trip the new cost fields through TradeExecution JSON.
    2. Build complete, incomplete, and zero-turnover rebalances.
    3. Add repaired and transactionless history that must be ignored.
    4. Verify exact totals, lower-bound gas metrics, and ignored counts.
    """
    timestamp = datetime.datetime(2026, 7, 1, 12, 0)
    quote = AssetIdentifier(
        chain_id=999,
        address="0x0000000000000000000000000000000000000001",
        token_symbol="USDC",
        decimals=6,
    )
    pair = create_hypercore_vault_pair(
        quote=quote,
        vault_address="0x0000000000000000000000000000000000000002",
    )
    confirmed_transaction = SimpleNamespace(
        realised_gas_units_consumed=100_000,
        realised_gas_price=2_000_000_000,
        included_at=timestamp,
    )
    base_trade = TradeExecution(
        trade_id=1,
        position_id=1,
        trade_type=TradeType.rebalance,
        pair=pair,
        opened_at=timestamp,
        planned_quantity=Decimal(1),
        planned_reserve=Decimal(100),
        planned_price=100.0,
        reserve_currency=pair.quote,
        flags=set(),
        executed_at=timestamp,
        executed_quantity=Decimal(1),
        executed_reserve=Decimal(100),
        executed_price=100.0,
        native_token_price=20.0,
        cost_of_gas=Decimal("0.05"),
        bridge_input_amount=Decimal(100),
        bridge_output_amount=Decimal("99.5"),
        bridge_fee_amount=Decimal("0.5"),
        bridge_fee_asset="USDC",
        bridge_fee_usd=0.5,
        account_activation_fee_usd=0.2,
        hypercore_cost_data_complete=True,
        blockchain_transactions=[],
    )

    # 1. Round-trip the new cost fields through TradeExecution JSON.
    restored_trade = TradeExecution.from_json(base_trade.to_json())
    assert restored_trade.bridge_input_amount == Decimal(100)
    assert restored_trade.bridge_output_amount == Decimal("99.5")
    assert restored_trade.bridge_fee_amount == Decimal("0.5")
    assert restored_trade.bridge_fee_asset == "USDC"
    assert restored_trade.hypercore_cost_data_complete is True

    # 2. Build complete, incomplete, and zero-turnover rebalances.
    base_trade.blockchain_transactions = [confirmed_transaction]
    close_trade = replace(
        base_trade,
        trade_id=2,
        position_id=2,
        planned_quantity=Decimal(-1),
        account_activation_fee_usd=None,
        bridge_fee_usd=0.3,
        cost_of_gas=Decimal("0.1"),
        hypercore_close_other_loss_usd=0.4,
    )
    incomplete_trade = replace(
        base_trade,
        trade_id=3,
        position_id=3,
        opened_at=timestamp + datetime.timedelta(days=1),
        account_activation_fee_usd=None,
        bridge_fee_usd=None,
        cost_of_gas=Decimal("0.15"),
        hypercore_cost_data_complete=False,
    )
    zero_turnover_trade = replace(
        base_trade,
        trade_id=7,
        position_id=7,
        opened_at=timestamp + datetime.timedelta(days=2),
        executed_reserve=Decimal(0),
        account_activation_fee_usd=None,
        bridge_fee_usd=0.0,
    )

    # 3. Add repaired and transactionless history that must be ignored.
    repaired_trade = replace(
        base_trade,
        trade_id=4,
        position_id=4,
        repaired_at=timestamp,
    )
    repair_trade = replace(
        base_trade,
        trade_id=5,
        position_id=5,
        trade_type=TradeType.repair,
    )
    transactionless_trade = replace(
        base_trade,
        trade_id=6,
        position_id=6,
        blockchain_transactions=[],
    )
    # The report only needs Portfolio.get_all_trades(); this narrow mock keeps
    # the test focused on grouping and exclusion semantics.
    state = SimpleNamespace(
        portfolio=SimpleNamespace(
            get_all_trades=lambda: [
                base_trade,
                close_trade,
                incomplete_trade,
                zero_turnover_trade,
                repaired_trade,
                repair_trade,
                transactionless_trade,
            ],
        ),
    )

    # 4. Verify exact totals, lower-bound gas metrics, and ignored counts.
    report = build_hypercore_rebalance_cost_report(state)
    first_rebalance = report.rebalances.iloc[0]
    summary = report.summary.iloc[0]
    ignored = dict(zip(report.ignored["reason"], report.ignored["trades"]))
    assert len(report.trades) == 4
    assert len(report.rebalances) == 3
    assert first_rebalance["total_cost_usd"] == pytest.approx(4.4)
    assert first_rebalance["cost_bps"] == pytest.approx(220)
    assert summary["complete_rebalances"] == 2
    assert summary["incomplete_rebalances"] == 1
    assert summary["zero_turnover_rebalances"] == 1
    assert summary["total_cost_usd"] == pytest.approx(5.4)
    assert summary["weighted_cost_bps"] == pytest.approx(270)
    assert summary["average_cost_bps_per_rebalance"] == pytest.approx(220)
    assert summary["known_gas_lower_bound_usd"] == pytest.approx(7)
    assert summary["observed_gas_bps"] == pytest.approx(233.333333)
    assert ignored == {
        "repaired original": 1,
        "successful trade without transactions": 1,
        "trade type repair": 1,
    }
