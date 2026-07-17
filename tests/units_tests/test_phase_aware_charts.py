"""Unit tests for the phase-aware charts and diagnostics (cleanup CU-4).

Covers the three additions that surface the queue-venue mechanism to an operator:

- ``pending_trigger_queue`` (``chart/standard/vault.py``): the waiting-deposit buffer parked in the
  queue venue, reconstructed from the durable park/promote/close event log in ``state.other_data``.
- ``format_signals`` (``alpha_model.py``): Parked / Waiting deposit / Waiting redemption USD columns.
- ``is_queue_vault_position`` (``phase_aware.py``): venue identity from state alone (used by
  ``chart/standard/weight.py`` to split the queue venue into its own band; exercised against a real
  run in ``tests/backtest/test_phase_aware_backtest.py``).

All charts are built from a hand-made ``State`` + ``ChartInput`` (no backtest needed).
"""
import datetime
from decimal import Decimal

import pandas as pd
import pytest

from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier, TradingPairKind
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.reserve import ReservePosition
from tradeexecutor.state.state import State
from tradeexecutor.state.statistics import PortfolioStatistics, PositionStatistics
from tradeexecutor.state.trade import TradeExecution, TradeType
from tradeexecutor.strategy.alpha_model import AlphaModel, TradingPairSignal, format_signals
from tradeexecutor.strategy.chart.definition import ChartInput
from tradeexecutor.strategy.chart.standard.weight import equity_curve_by_asset
from tradeexecutor.strategy.chart.standard.vault import (
    pending_trigger_queue,
    phase_aware_queue_duration,
    phase_aware_queue_duration_summary,
)
from tradeexecutor.strategy.execution_context import unit_test_execution_context
from tradeexecutor.strategy.phase_aware import (
    EVENT_PARK,
    EVENT_PROMOTE,
    EVENT_CLOSE,
    EVENT_REDEEM_BLOCK,
    EVENT_REDEEM_CLEAR,
    QueueVaultEvent,
    append_queue_event,
)
from tradeexecutor.testing.synthetic_ethereum_data import generate_random_ethereum_address
from tradingstrategy.chain import ChainId


def _make_pair(
    internal_id: int,
    *,
    kind: TradingPairKind = TradingPairKind.spot_market_hold,
    vault_name: str | None = None,
) -> TradingPairIdentifier:
    base = AssetIdentifier(ChainId.ethereum.value, generate_random_ethereum_address(), "VLT", 18, internal_id)
    quote = AssetIdentifier(ChainId.ethereum.value, generate_random_ethereum_address(), "USDC", 6, 999999)
    pair = TradingPairIdentifier(
        base,
        quote,
        generate_random_ethereum_address(),
        generate_random_ethereum_address(),
        internal_id=internal_id,
        kind=kind,
    )
    if vault_name is not None:
        pair.other_data["vault_name"] = vault_name
    return pair


def _make_position(
    position_id: int,
    pair: TradingPairIdentifier,
    reserve: AssetIdentifier,
    timestamp: datetime.datetime,
    queue_venue: bool = False,
) -> TradingPosition:
    position = TradingPosition(
        position_id=position_id,
        pair=pair,
        opened_at=timestamp,
        last_pricing_at=timestamp,
        last_token_price=1.0,
        last_reserve_price=1.0,
        reserve_currency=reserve,
    )
    if queue_venue:
        trade = TradeExecution(
            trade_id=position_id,
            position_id=position_id,
            trade_type=TradeType.rebalance,
            pair=pair,
            opened_at=timestamp,
            planned_quantity=Decimal(1),
            planned_reserve=Decimal(1),
            planned_price=1.0,
            reserve_currency=reserve,
        )
        trade.other_data["yield_decision"] = {"weight": 1.0}
        position.trades[trade.trade_id] = trade
    return position


def test_pending_trigger_queue_reconstructs_waiting_deposits():
    """pending_trigger_queue reconstructs the open-park (waiting deposit) buffer from the event log.

    1. Seed the durable event log: a $5,000 park on cycle 1, promoted on cycle 4, both timestamped.
    2. Seed six daily portfolio-statistics timestamps for the x-axis.
    3. The chart shows $5,000 waiting while the park is open (days 1-3) and drops it once promoted
       (day 4 onward has no waiting-deposit row).
    """
    start = datetime.datetime(2024, 1, 1)
    one_day = datetime.timedelta(days=1)
    state = State()

    # 1. Event log: park on cycle 1, promote on cycle 4.
    append_queue_event(state.other_data, QueueVaultEvent(EVENT_PARK, 601, 5000.0, 1, timestamp=(start + one_day).isoformat()))
    append_queue_event(state.other_data, QueueVaultEvent(EVENT_PROMOTE, 601, 5000.0, 4, timestamp=(start + 4 * one_day).isoformat()))

    # 2. Six daily statistics timestamps.
    for n in range(6):
        state.stats.portfolio.append(PortfolioStatistics(calculated_at=start + n * one_day, total_equity=10_000.0))

    # 3. Waiting deposits present while parked, gone after promote.
    _fig, df = pending_trigger_queue(ChartInput(execution_context=unit_test_execution_context, state=state))
    assert df.loc[pd.Timestamp(start + one_day)]["Waiting deposits"] == pytest.approx(5000.0)
    assert df.loc[pd.Timestamp(start + 3 * one_day)]["Waiting deposits"] == pytest.approx(5000.0)
    assert pd.Timestamp(start + 4 * one_day) not in df.index  # promoted -> no longer waiting


def test_pending_trigger_queue_shows_redemption_locked():
    """pending_trigger_queue renders redemption-locked value as a negative band until cleared (CU-7).

    1. Seed the durable event log: a $3,000 redeem-block on cycle 2, cleared on cycle 4, both
       timestamped.
    2. Seed six daily portfolio-statistics timestamps for the x-axis.
    3. The chart shows -$3,000 locked while the redemption is blocked (days 2-3) and drops it once
       cleared (day 4 onward has no redemption-locked row).
    """
    start = datetime.datetime(2024, 1, 1)
    one_day = datetime.timedelta(days=1)
    state = State()

    # 1. Event log: blocked on cycle 2, cleared on cycle 4.
    append_queue_event(state.other_data, QueueVaultEvent(EVENT_REDEEM_BLOCK, 700, 3000.0, 2, timestamp=(start + 2 * one_day).isoformat()))
    append_queue_event(state.other_data, QueueVaultEvent(EVENT_REDEEM_CLEAR, 700, 0.0, 4, timestamp=(start + 4 * one_day).isoformat()))

    # 2. Six daily statistics timestamps.
    for n in range(6):
        state.stats.portfolio.append(PortfolioStatistics(calculated_at=start + n * one_day, total_equity=10_000.0))

    # 3. Negative redemption-locked band while blocked, gone after the clear.
    _fig, df = pending_trigger_queue(ChartInput(execution_context=unit_test_execution_context, state=state))
    assert df.loc[pd.Timestamp(start + 2 * one_day)]["Redemption locked"] == pytest.approx(-3000.0)
    assert df.loc[pd.Timestamp(start + 3 * one_day)]["Redemption locked"] == pytest.approx(-3000.0)
    assert pd.Timestamp(start + 4 * one_day) not in df.index  # cleared -> no longer locked


def test_pending_trigger_queue_empty_without_events():
    """With no queue events the chart returns an empty frame rather than crashing.

    1. A state with statistics timestamps but no park/promote events.
    2. pending_trigger_queue returns an empty DataFrame.
    """
    state = State()
    state.stats.portfolio.append(PortfolioStatistics(calculated_at=datetime.datetime(2024, 1, 1), total_equity=10_000.0))

    # 1-2. No events -> empty frame.
    _fig, df = pending_trigger_queue(ChartInput(execution_context=unit_test_execution_context, state=state))
    assert df.empty


def test_phase_aware_queue_duration_summary():
    """phase_aware_queue_duration_summary reports completed and open wait intervals.

    1. Vault 601 parks for three days and promotes.
    2. Vault 602 parks, updates the parked amount, and is still open at the last stats timestamp.
    3. The summary reports duration, episode counts, max queued USD, and USD-days.
    """
    start = datetime.datetime(2024, 1, 1)
    one_day = datetime.timedelta(days=1)
    state = State()

    append_queue_event(state.other_data, QueueVaultEvent(EVENT_PARK, 601, 5000.0, 1, timestamp=(start + one_day).isoformat()))
    append_queue_event(state.other_data, QueueVaultEvent(EVENT_PROMOTE, 601, 5000.0, 4, timestamp=(start + 4 * one_day).isoformat()))
    append_queue_event(state.other_data, QueueVaultEvent(EVENT_PARK, 602, 2000.0, 2, timestamp=(start + 2 * one_day).isoformat()))
    append_queue_event(state.other_data, QueueVaultEvent(EVENT_PARK, 602, 3000.0, 3, timestamp=(start + 3 * one_day).isoformat()))

    for n in range(6):
        state.stats.portfolio.append(PortfolioStatistics(calculated_at=start + n * one_day, total_equity=10_000.0))

    df = phase_aware_queue_duration_summary(ChartInput(execution_context=unit_test_execution_context, state=state))

    row_601 = df.loc[df["Vault id"] == 601].iloc[0]
    assert row_601["Episodes"] == 1
    assert row_601["Promoted"] == 1
    assert row_601["Open"] == 0
    assert row_601["Average wait days"] == pytest.approx(3.0)
    assert row_601["USD days"] == pytest.approx(15_000.0)

    row_602 = df.loc[df["Vault id"] == 602].iloc[0]
    assert row_602["Episodes"] == 1
    assert row_602["Promoted"] == 0
    assert row_602["Open"] == 1
    assert row_602["Average wait days"] == pytest.approx(3.0)
    assert row_602["Max queued USD"] == pytest.approx(3000.0)
    assert row_602["USD days"] == pytest.approx(8_000.0)


def test_phase_aware_queue_duration_chart_and_closed_wait():
    """phase_aware_queue_duration returns a chart and counts abandoned waits as closed."""
    start = datetime.datetime(2024, 1, 1)
    one_day = datetime.timedelta(days=1)
    state = State()

    append_queue_event(state.other_data, QueueVaultEvent(EVENT_PARK, 701, 1000.0, 1, timestamp=(start + one_day).isoformat()))
    append_queue_event(state.other_data, QueueVaultEvent(EVENT_CLOSE, 701, 0.0, 3, timestamp=(start + 3 * one_day).isoformat()))
    for n in range(4):
        state.stats.portfolio.append(PortfolioStatistics(calculated_at=start + n * one_day, total_equity=10_000.0))

    fig, df = phase_aware_queue_duration(ChartInput(execution_context=unit_test_execution_context, state=state))

    assert fig is not None
    row = df.loc[df["Vault id"] == 701].iloc[0]
    assert row["Closed"] == 1
    assert row["Average wait days"] == pytest.approx(2.0)


def test_format_signals_phase_aware_columns():
    """format_signals surfaces the phase-aware queue-venue diagnostics as columns.

    1. A signal carrying parked_usd / missed_deposit_usd / missed_redemption_usd in other_data.
    2. format_signals emits Parked USD / Waiting deposit USD / Waiting redemption USD with those values.
    """
    # 1. A chosen signal with the phase-aware diagnostics attached.
    alpha = AlphaModel(datetime.datetime(2024, 1, 1))
    pair = _make_pair(101)
    signal = TradingPairSignal(pair=pair, signal=0.5)
    signal.other_data["parked_usd"] = 1000.0
    signal.other_data["missed_deposit_usd"] = 500.0
    signal.other_data["missed_redemption_usd"] = 250.0
    alpha.signals = {101: signal}

    # 2. The new columns are present with the recorded values.
    df = format_signals(alpha, signal_type="chosen")
    row = df.loc[pair.get_ticker()]
    assert row["Parked USD"] == pytest.approx(1000.0)
    assert row["Waiting deposit USD"] == pytest.approx(500.0)
    assert row["Waiting redemption USD"] == pytest.approx(250.0)


def test_equity_curve_by_asset_relabels_queue_venues_by_position_id():
    """equity_curve_by_asset separates queue venues before same-label positions are grouped.

    1. Build one non-queue vault with the same chart label as a queue venue.
    2. Build another queue venue whose vault has no ``vault_name`` metadata.
    3. Render equity_curve_by_asset and assert the same-label directional value is not repainted
       as queue, while the missing-label venue falls back to ticker + ``" [queue]"`` instead of
       disappearing from the grouped series.
    """
    timestamp = datetime.datetime(2024, 1, 1)
    state = State()
    usdc = AssetIdentifier(ChainId.ethereum.value, generate_random_ethereum_address(), "USDC", 6, 999999)
    state.portfolio.reserves[usdc.get_identifier()] = ReservePosition(
        asset=usdc,
        quantity=Decimal(30),
        last_sync_at=timestamp,
        reserve_token_price=1.0,
        last_pricing_at=timestamp,
    )

    # 1. Non-queue vault shares the same chart label as the queue venue below.
    same_label = "Steakhouse USDC"
    directional_pair = _make_pair(101, kind=TradingPairKind.vault, vault_name=same_label)
    same_label_queue_pair = _make_pair(202, kind=TradingPairKind.vault, vault_name=same_label)
    # 2. This queue venue has no vault_name metadata, so the chart must fall back to ticker.
    missing_label_queue_pair = _make_pair(303, kind=TradingPairKind.vault)
    state.portfolio.open_positions = {
        1: _make_position(1, directional_pair, usdc, timestamp),
        2: _make_position(2, same_label_queue_pair, usdc, timestamp, queue_venue=True),
        3: _make_position(3, missing_label_queue_pair, usdc, timestamp, queue_venue=True),
    }
    state.stats.positions = {
        1: [PositionStatistics(timestamp, timestamp, 0.0, 0.0, 70.0, 70.0)],
        2: [PositionStatistics(timestamp, timestamp, 0.0, 0.0, 120.0, 120.0)],
        3: [PositionStatistics(timestamp, timestamp, 0.0, 0.0, 50.0, 50.0)],
    }
    state.stats.portfolio.append(
        PortfolioStatistics(calculated_at=timestamp, total_equity=270.0, open_position_equity=240.0),
    )

    # 3. Real chart path, not hand-built visualise_weights params.
    fig = equity_curve_by_asset(ChartInput(execution_context=unit_test_execution_context, state=state))
    trace_values = {trace.name: list(trace.y) for trace in fig.data}
    assert trace_values["Steakhouse USDC"] == pytest.approx([70.0])
    assert trace_values["Steakhouse USDC [queue]"] == pytest.approx([120.0])
    assert trace_values["VLT-USDC [queue]"] == pytest.approx([50.0])
    assert trace_values["USDC"] == pytest.approx([30.0])
