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

import pandas as pd
import pytest

from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.state.statistics import PortfolioStatistics
from tradeexecutor.strategy.alpha_model import AlphaModel, TradingPairSignal, format_signals
from tradeexecutor.strategy.chart.definition import ChartInput
from tradeexecutor.strategy.chart.standard.vault import pending_trigger_queue
from tradeexecutor.strategy.execution_context import unit_test_execution_context
from tradeexecutor.strategy.phase_aware import (
    EVENT_PARK,
    EVENT_PROMOTE,
    EVENT_REDEEM_BLOCK,
    EVENT_REDEEM_CLEAR,
    QueueVaultEvent,
    append_queue_event,
)
from tradeexecutor.testing.synthetic_ethereum_data import generate_random_ethereum_address
from tradingstrategy.chain import ChainId


def _make_pair(internal_id: int) -> TradingPairIdentifier:
    base = AssetIdentifier(ChainId.ethereum.value, generate_random_ethereum_address(), "VLT", 18, internal_id)
    quote = AssetIdentifier(ChainId.ethereum.value, generate_random_ethereum_address(), "USDC", 6, 999999)
    return TradingPairIdentifier(
        base,
        quote,
        generate_random_ethereum_address(),
        generate_random_ethereum_address(),
        internal_id=internal_id,
    )


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
