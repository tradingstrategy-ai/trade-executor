"""Unit-style tests for Hypercore replay market data."""

from __future__ import annotations

import datetime
from decimal import Decimal
from tradeexecutor.state.identifier import TradingPairIdentifier

import pytest
import pandas as pd

from tradeexecutor.ethereum.vault.hypercore_valuation import HypercoreVaultPricing
from tradeexecutor.testing.hypercore_replay import HypercoreDailyMetricsReplay


def test_hypercore_replay_snapshot_asof(
    hypercore_replay_source: HypercoreDailyMetricsReplay,
    hypercore_vault_pair: TradingPairIdentifier,
) -> None:
    """Verify the replay snapshot lookup rules.

    1. Ask the replay source for a snapshot after the middle data point.
    2. Confirm the replay picks the latest available historical row on or before that date.
    3. Confirm the hardcoded v1 defaults still fill the missing Hypercore fields.
    """
    # 1. Ask the replay source for a snapshot after the middle data point.
    # We use the replay source because Hypercore does not expose historical simulation for this test.
    snapshot = hypercore_replay_source.get_snapshot(
        datetime.datetime(2026, 1, 30),
        hypercore_vault_pair,
        net_deposited_usdc=Decimal("100"),
    )

    # 2. Confirm the replay picks the latest available historical row on or before that date.
    assert snapshot.data_date == datetime.date(2026, 1, 21)
    assert snapshot.tvl_usd == Decimal("269004391.733933")
    assert snapshot.account_pnl_usd == Decimal("119191482.033933")

    # 3. Confirm the hardcoded v1 defaults still fill the missing Hypercore fields.
    # These fields are mocked because historical daily metrics only give us TVL and account PnL.
    assert snapshot.leader_fraction == Decimal("0.10")
    assert snapshot.allow_deposits is True
    assert snapshot.is_closed is False
    assert snapshot.relationship_type == "normal"


def test_hypercore_replay_share_price_calculation(
    hypercore_replay_source: HypercoreDailyMetricsReplay,
    hypercore_vault_pair: TradingPairIdentifier,
) -> None:
    """Verify the mock share price reconstruction.

    1. Load one early replay snapshot and one later replay snapshot for the same net deposit.
    2. Check that the reconstructed share price grows across the replay window.
    3. Check that replay equity is derived from ``net_deposited_usdc * share_price``.
    """
    # 1. Load one early replay snapshot and one later replay snapshot for the same net deposit.
    # We use replay snapshots because share price is reconstructed by the mock from historical inputs.
    first = hypercore_replay_source.get_snapshot(
        datetime.datetime(2026, 1, 21),
        hypercore_vault_pair,
        net_deposited_usdc=Decimal("125"),
    )
    later = hypercore_replay_source.get_snapshot(
        datetime.datetime(2026, 2, 3),
        hypercore_vault_pair,
        net_deposited_usdc=Decimal("125"),
    )

    # 2. Check that the reconstructed share price grows across the replay window.
    assert first.share_price > Decimal("1")
    assert later.share_price > first.share_price

    # 3. Check that replay equity is derived from ``net_deposited_usdc * share_price``.
    assert first.equity_usd == Decimal("125") * first.share_price
    assert later.equity_usd == Decimal("125") * later.share_price


def test_hypercore_pricing_uses_replay_tvl_and_gating(
    hypercore_replay_source: HypercoreDailyMetricsReplay,
    hypercore_daily_metrics_frame: pd.DataFrame,
    hypercore_vault_pair: TradingPairIdentifier,
) -> None:
    """Verify replay-backed pricing and redemption gating.

    1. Build a Hypercore pricing model that reads TVL and gating data from the replay source.
    2. Check that TVL, deposit permission and redeem permission come from the replay snapshot.
    3. Rebuild the source with a future lock-up expiry and confirm redemption is blocked.
    """
    # 1. Build a Hypercore pricing model that reads TVL and gating data from the replay source.
    # We inject the replay source because the live Hypercore API cannot simulate this history for tests.
    # The other callables are minimal mocks because this test only cares about replay TVL and gating behaviour.
    pricing = HypercoreVaultPricing(
        value_func=lambda pair: Decimal("150"),
        safe_address_resolver=lambda pair: "0x0000000000000000000000000000000000000001",
        session_factory=lambda pair: None,
        simulate=False,
        market_data_source=hypercore_replay_source,
    )

    timestamp = datetime.datetime(2026, 2, 3)

    # 2. Check that TVL, deposit permission and redeem permission come from the replay snapshot.
    assert pricing.get_usd_tvl(timestamp, hypercore_vault_pair) == pytest.approx(308585706.340077)
    assert pricing.can_deposit(timestamp, hypercore_vault_pair) is True
    assert pricing.can_redeem(timestamp, hypercore_vault_pair) is True
    assert pricing.get_max_redemption(timestamp, hypercore_vault_pair) == Decimal("150")

    # 3. Rebuild the source with a future lock-up expiry and confirm redemption is blocked.
    # We mock the lock-up boundary here because that historical state is not present in daily metrics.
    locked_source = HypercoreDailyMetricsReplay.from_single_vault_dataframe(
        hypercore_vault_pair.other_data["hypercore_vault_address"],
        hypercore_daily_metrics_frame,
        lockup_expired_after=datetime.datetime(2026, 3, 1),
    )
    # The same minimal mocks are enough here because we only want to flip the replay lock-up rule.
    locked_pricing = HypercoreVaultPricing(
        value_func=lambda pair: Decimal("150"),
        safe_address_resolver=lambda pair: "0x0000000000000000000000000000000000000001",
        session_factory=lambda pair: None,
        simulate=False,
        market_data_source=locked_source,
    )

    assert locked_pricing.get_max_redemption(timestamp, hypercore_vault_pair) == Decimal(0)
    assert locked_pricing.can_redeem(timestamp, hypercore_vault_pair) is False
