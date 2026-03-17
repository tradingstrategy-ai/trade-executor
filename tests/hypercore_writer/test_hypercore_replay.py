"""Unit-style tests for Hypercore replay market data."""

from __future__ import annotations

import datetime
from decimal import Decimal

import pytest

from tradeexecutor.ethereum.vault.hypercore_valuation import HypercoreVaultPricing
from tradeexecutor.testing.hypercore_replay import HypercoreDailyMetricsReplay


def test_hypercore_replay_snapshot_asof(
    hypercore_replay_source: HypercoreDailyMetricsReplay,
    hypercore_vault_pair,
):
    snapshot = hypercore_replay_source.get_snapshot(
        datetime.datetime(2026, 1, 30),
        hypercore_vault_pair,
        net_deposited_usdc=Decimal("100"),
    )

    assert snapshot.data_date == datetime.date(2026, 1, 21)
    assert snapshot.tvl_usd == Decimal("269004391.733933")
    assert snapshot.account_pnl_usd == Decimal("119191482.033933")

    # v1 replay defaults: only TVL and account PnL are historical truth.
    assert snapshot.leader_fraction == Decimal("0.10")
    assert snapshot.allow_deposits is True
    assert snapshot.is_closed is False
    assert snapshot.relationship_type == "normal"


def test_hypercore_replay_share_price_calculation(
    hypercore_replay_source: HypercoreDailyMetricsReplay,
    hypercore_vault_pair,
):
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

    assert first.share_price > Decimal("1")
    assert later.share_price > first.share_price
    assert first.equity_usd == Decimal("125") * first.share_price
    assert later.equity_usd == Decimal("125") * later.share_price


def test_hypercore_pricing_uses_replay_tvl_and_gating(
    hypercore_replay_source: HypercoreDailyMetricsReplay,
    hypercore_daily_metrics_frame,
    hypercore_vault_pair,
):
    pricing = HypercoreVaultPricing(
        value_func=lambda pair: Decimal("150"),
        safe_address_resolver=lambda pair: "0x0000000000000000000000000000000000000001",
        session_factory=lambda pair: None,
        simulate=False,
        market_data_source=hypercore_replay_source,
    )

    timestamp = datetime.datetime(2026, 2, 3)

    assert pricing.get_usd_tvl(timestamp, hypercore_vault_pair) == pytest.approx(308585706.340077)
    assert pricing.can_deposit(timestamp, hypercore_vault_pair) is True
    assert pricing.can_redeem(timestamp, hypercore_vault_pair) is True
    assert pricing.get_max_redemption(timestamp, hypercore_vault_pair) == Decimal("150")

    locked_source = HypercoreDailyMetricsReplay.from_single_vault_dataframe(
        hypercore_vault_pair.other_data["hypercore_vault_address"],
        hypercore_daily_metrics_frame,
        lockup_expired_after=datetime.datetime(2026, 3, 1),
    )
    locked_pricing = HypercoreVaultPricing(
        value_func=lambda pair: Decimal("150"),
        safe_address_resolver=lambda pair: "0x0000000000000000000000000000000000000001",
        session_factory=lambda pair: None,
        simulate=False,
        market_data_source=locked_source,
    )

    assert locked_pricing.get_max_redemption(timestamp, hypercore_vault_pair) == Decimal(0)
    assert locked_pricing.can_redeem(timestamp, hypercore_vault_pair) is False
