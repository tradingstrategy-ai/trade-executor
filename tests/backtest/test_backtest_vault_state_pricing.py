"""Unit tests for historical deposit/redemption availability in BacktestPricing.

The backtest pricing model answers ``can_deposit`` / ``check_redemption`` / ``get_max_*`` from a
per-(vault, timestamp) availability frame so the alpha model can skip impossible rebalances.

Critical semantics under test:

- explicit ``deposits_open=False`` (or ``max_deposit==0``) -> blocked
- explicit ``deposits_open=True`` -> allowed
- unknown / NA / pre-history / out-of-tolerance / missing pair / no state frame -> allowed
- no look-ahead: a sample stamped strictly after the decision timestamp is never used
"""
from decimal import Decimal

import pandas as pd
import pytest

from tradeexecutor.backtest.backtest_pricing import BacktestPricing
from tradingstrategy.candle import GroupedCandleUniverse


class _FakePair:
    def __init__(self, internal_id: int):
        self.internal_id = internal_id
        self.pool_address = f"0x{internal_id:040x}"

    def get_ticker(self) -> str:
        return f"VAULT{self.internal_id}-USDC"


def _candle_universe() -> GroupedCandleUniverse:
    candles = pd.DataFrame(
        [{"pair_id": 1, "timestamp": pd.Timestamp("2026-03-01"), "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 0}]
    ).set_index("timestamp", drop=False)
    return GroupedCandleUniverse(candles)


def _vault_state() -> pd.DataFrame:
    """Daily state for pair 1 (deposits) and pair 2 (redemptions)."""
    rows = [
        # pair 1 — deposits toggle open -> closed -> reopened
        _state_row(1, "2026-03-05", deposits_open=True),
        _state_row(1, "2026-03-06", deposits_open=False, deposit_closed_reason="Vault deposits disabled by leader"),
        _state_row(1, "2026-03-07", deposits_open=False, deposit_closed_reason="Vault deposits disabled by leader"),
        _state_row(1, "2026-03-10", deposits_open=True),
        # pair 1 — unknown deposits but explicit zero hard cap on this day
        _state_row(1, "2026-03-11", deposits_open=None, max_deposit=0.0),
        # pair 2 — redemptions closed on 03-06
        _state_row(2, "2026-03-05", redemption_open=None),
        _state_row(2, "2026-03-06", redemption_open=False, redemption_closed_reason="Redemptions paused", max_redeem=0.0),
        _state_row(2, "2026-03-07", redemption_open=True, max_redeem=1234.0),
    ]
    df = pd.DataFrame(rows)
    df["deposits_open"] = df["deposits_open"].astype("boolean")
    df["redemption_open"] = df["redemption_open"].astype("boolean")
    return df


def _state_row(pair_id, day, deposits_open=None, redemption_open=None, deposit_closed_reason=None, redemption_closed_reason=None, max_deposit=float("nan"), max_redeem=float("nan")):
    return {
        "pair_id": pair_id,
        "address": f"0x{pair_id:040x}",
        "timestamp": pd.Timestamp(day),
        "deposits_open": deposits_open,
        "redemption_open": redemption_open,
        "deposit_closed_reason": deposit_closed_reason,
        "redemption_closed_reason": redemption_closed_reason,
        "max_deposit": max_deposit,
        "max_redeem": max_redeem,
    }


@pytest.fixture
def pricing() -> BacktestPricing:
    return BacktestPricing(
        _candle_universe(),
        routing_model=None,
        data_delay_tolerance=pd.Timedelta("2d"),
        vault_state=_vault_state(),
    )


def test_can_deposit_open(pricing):
    assert pricing.can_deposit(pd.Timestamp("2026-03-05"), _FakePair(1)) is True


def test_can_deposit_closed(pricing):
    assert pricing.can_deposit(pd.Timestamp("2026-03-06"), _FakePair(1)) is False


def test_can_deposit_backfill_within_closed_stretch(pricing):
    # No exact sample at 03-07 12:00 -> backward-fill to the 03-07 (closed) sample.
    assert pricing.can_deposit(pd.Timestamp("2026-03-07 12:00"), _FakePair(1)) is False


def test_can_deposit_reopened(pricing):
    assert pricing.can_deposit(pd.Timestamp("2026-03-10"), _FakePair(1)) is True


def test_can_deposit_zero_hard_cap_blocks(pricing):
    # deposits_open unknown but max_deposit == 0 -> blocked.
    assert pricing.can_deposit(pd.Timestamp("2026-03-11"), _FakePair(1)) is False


def test_can_deposit_pre_history_allowed(pricing):
    # Before the first sample -> unknown -> allowed.
    assert pricing.can_deposit(pd.Timestamp("2026-03-04"), _FakePair(1)) is True


def test_can_deposit_out_of_tolerance_allowed(pricing):
    # Last pair-1 sample is 03-11; 03-25 is well beyond the 2d tolerance -> allowed.
    assert pricing.can_deposit(pd.Timestamp("2026-03-25"), _FakePair(1)) is True


def test_can_deposit_no_look_ahead(pricing):
    # At 03-05 23:00 the only sample at-or-before is 03-05 (open). The 03-06 (closed)
    # sample is in the future and must NOT be used.
    assert pricing.can_deposit(pd.Timestamp("2026-03-05 23:00"), _FakePair(1)) is True


def test_unknown_pair_allowed(pricing):
    assert pricing.can_deposit(pd.Timestamp("2026-03-06"), _FakePair(999)) is True


def test_none_timestamp_allowed(pricing):
    assert pricing.can_deposit(None, _FakePair(1)) is True


def test_no_vault_state_allowed():
    p = BacktestPricing(_candle_universe(), routing_model=None, vault_state=None)
    assert p.can_deposit(pd.Timestamp("2026-03-06"), _FakePair(1)) is True
    assert p.check_redemption(pd.Timestamp("2026-03-06"), _FakePair(1)).can_redeem is True
    assert p.get_max_deposit(pd.Timestamp("2026-03-06"), _FakePair(1)) is None


def test_check_redemption_closed(pricing):
    result = pricing.check_redemption(pd.Timestamp("2026-03-06"), _FakePair(2))
    assert result.can_redeem is False
    assert result.max_redemption == 0.0
    assert result.message == "Redemptions paused"


def test_check_redemption_unknown_allowed(pricing):
    # redemption_open is NA on 03-05 -> allowed.
    assert pricing.check_redemption(pd.Timestamp("2026-03-05"), _FakePair(2)).can_redeem is True


def test_check_redemption_open_with_cap(pricing):
    result = pricing.check_redemption(pd.Timestamp("2026-03-07"), _FakePair(2))
    assert result.can_redeem is True
    assert result.max_redemption == 1234.0


def test_get_max_redemption(pricing):
    assert pricing.get_max_redemption(pd.Timestamp("2026-03-07"), _FakePair(2)) == Decimal("1234.0")
    # Unknown cap (NA) -> None
    assert pricing.get_max_redemption(pd.Timestamp("2026-03-05"), _FakePair(2)) is None
