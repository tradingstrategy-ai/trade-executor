"""Unit tests for the opt-in synchronous cash cap (``cap_buys_to_sync_cash``).

Covers ``AlphaModel._cap_buys_by_realisable_sync_cash`` and its read-only
classification predicates (``_will_sell_execute``,
``_is_executable_cash_spending_buy``).

Why: the rebalance generation loop drops some sells *after* buy sizing has
assumed their proceeds — most importantly sub-threshold trims, which are only
flagged and never zeroed — so buys could overspend the reserve and crash the
backtest wallet with ``OutOfSimulatedBalance`` (hyper-ai.py cycle 2025-08-20:
a $46.72 buy against $44.10 of realisable cash). The cap scales buys down to
the cash that actually arrives; these tests pin its budget arithmetic, its
exclusion rules, and its read-only guarantee.

Small local doubles (``_Stub*``) stand in for PositionManager / pricing so the
cap can be exercised in isolation without a full backtest, mirroring the
pattern of ``test_phase_aware_alpha_model.py``. The end-to-end proof that the
opt-in fixes the real strategy is the hyper-ai.py CLI backtest itself.

The buy pair uses a non-1.0 share price (the real Scared Money ~1.07e-05) so a
quantity-vs-USD units regression would also be caught (lesson from PR #1561).
"""
import dataclasses
import datetime

import pytest

from tradeexecutor.state.identifier import (AssetIdentifier,
                                            TradingPairIdentifier,
                                            TradingPairKind)
from tradeexecutor.strategy.alpha_model import (AlphaModel, TradingPairSignal,
                                                TradingPairSignalFlags)
from tradeexecutor.testing.synthetic_ethereum_data import \
    generate_random_ethereum_address
from tradingstrategy.chain import ChainId

#: The real Scared Money share price on the failing cycle — non-1.0 so a
#: quantity/USD units mix-up cannot cancel out (see PR #1561 lessons).
SHARE_PRICE = 1.0683740059460664e-05


def _make_pair(internal_id: int, symbol: str = "VLT", kind: TradingPairKind = TradingPairKind.vault) -> TradingPairIdentifier:
    base = AssetIdentifier(ChainId.ethereum.value, generate_random_ethereum_address(), symbol, 18, internal_id)
    quote = AssetIdentifier(ChainId.ethereum.value, generate_random_ethereum_address(), "USDC", 6, 999999)
    return TradingPairIdentifier(
        base,
        quote,
        generate_random_ethereum_address(),
        generate_random_ethereum_address(),
        internal_id=internal_id,
        kind=kind,
    )


def _make_signal(pair: TradingPairIdentifier, adjust_usd: float, share_price: float = SHARE_PRICE) -> TradingPairSignal:
    signal = TradingPairSignal(pair=pair, signal=1.0)
    signal.position_adjust_usd = adjust_usd
    # Quantity in share units at the pair's share price — sign follows the adjust.
    signal.position_adjust_quantity = adjust_usd / share_price
    signal.normalised_weight = 0.12
    return signal


@dataclasses.dataclass
class _StubPosition:
    pair: TradingPairIdentifier
    pending_settlement: bool = False

    def has_pending_vault_settlement(self) -> bool:
        return self.pending_settlement


@dataclasses.dataclass
class _StubRedemptionResult:
    can_redeem: bool


@dataclasses.dataclass
class _StubPricing:
    #: internal ids whose deposit window is open
    open_pairs: set
    #: internal id -> can_redeem for check_redemption; also counts invocations
    redeemable: dict
    check_redemption_calls: int = 0

    def can_deposit(self, timestamp, pair) -> bool:
        return pair.internal_id in self.open_pairs

    def check_redemption(self, timestamp, pair, *, stage, position):
        self.check_redemption_calls += 1
        return _StubRedemptionResult(can_redeem=self.redeemable.get(pair.internal_id, True))


@dataclasses.dataclass
class _StubPositionManager:
    pricing_model: _StubPricing
    cash: float = 0.0
    #: internal ids considered blacklisted
    problematic_ids: set = dataclasses.field(default_factory=set)
    #: internal ids classified as async vault sells
    async_sell_ids: set = dataclasses.field(default_factory=set)
    #: internal id -> position for the pending=True fallback lookup
    pending_positions: dict = dataclasses.field(default_factory=dict)

    def get_current_cash(self) -> float:
        return self.cash

    def is_problematic_pair(self, pair: TradingPairIdentifier) -> bool:
        return pair.internal_id in self.problematic_ids

    def is_async_vault_sell_pair(self, pair: TradingPairIdentifier, *, position_pair: TradingPairIdentifier | None = None) -> bool:
        return pair.internal_id in self.async_sell_ids

    def get_current_position_for_pair(self, pair: TradingPairIdentifier, pending: bool = False):
        return self.pending_positions.get(pair.internal_id)


def _make_alpha(signals: list[TradingPairSignal]) -> AlphaModel:
    alpha = AlphaModel(datetime.datetime(2025, 8, 20))
    alpha.signals = {s.pair.internal_id: s for s in signals}
    return alpha


def _run_cap(
    alpha: AlphaModel,
    pm: _StubPositionManager,
    current_positions: dict | None = None,
    redemption_results: dict | None = None,
    frozen_pairs: set | None = None,
    individual_threshold: float = 5.0,
    sell_threshold: float | None = 5.0,
    headroom: float = 0.50,
) -> dict:
    redemption_results = redemption_results if redemption_results is not None else {}
    alpha._cap_buys_by_realisable_sync_cash(
        pm,
        current_positions or {},
        redemption_results,
        frozen_pairs or set(),
        individual_threshold,
        sell_threshold,
        headroom,
    )
    return redemption_results


def test_sync_cash_cap_scales_buy_to_realisable_cash():
    """The cap scales the big buy to cash + threshold-surviving sells − headroom, touching nothing else.

    Reproduces the exact hyper-ai.py cycle #19 (2025-08-20) shape that crashed
    the backtest, and proves the classification is read-only for every signal
    the cap does not scale.

    1. Nine positions: sells −13.69 / −8.61 (above the $5 sell threshold) and six
       trims −4.99, −4.97, −4.39, −4.39, −3.81, −1.13 (below it, will be dropped
       by the generation loop), plus one +46.72 vault buy at a non-1.0 share price.
    2. Cash $21.80, headroom $0.50: budget = 21.80 + (13.69 + 8.61) − 0.50 = 43.60.
    3. The buy is scaled to exactly the budget and flagged ``capped_by_sync_cash``.
    4. Every sell keeps its original adjust, gains no flags and is not marked
       ignored (read-only classification, no diagnostics drift).
    5. With ample cash the cap is a no-op: adjust unchanged, no flag.
    """
    sell_values = [-13.69, -8.61, -4.99, -4.97, -4.39, -4.39, -3.81, -1.13]
    # 1. Cycle #19 signal shape.
    sells = [_make_signal(_make_pair(100 + i), v) for i, v in enumerate(sell_values)]
    buy = _make_signal(_make_pair(200, symbol="SCARED"), 46.72)
    alpha = _make_alpha(sells + [buy])
    pm = _StubPositionManager(pricing_model=_StubPricing(open_pairs={200}, redeemable={}), cash=21.80)

    # 2-3. Budget arithmetic: only the two threshold-surviving sells fund the buy.
    _run_cap(alpha, pm)
    expected_budget = 21.80 + 13.69 + 8.61 - 0.50
    assert buy.position_adjust_usd == pytest.approx(expected_budget)
    assert TradingPairSignalFlags.capped_by_sync_cash in buy.flags

    # 4. Read-only proof: sells untouched — adjusts, flags, ignored state.
    for s, original in zip(sells, sell_values):
        assert s.position_adjust_usd == pytest.approx(original)
        assert s.flags == set()
        assert not s.position_adjust_ignored

    # 5. No-op when buys already fit the realisable cash.
    rich_buy = _make_signal(_make_pair(201), 46.72)
    alpha2 = _make_alpha([rich_buy])
    pm2 = _StubPositionManager(pricing_model=_StubPricing(open_pairs={201}, redeemable={}), cash=1000.0)
    _run_cap(alpha2, pm2)
    assert rich_buy.position_adjust_usd == pytest.approx(46.72)
    assert TradingPairSignalFlags.capped_by_sync_cash not in rich_buy.flags


def test_sync_cash_cap_excludes_non_countable_sells_and_phantom_buys():
    """Sells that will not free cash and buys that will not execute are excluded from the budget.

    A sell the generation loop drops (problematic, frozen, dust, pending
    settlement, not redeemable, async) must not fund buys; a buy the loop drops
    (below threshold, deposit window closed, pending settlement) must not
    inflate the demand and over-scale the real buy.

    1. One countable sell (−20) plus six non-countable sells (problematic,
       frozen, dust-quantity, pending-settlement, non-redeemable, async), each −50.
    2. One real buy (+60) plus three phantom buys (below threshold, closed
       window, pending settlement).
    3. Cash $10, headroom $0: budget = 10 + 20 = 30; the real buy alone is the
       demand and is scaled to 30. Phantom buys keep their adjusts, no flags.
    4. The redemption check for the non-redeemable sell ran exactly once and its
       result was stored for the generation loop to reuse.
    """
    # 1. Sells: one countable, six non-countable.
    good_sell = _make_signal(_make_pair(1), -20.0)
    problematic_sell = _make_signal(_make_pair(2), -50.0)
    frozen_sell = _make_signal(_make_pair(3), -50.0)
    dust_sell = _make_signal(_make_pair(4), -50.0)
    dust_sell.position_adjust_quantity = 0.0  # below any dust epsilon
    pending_sell = _make_signal(_make_pair(5), -50.0)
    unredeemable_sell = _make_signal(_make_pair(6), -50.0)
    async_sell = _make_signal(_make_pair(7), -50.0)

    # 2. Buys: one real, three phantom.
    real_buy = _make_signal(_make_pair(10), 60.0)
    small_buy = _make_signal(_make_pair(11), 3.0)          # below the $5 buy threshold
    closed_window_buy = _make_signal(_make_pair(12), 40.0)  # deposit window closed
    pending_buy = _make_signal(_make_pair(13), 40.0)        # opening settlement in flight

    alpha = _make_alpha([
        good_sell, problematic_sell, frozen_sell, dust_sell, pending_sell,
        unredeemable_sell, async_sell, real_buy, small_buy, closed_window_buy, pending_buy,
    ])
    pm = _StubPositionManager(
        pricing_model=_StubPricing(open_pairs={10, 11, 13}, redeemable={6: False}),
        cash=10.0,
        problematic_ids={2},
        async_sell_ids={7},
        pending_positions={
            5: _StubPosition(pair=pending_sell.pair, pending_settlement=True),
            13: _StubPosition(pair=pending_buy.pair, pending_settlement=True),
        },
    )
    current_positions = {6: _StubPosition(pair=unredeemable_sell.pair)}

    # 3. Only the good sell funds; only the real buy is scaled.
    redemption_results = _run_cap(
        alpha,
        pm,
        current_positions=current_positions,
        frozen_pairs={frozen_sell.pair},
        headroom=0.0,
    )
    assert real_buy.position_adjust_usd == pytest.approx(30.0)
    assert TradingPairSignalFlags.capped_by_sync_cash in real_buy.flags
    for phantom in (small_buy, closed_window_buy, pending_buy):
        assert TradingPairSignalFlags.capped_by_sync_cash not in phantom.flags
    assert small_buy.position_adjust_usd == pytest.approx(3.0)
    assert closed_window_buy.position_adjust_usd == pytest.approx(40.0)
    assert pending_buy.position_adjust_usd == pytest.approx(40.0)

    # 4. Redemption check ran once and is shared with the generation loop.
    assert pm.pricing_model.check_redemption_calls == 1
    assert redemption_results[6].can_redeem is False


def test_sync_cash_cap_threshold_semantics_match_generator():
    """The cap uses the generator's exact threshold gate semantics.

    ``_should_skip_signal_rebalance`` suppresses sells only when
    ``individual_rebalance_min_threshold`` is truthy, and falls back to it when
    no sell threshold is given — the cap must agree or the two would disagree
    about which sells fund the cycle.

    1. Zero individual threshold: suppression disabled — a $2 trim counts as
       funding even though it is below the old $5 threshold.
    2. Sell threshold ``None``: falls back to the individual threshold — a $4
       trim does not count, a $6 sell does.
    """
    # 1. individual threshold 0 disables sell-size suppression entirely.
    trim = _make_signal(_make_pair(1), -2.0)
    buy = _make_signal(_make_pair(10), 50.0)
    alpha = _make_alpha([trim, buy])
    pm = _StubPositionManager(pricing_model=_StubPricing(open_pairs={10}, redeemable={}), cash=10.0)
    _run_cap(alpha, pm, individual_threshold=0.0, sell_threshold=None, headroom=0.0)
    assert buy.position_adjust_usd == pytest.approx(12.0)  # 10 cash + 2 trim

    # 2. sell threshold None falls back to the individual threshold ($5).
    small_trim = _make_signal(_make_pair(2), -4.0)
    big_sell = _make_signal(_make_pair(3), -6.0)
    buy2 = _make_signal(_make_pair(11), 50.0)
    alpha2 = _make_alpha([small_trim, big_sell, buy2])
    pm2 = _StubPositionManager(pricing_model=_StubPricing(open_pairs={11}, redeemable={}), cash=10.0)
    _run_cap(alpha2, pm2, individual_threshold=5.0, sell_threshold=None, headroom=0.0)
    assert buy2.position_adjust_usd == pytest.approx(16.0)  # 10 cash + 6 sell; the 4 trim excluded


def test_sync_cash_cap_buy_predicate_scope():
    """Only unleveraged, non-flip spot/vault buys are counted and scaled.

    A Hyperliquid vault pair is ``kind=vault`` and NOT ``is_spot()`` — it must
    be included or the very buys that overshoot would be exempt (the bug this
    cap fixes). Leveraged and flip signals size from ``position_target`` and
    must never be touched.

    1. A ``kind=vault`` buy is classified executable (the #1561-style trap).
    2. A ``kind=spot_market_hold`` buy is classified executable.
    3. A leveraged buy and a long→short flip are excluded.
    4. When the cap runs over a mixed set, only the vault/spot buys are scaled.
    """
    pm = _StubPositionManager(pricing_model=_StubPricing(open_pairs={1, 2, 3, 4}, redeemable={}), cash=10.0)
    alpha = _make_alpha([])

    # 1-2. Vault and spot kinds are both cash-spending buys.
    vault_buy = _make_signal(_make_pair(1, kind=TradingPairKind.vault), 20.0)
    spot_buy = _make_signal(_make_pair(2, kind=TradingPairKind.spot_market_hold), 20.0)
    assert alpha._is_executable_cash_spending_buy(vault_buy, pm, {}, set(), 5.0)
    assert alpha._is_executable_cash_spending_buy(spot_buy, pm, {}, set(), 5.0)

    # 3. Leveraged and flipping signals are out of scope.
    leveraged_buy = _make_signal(_make_pair(3), 20.0)
    leveraged_buy.leverage = 2.0
    assert not alpha._is_executable_cash_spending_buy(leveraged_buy, pm, {}, set(), 5.0)
    flip_buy = _make_signal(_make_pair(4), 20.0)
    flip_buy.signal = -1.0  # flipping a previously long spot position to short
    flip_buy.old_pair = _make_pair(5, kind=TradingPairKind.spot_market_hold)
    assert flip_buy.is_flipping()
    assert not alpha._is_executable_cash_spending_buy(flip_buy, pm, {}, set(), 5.0)

    # 4. Cap over a mixed set: only vault/spot buys scaled, leveraged untouched.
    alpha2 = _make_alpha([vault_buy, spot_buy, leveraged_buy])
    _run_cap(alpha2, pm, headroom=0.0)
    scaled_total = vault_buy.position_adjust_usd + spot_buy.position_adjust_usd
    assert scaled_total == pytest.approx(10.0)  # scaled to cash budget
    assert leveraged_buy.position_adjust_usd == pytest.approx(20.0)
    assert TradingPairSignalFlags.capped_by_sync_cash not in leveraged_buy.flags
