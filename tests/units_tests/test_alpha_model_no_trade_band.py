"""Unit tests for the alpha model no-trade band.

Covers :py:mod:`tradeexecutor.strategy.no_trade_band` rule resolution and
:py:meth:`tradeexecutor.strategy.alpha_model.AlphaModel.apply_no_trade_band`.

Why: the band used to live in strategy modules as a fixed share of portfolio
equity, which blocked top-ups of small positions. The engine version measures
the band against each position's own target and lets ``decide_trades()`` set
different rules for the whole portfolio, a vault category and a single vault.
These tests pin the suppression rules, the cases that must never be suppressed,
equivalence with the old strategy formula, and the diagnostics written to state.

Signals are built by hand with their target fields already set, as
``calculate_target_positions()`` would leave them, so no market data is needed.
"""
import dataclasses
import datetime
import random

import pytest

from tradeexecutor.ethereum.vault.hypercore_vault import create_hypercore_vault_pair
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier, TradingPairKind
from tradeexecutor.strategy.alpha_model import AlphaModel, TradingPairSignal, TradingPairSignalFlags
from tradeexecutor.strategy.no_trade_band import (
    NoTradeBandBasis,
    NoTradeBandLevel,
    NoTradeBandPolicy,
    NoTradeBandRule,
)
from tradeexecutor.testing.synthetic_ethereum_data import generate_random_ethereum_address
from tradingstrategy.chain import ChainId


USDC = AssetIdentifier(ChainId.ethereum.value, generate_random_ethereum_address(), "USDC", 6, 999999)


def _make_pair(
    internal_id: int,
    chain_id: int = ChainId.ethereum.value,
    kind: TradingPairKind = TradingPairKind.vault,
) -> TradingPairIdentifier:
    base = AssetIdentifier(chain_id, generate_random_ethereum_address(), f"V{internal_id}", 18, internal_id)
    quote = AssetIdentifier(chain_id, generate_random_ethereum_address(), "USDC", 6, 999999)
    return TradingPairIdentifier(
        base,
        quote,
        generate_random_ethereum_address(),
        generate_random_ethereum_address(),
        internal_id=internal_id,
        kind=kind,
    )


def _make_signal(
    pair: TradingPairIdentifier,
    old_value: float,
    target: float,
    weight: float = 0.05,
    carry_forward: bool = False,
) -> TradingPairSignal:
    """A signal as calculate_target_positions() leaves it."""
    signal = TradingPairSignal(pair=pair, signal=1.0)
    signal.old_value = old_value
    signal.old_pair = pair if old_value > 0 else None
    signal.position_target = target
    signal.normalised_weight = weight
    signal.synthetic_pair = pair
    signal.carry_forward_position = carry_forward
    signal.position_adjust_usd = 0.0 if carry_forward else target - old_value
    signal.position_adjust_quantity = signal.position_adjust_usd
    return signal


def _make_alpha(signals: list[TradingPairSignal]) -> AlphaModel:
    alpha = AlphaModel(datetime.datetime(2026, 11, 12))
    alpha.signals = {s.pair.internal_id: s for s in signals}
    return alpha


def test_no_trade_band_policy_rules():
    """Rules resolve vault first, then category, then default, and reject bad values.

    1. A vault rule beats the category rule, and its key matches whatever the address case.
    2. A category rule applies to its pairs; a category without a rule falls through to the default.
    3. Band widths follow the basis and the min_usd floor; pct 0 disables a rule even with min_usd.
    4. Invalid rules and policies are rejected.
    """
    hft = _make_pair(1)
    hft_overridden = _make_pair(2)
    carry = _make_pair(3)
    other = _make_pair(4)
    categories = {1: "hft", 2: "hft", 3: "carry_without_rule", 4: None}
    default = NoTradeBandRule(pct=0.15)
    hft_rule = NoTradeBandRule(pct=0.05, min_usd=100.0)
    vault_rule = NoTradeBandRule.disabled()
    policy = NoTradeBandPolicy(
        default=default,
        categories={"hft": hft_rule},
        pairs={(hft_overridden.chain_id, hft_overridden.pool_address.upper().replace("0X", "0x")): vault_rule},
        category_function=lambda pair: categories[pair.internal_id],
    )

    # 1. A vault rule beats the category rule, and its key matches whatever the address case.
    assert policy.resolve(hft_overridden) == (vault_rule, NoTradeBandLevel.pair)

    # 2. A category rule applies to its pairs; a category without a rule falls through to the default.
    assert policy.resolve(hft) == (hft_rule, NoTradeBandLevel.category)
    assert policy.resolve(carry) == (default, NoTradeBandLevel.default)
    assert policy.resolve(other) == (default, NoTradeBandLevel.default)

    # 3. Band widths follow the basis and the min_usd floor; pct 0 disables a rule even with min_usd.
    assert default.calculate_band_usd(6_000, None) == pytest.approx(900)
    assert hft_rule.calculate_band_usd(1_000, None) == pytest.approx(100)  # floor beats 5% of 1,000
    equity_rule = NoTradeBandRule(pct=0.015, basis=NoTradeBandBasis.portfolio_equity)
    assert equity_rule.calculate_band_usd(6_000, 120_000) == pytest.approx(1_800)
    with pytest.raises(AssertionError):
        equity_rule.calculate_band_usd(6_000, None)
    assert NoTradeBandRule(pct=0.0, min_usd=500.0).calculate_band_usd(6_000, None) == 0
    one_sided = NoTradeBandRule(pct=0.1, apply_to_reductions=False)
    assert one_sided.applies_to(100.0) and not one_sided.applies_to(-100.0) and not one_sided.applies_to(0.0)

    # 4. Invalid rules and policies are rejected.
    for bad in (dict(pct=-0.1), dict(pct=float("nan")), dict(pct=float("inf")), dict(pct=0.1, min_usd=-1.0)):
        with pytest.raises(AssertionError):
            NoTradeBandRule(**bad)
    with pytest.raises(AssertionError):
        NoTradeBandPolicy(default=default, categories={"hft": hft_rule})  # no category_function
    with pytest.raises(AssertionError):
        NoTradeBandPolicy(default=default, pairs={(1, "0xAB"): default, (1, "0xab"): default})  # duplicate after lower-casing


def test_apply_no_trade_band_suppression():
    """The band suppresses only small adjustments to positions that stay open.

    1. A relative 15% band suppresses a top-up and a reduction inside it, at the strict boundary.
    2. Adjustments at or above the band, new entries, carried-forward signals and closes pass.
    3. Zero and already-ignored adjustments are left untouched, and a one-sided rule skips its other direction.
    4. Suppressed signals keep their target, carry the flag and a JSON-safe record, and are returned.
    5. Calling the method again changes nothing.
    6. A target that moves between cycles trades only once the gap leaves the band.
    """
    pairs = {i: _make_pair(i) for i in range(1, 12)}
    signals = {
        # Inside the band: top-up of 800 < 900 and reduction of 899 < 900
        1: _make_signal(pairs[1], old_value=5_200, target=6_000),
        2: _make_signal(pairs[2], old_value=6_899, target=6_000),
        # At the band (900 is not < 900), above it, new entry, carried forward, closes in both directions
        3: _make_signal(pairs[3], old_value=5_100, target=6_000),
        4: _make_signal(pairs[4], old_value=4_000, target=6_000),
        5: _make_signal(pairs[5], old_value=0, target=100),
        6: _make_signal(pairs[6], old_value=6_000, target=6_000, carry_forward=True),
        7: _make_signal(pairs[7], old_value=110, target=100, weight=0.001),
        8: _make_signal(pairs[8], old_value=90, target=100, weight=0.001),
        # Zero adjustment, already ignored, reduction under a rule for increases only
        9: _make_signal(pairs[9], old_value=6_000, target=6_000),
        10: _make_signal(pairs[10], old_value=5_500, target=6_000),
        11: _make_signal(pairs[11], old_value=6_500, target=6_000),
    }
    signals[10].position_adjust_ignored = True
    alpha = _make_alpha(list(signals.values()))
    policy = NoTradeBandPolicy(
        default=NoTradeBandRule(pct=0.15),
        categories={"top_up_only": NoTradeBandRule(pct=0.15, apply_to_reductions=False)},
        category_function=lambda pair: "top_up_only" if pair.internal_id == 11 else None,
    )

    suppressed = alpha.apply_no_trade_band(policy)

    # 1. A relative 15% band suppresses a top-up and a reduction inside it, at the strict boundary.
    assert [s.pair.internal_id for s in suppressed] == [1, 2]
    for pair_id in (1, 2):
        s = signals[pair_id]
        assert s.position_adjust_usd == 0 and s.position_adjust_quantity == 0 and s.position_adjust_ignored

    # 2. Adjustments at or above the band, new entries, carried-forward signals and closes pass.
    assert signals[3].position_adjust_usd == pytest.approx(900)
    assert signals[4].position_adjust_usd == pytest.approx(2_000)
    assert signals[5].position_adjust_usd == pytest.approx(100)
    assert signals[7].position_adjust_usd == pytest.approx(-10)
    assert signals[8].position_adjust_usd == pytest.approx(10)
    for pair_id in (3, 4, 5, 6, 7, 8):
        assert not signals[pair_id].position_adjust_ignored
        assert TradingPairSignalFlags.inside_no_trade_band not in signals[pair_id].flags

    # 3. Zero and already-ignored adjustments are left untouched, and a one-sided rule skips its other direction.
    assert not signals[9].position_adjust_ignored and "no_trade_band" not in signals[9].other_data
    assert signals[10].position_adjust_usd == pytest.approx(500) and "no_trade_band" not in signals[10].other_data
    assert signals[11].position_adjust_usd == pytest.approx(-500) and not signals[11].position_adjust_ignored

    # 4. Suppressed signals keep their target, carry the flag and a JSON-safe record, and are returned.
    assert signals[1].position_target == 6_000
    assert TradingPairSignalFlags.inside_no_trade_band in signals[1].flags
    assert signals[1].other_data["no_trade_band"] == {
        "band_usd": pytest.approx(900),
        "basis": "position_target",
        "pct": 0.15,
        "level": "default",
        "suppressed_adjust_usd": pytest.approx(800),
    }
    restored = AlphaModel.from_dict(alpha.to_dict())
    record = restored.signals[1].other_data["no_trade_band"]
    assert record["basis"] == "position_target" and record["level"] == "default"
    assert TradingPairSignalFlags.inside_no_trade_band in restored.signals[1].flags

    # 5. Calling the method again changes nothing.
    assert alpha.apply_no_trade_band(policy) == []
    assert signals[3].position_adjust_usd == pytest.approx(900)

    # 6. A target that moves between cycles trades only once the gap leaves the band.
    pair = _make_pair(20)
    held = 6_000.0
    outcomes = []
    for target in (5_500.0, 5_000.0, 5_800.0, 6_500.0):
        signal = _make_signal(pair, old_value=held, target=target)
        _make_alpha([signal]).apply_no_trade_band(policy)
        held += signal.position_adjust_usd
        outcomes.append(held)
    # 500 < 825 held; 1,000 >= 750 trades down; 800 < 870 held; 1,500 >= 975 trades up
    assert outcomes == pytest.approx([6_000, 5_000, 5_000, 6_500])


def test_apply_no_trade_band_matches_strategy_formula():
    """The legacy policy suppresses the same adjustments as the old strategy guard.

    The strategy modules applied a band of 1.5% of portfolio equity to every
    open position and let HFT reductions bypass it. The engine method must
    reproduce that for any non-zero adjustment of a position that stays open.

    1. Generate random positions, targets and HFT membership.
    2. Apply the legacy policy through the engine method.
    3. Compare with the strategy formula, signal by signal.
    """
    rng = random.Random(1)
    equity = 120_000.0
    hft_ids = set()

    # 1. Generate random positions, targets and HFT membership.
    signals = []
    for pair_id in range(1, 301):
        old_value = rng.choice([0.0, rng.uniform(500, 15_000)])
        target = rng.uniform(1_000, 15_000)
        signals.append(_make_signal(_make_pair(pair_id), old_value=old_value, target=target, weight=target / equity))
        if rng.random() < 0.4:
            hft_ids.add(pair_id)
    expected = {
        s.pair.internal_id
        for s in signals
        if s.old_value > 0
        and s.position_target > 0
        and s.position_adjust_usd != 0
        and not (s.pair.internal_id in hft_ids and s.position_adjust_usd < 0)
        and abs(s.position_adjust_usd) < equity * 0.015
    }

    # 2. Apply the legacy policy through the engine method.
    legacy = NoTradeBandRule(pct=0.015, basis=NoTradeBandBasis.portfolio_equity)
    policy = NoTradeBandPolicy(
        default=legacy,
        categories={"hft": dataclasses.replace(legacy, apply_to_reductions=False)},
        category_function=lambda pair: "hft" if pair.internal_id in hft_ids else None,
    )
    suppressed = _make_alpha(signals).apply_no_trade_band(policy, portfolio_equity=equity)

    # 3. Compare with the strategy formula, signal by signal.
    assert len(expected) > 20
    assert {s.pair.internal_id for s in suppressed} == expected


def test_apply_no_trade_band_rejects_unsupported_input():
    """The method refuses a model it cannot handle safely, and leaves HyperCore sell preparation alone.

    1. A model whose targets were not calculated (no mapped pair) is rejected.
    2. Leveraged and flipping signals are rejected.
    3. A suppressed HyperCore reduction is not sent to the redemption check by sell preparation.
    """
    policy = NoTradeBandPolicy(default=NoTradeBandRule(pct=0.15))

    # 1. A model whose targets were not calculated (no mapped pair) is rejected.
    unmapped = _make_signal(_make_pair(1), old_value=5_500, target=6_000)
    unmapped.synthetic_pair = None
    with pytest.raises(AssertionError, match="calculate_target_positions"):
        _make_alpha([unmapped]).apply_no_trade_band(policy)

    # 2. Leveraged and flipping signals are rejected.
    leveraged = _make_signal(_make_pair(2), old_value=5_500, target=6_000)
    leveraged.leverage = 2.0
    with pytest.raises(AssertionError, match="unleveraged"):
        _make_alpha([leveraged]).apply_no_trade_band(policy)
    flipping = _make_signal(_make_pair(3, kind=TradingPairKind.spot_market_hold), old_value=5_500, target=6_000)
    flipping.signal = -1.0  # short signal against a spot holding
    with pytest.raises(AssertionError, match="unleveraged"):
        _make_alpha([flipping]).apply_no_trade_band(policy)

    # 3. A suppressed HyperCore reduction is not sent to the redemption check by sell preparation.
    hypercore = create_hypercore_vault_pair(USDC, generate_random_ethereum_address(), internal_id=4)
    signal = _make_signal(hypercore, old_value=6_500, target=6_000)
    alpha = _make_alpha([signal])
    alpha.apply_no_trade_band(policy)
    manager = _RedemptionTrackingPositionManager(position=_StubPosition())
    alpha._prepare_hypercore_sell_signals(manager)
    assert signal.position_adjust_usd == 0
    assert manager.redemption_checks == 0


class _StubPosition:
    """An open position with no profit, enough for the current-position lookup."""

    def get_total_profit_usd(self) -> float:
        return 0.0

    def get_total_profit_percent(self) -> float:
        return 0.0


@dataclasses.dataclass
class _RedemptionTrackingPositionManager:
    """Counts redemption checks, so the test can show sell preparation skipped a suppressed signal.

    A stub instead of a real PositionManager, because sell preparation only
    needs the current position lookup and the pricing model's redemption check.
    """

    position: object
    redemption_checks: int = 0

    def get_current_position_for_pair(self, pair, pending: bool = False):
        return self.position

    @property
    def pricing_model(self):
        return self

    def check_redemption(self, timestamp, pair, *, stage, position):
        self.redemption_checks += 1
        raise AssertionError("A suppressed HyperCore reduction must not reach the redemption check")
