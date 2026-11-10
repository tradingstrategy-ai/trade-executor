"""Priority funding keeps ordinary weighting, TVL caps and locked capital intact."""
import datetime
from types import SimpleNamespace

import pytest

from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.strategy.alpha_model import AlphaModel, TradingPairSignal
from tradeexecutor.strategy.tvl_size_risk import USDTVLSizeRiskModel


def make_model(weights):
    model = AlphaModel(datetime.datetime(2026, 4, 13))
    quote = AssetIdentifier(1, '0x' + 'f' * 40, 'USDC', 6)
    for pair_id, weight in weights.items():
        address = f'0x{pair_id:040x}'
        base = AssetIdentifier(1, address, f'V{pair_id}', 18)
        pair = TradingPairIdentifier(base, quote, address, '0x' + 'e' * 40, internal_id=pair_id)
        model.signals[pair_id] = TradingPairSignal(pair=pair, signal=weight, raw_weight=weight)
    return model


def allocate(model, priority=None, tvls=None, cap=1.0, **kwargs):
    pricing = SimpleNamespace(get_usd_tvl=lambda timestamp, pair: (tvls or {}).get(pair.internal_id, 100_000.0))
    model.normalise_weights(investable_equity=1000.0,
        size_risk_model=USDTVLSizeRiskModel(pricing, per_position_cap=0.2),
        max_weight=cap, max_positions=12, priority_pair_ids=priority, **kwargs)
    return {i: s.position_target for i, s in model.signals.items()}


def test_priority_funds_before_larger_ordinary_signal_and_keeps_original_caps():
    # 1. The weaker priority member receives its $200 cap before ordinary sizing.
    model = make_model({1: 0.01, 2: 0.6, 3: 0.3})
    targets = allocate(model, {1}, cap=0.2)
    assert targets == pytest.approx({1: 200, 2: 200, 3: 200})
    assert model.accepted_investable_equity == pytest.approx(600)
    assert sum(s.normalised_weight for s in model.signals.values()) == pytest.approx(1)

    # 2. Without a binding concentration cap, remainder preserves the 2:1 ratio.
    model = make_model({1: 0.01, 2: 0.6, 3: 0.3})
    assert allocate(model, {1}, tvls={1: 500}) == pytest.approx({1: 100, 2: 600, 3: 300})


def test_priority_tvl_overflow_and_locked_position_do_not_consume_budget_twice():
    # 1. A pending holding is already excluded from this deployable $1,000 budget.
    model = make_model({1: 0.1, 2: 0.5, 3: 0.4})
    locked = model.signals[3]
    locked.carry_forward_position = True
    locked.position_target = 250.0
    locked.old_value = 250.0
    assert allocate(model, {1, 3}, tvls={1: 100}) == pytest.approx({1: 20, 2: 980, 3: 250})
    assert locked.normalised_weight == 0
    assert model.accepted_investable_equity == pytest.approx(1000)


def test_multiple_priority_members_receive_budget_by_rank_before_ordinary():
    # 1. Higher priority score takes $600; weaker priority takes the remaining $400.
    model = make_model({1: 0.1, 2: 0.2, 3: 0.7})
    assert allocate(model, {1, 2}, cap=0.6) == pytest.approx({1: 400, 2: 600, 3: 0})


@pytest.mark.parametrize('priority', [None, set(), {999}])
def test_no_selected_priority_preserves_ordinary_sizing(priority):
    # 1. Unselected identities and empty priorities cannot change normal targets.
    assert allocate(make_model({1: 0.6, 2: 0.4}), priority) == pytest.approx({1: 600, 2: 400})


def test_unfundable_priority_releases_budget_and_zero_capacity_is_safe():
    # 1. A priority vault with zero capacity leaves all funding for ordinary members.
    assert allocate(make_model({1: 0.1, 2: 0.9}), {1}, tvls={1: 0}) == pytest.approx({1: 0, 2: 1000})
    model = make_model({1: 1.0})
    assert allocate(model, {1}, tvls={1: 0}) == {1: 0}
    assert model.signals[1].normalised_weight == 0


def test_priority_rejects_unsupported_allocator_mode():
    # 1. Callers cannot silently request priority on the unrelated waterfall path.
    with pytest.raises(ValueError, match='Priority allocation requires'):
        allocate(make_model({1: 1.0}), {1}, waterfall=True)
