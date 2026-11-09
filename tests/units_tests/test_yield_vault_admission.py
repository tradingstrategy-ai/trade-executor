"""Yield parking follows admission without disabling releases of existing holdings."""
import datetime
from decimal import Decimal
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier, TradingPairKind, IGNORE_REASON_REQUIRES_WHITELIST
from tradeexecutor.strategy.pandas_trader.yield_manager import YieldManager, YieldRuleset, YieldWeightingRule, YieldDecision
from tradeexecutor.strategy.vault_closure import VaultClosure, VaultClosureOutcome


class _CapturedClosurePricing:
    """Supply a captured provider assessment; query arguments do not change this fixture."""

    assessment: VaultClosure

    def get_vault_closure(self, *args, **kwargs) -> VaultClosure:
        """Return the fixture assessment; query arguments are intentionally ignored."""
        return self.assessment


def test_yield_vault_admission_keeps_releases_available() -> None:
    """An excluded or closed venue blocks purchases while its existing capital can exit.

    1. Build real pair/rule objects and substitute external pricing and trade generation.
    2. Check explicit exclusions and historical closure refuse new parking deposits.
    3. Check withdrawal still emits, then allow an unknown historical assumption.

    Pricing returns captured closure assessments and trade generation is mocked
    to inspect intents without a chain, wallet execution or provider downloads.
    """
    # 1. Build real pair/rule objects and substitute external pricing and trade generation.
    quote = AssetIdentifier(1, '0x' + '11' * 20, 'USDC', 6)
    base = AssetIdentifier(1, '0x' + '22' * 20, 'SHARE', 18)
    pair = TradingPairIdentifier(base, quote, base.address, base.address, fee=0,
        internal_id=1, kind=TradingPairKind.vault)
    rule = YieldWeightingRule(pair=pair, max_concentration=1.0)
    assessment = VaultClosure(True, VaultClosureOutcome.closed, 'historical_state', 'explicit_closed')
    pricing = _CapturedClosurePricing()
    pricing.assessment = assessment
    adjust = Mock(return_value=['trade intent'])
    manager = YieldManager(SimpleNamespace(pricing_model=pricing, adjust_position=adjust,
        state=SimpleNamespace(portfolio=SimpleNamespace(get_frozen_position=lambda p: None))),
        YieldRuleset(position_allocation=.95, buffer_pct=.01, cash_change_tolerance_usd=5,
            weights=[rule]))
    timestamp = datetime.datetime(2026, 1, 1)
    buy = {pair: YieldDecision(rule, 1.0, 200., 100.)}
    holding = SimpleNamespace(get_value=lambda: 100., get_current_price=lambda: 1.,
        get_quantity=lambda: Decimal(100), position_id=1)
    # 2. Check explicit exclusions and historical closure refuse new parking deposits.
    pair.set_ignore_reason(IGNORE_REASON_REQUIRES_WHITELIST)
    assert manager.generate_rebalance_trades(1, timestamp, {pair: holding}, buy) == []
    pair.other_data.pop('ignore_reason')
    assert manager.generate_rebalance_trades(1, timestamp, {pair: holding}, buy) == []
    adjust.assert_not_called()
    # 3. Check withdrawal still emits, then allow an unknown historical assumption.
    sell = {pair: YieldDecision(rule, 0.0, 0., 100.)}
    pair.set_ignore_reason(IGNORE_REASON_REQUIRES_WHITELIST)
    assert manager.generate_rebalance_trades(1, timestamp, {pair: holding}, sell) == ['trade intent']
    pair.other_data.pop('ignore_reason')
    assert manager.generate_rebalance_trades(1, timestamp, {pair: holding}, sell) == ['trade intent']
    assert adjust.call_args.kwargs['dollar_delta'] == pytest.approx(-100)
    assessment = VaultClosure(False, VaultClosureOutcome.unknown, 'protocol_assumption', 'missing_history')
    pricing.assessment = assessment
    assert manager.generate_rebalance_trades(1, timestamp, {pair: holding}, buy) == ['trade intent']
    assert adjust.call_args.kwargs['dollar_delta'] == pytest.approx(100)

    # Explicit entry policy blocks low-TVL parking but never the release path.
    manager.rules = replace(manager.rules, entry_eligibility=lambda pair: False)
    assert manager.generate_rebalance_trades(1, timestamp, {pair: holding}, buy) == []
    assert manager.generate_rebalance_trades(1, timestamp, {pair: holding}, sell) == ['trade intent']
