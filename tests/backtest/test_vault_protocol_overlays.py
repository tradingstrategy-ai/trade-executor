"""Integration checks for construction-time protocol data and its actual consumers."""

import datetime
from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest
from tradingstrategy.alternative_data.vault import convert_vault_prices_to_vault_state

from tradeexecutor.backtest.backtest_execution import BacktestExecution
from tradeexecutor.backtest.backtest_routing import BacktestRoutingIgnoredModel
from tradeexecutor.backtest.backtest_pricing import BacktestPricing
from tradeexecutor.backtest.backtest_runner import run_backtest_inline
from tradeexecutor.backtest.simulated_wallet import SimulatedWallet
from tradeexecutor.backtest.vault_fees import FEE_KEY
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.execution_context import ExecutionMode
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet
from tradeexecutor.strategy.vault_fee_data import validate_vault_fee_data
from tradeexecutor.testing.vault_protocol_universe import create_vault_protocol_universe


def test_loaded_protocol_rules_and_real_settlement() -> None:
    """Loaded fields drive a real immediate deposit and fee-bearing delayed exit.

    1. Load raw hardcoded metadata through normal universe and pair constructors.
    2. Check direction-specific rules and matching strategy/executor timing.
    3. Run normal pricing, wallet and settlement; verify the single realised charge.
    """
    # 1. Load raw hardcoded metadata through normal universe and pair constructors.
    universe = create_vault_protocol_universe()
    pairs = {p.get_vault_metadata().vault_name: p for p in universe.iterate_pairs()}
    validate_vault_fee_data(universe)
    atoma = pairs['Atoma']
    assert universe.get_pair_by_id(atoma.internal_id) is atoma
    assert atoma.get_vault_protocol_rules().fee_profile == 'atoma_standard_exit'
    assert atoma.get_sell_tax() == pytest.approx(0.005)
    assert atoma.base.get_sell_tax() is None
    assert pairs['HYPE++'].get_vault_protocol_rules().deposit_windowed
    assert pairs['Lagoon'].get_vault_protocol_rules().deposit_requires_settlement_event
    # 2. Check direction-specific rules and matching strategy/executor timing.
    execution = BacktestExecution(SimulatedWallet())
    ts = datetime.datetime(2025, 4, 1)
    assert not execution._is_async_vault(atoma, is_buy=True)
    assert execution._is_async_vault(atoma, is_buy=False)
    assert execution._get_settlement_due(atoma, ts) == ts + datetime.timedelta(days=7)
    # A real pricing instance receives overrides through the ordinary runner binding.
    pricing = BacktestPricing(universe.data_universe.candles, BacktestRoutingIgnoredModel(universe.reserve_assets[0].address), pairs=universe.data_universe.pairs)
    execution.bind_vault_fee_pricing(pricing, SimpleNamespace())
    assert pricing.get_vault_settlement_expectation(ts, atoma)[1] == ts
    execution.vault_settlement_delay_overrides[atoma.pool_address] = datetime.timedelta(days=3)
    assert pricing.get_vault_settlement_expectation(ts, atoma)[1] == execution._get_settlement_due(atoma, ts)
    # 3. Run normal pricing, wallet and settlement; verify the single realised charge.
    def decide(input):
        manager = input.get_position_manager()
        if input.cycle == 1:
            return manager.open_spot(atoma, value=1000)
        if input.cycle == 3:
            return manager.close_spot_position(next(iter(input.state.portfolio.open_positions.values())))
        return []
    result = run_backtest_inline(start_at=ts, end_at=datetime.datetime(2025, 4, 20),
        client=None, universe=universe, decide_trades=decide,
        cycle_duration=CycleDuration.cycle_1d, initial_deposit=10000,
        trade_routing=TradeRouting.user_supplied_routing_model,
        routing_model=BacktestRoutingIgnoredModel(universe.reserve_assets[0].address),
        mode=ExecutionMode.unit_testing, allow_missing_fees=True, engine_version='0.5',
        parameters=StrategyParameters({'initial_cash': 10000, 'cycle_duration': CycleDuration.cycle_1d}),
        create_indicators=lambda timestamp, parameters, strategy_universe, execution_context: IndicatorSet())
    trades = list(result.state.portfolio.get_all_trades())
    assert len(trades) == 2 and all(t.is_success() for t in trades)
    sell = trades[-1]
    assert sell.executed_at >= sell.opened_at + datetime.timedelta(days=7)
    assert float(sell.executed_reserve) == pytest.approx(995)
    assert float(sell.other_data[FEE_KEY]['realised']['exit_fee']) == pytest.approx(5)
    assert result.state.portfolio.get_total_equity() == pytest.approx(9995)


def test_fee_override_and_observed_window_boundary() -> None:
    """Strict fees and observed availability must not hide incomplete data or look ahead.

    1. Remove an entry fee and check default refusal versus data-only override.
    2. Convert a midday D2 observation through the real upstream loader.
    3. Check assumed early windows and observed-state precedence in pricing.
    """
    # 1. Remove an entry fee and check default refusal versus data-only override.
    universe = create_vault_protocol_universe()
    pairs = {p.get_vault_metadata().vault_name: p for p in universe.iterate_pairs()}
    pairs['Generic'].other_data['token_metadata'] = replace(pairs['Generic'].get_vault_metadata(), deposit_fee=None)
    with pytest.raises(AssertionError, match='deposit fee'):
        validate_vault_fee_data(universe)
    validate_vault_fee_data(universe, allow_missing=True)
    assert pairs['Generic'].get_ignore_reason() == 'lacks_fee_data'
    # 2. Convert a midday D2 observation through the real upstream loader.
    pair = pairs['HYPE++']
    frame = pd.DataFrame([dict(chain=42161, address=pair.pool_address, protocol='D2 Finance',
        timestamp=pd.Timestamp('2026-02-11 02:32:56'), deposits_open='false', redemption_open='true')])
    state = convert_vault_prices_to_vault_state(frame)
    assert state.timestamp.iloc[0] == pd.Timestamp('2026-02-12')
    # 3. Check assumed early windows and observed-state precedence in pricing.
    pricing = BacktestPricing(universe.data_universe.candles, BacktestRoutingIgnoredModel(universe.reserve_assets[0].address),
        pairs=universe.data_universe.pairs, vault_state=state)
    early = pricing.get_vault_closure(datetime.datetime(2025, 3, 30), pair)
    assert not early.blocked and early.source == 'synthetic_schedule'
    observed = pricing.get_vault_closure(datetime.datetime(2026, 2, 12), pair)
    assert observed.blocked and observed.reason == 'window_closed'

    # Unknown later deposit evidence must be refused through the public provider
    # path too; that path forwards an explicit unknown_is_closed=None.
    assert not pricing.can_deposit(datetime.datetime(2027, 1, 1), pair)
    assert pricing.check_redemption(datetime.datetime(2027, 1, 1), pair).can_redeem
    # A missing early flag cannot hide explicit zero capacity, even while the
    # assumed schedule would have been open. Nullable reasons must be harmless.
    early_state = pd.DataFrame([dict(pair_id=pair.internal_id,
        timestamp=pd.Timestamp('2025-03-30'), deposits_open=None,
        redemption_open=None, max_deposit=0., deposit_closed_reason=float('nan'))])
    capped = BacktestPricing(universe.data_universe.candles,
        BacktestRoutingIgnoredModel(universe.reserve_assets[0].address),
        pairs=universe.data_universe.pairs, vault_state=early_state)
    assert not capped.can_deposit(datetime.datetime(2025, 3, 30), pair)
    early_state['max_deposit'] = float('nan')
    unknown = BacktestPricing(universe.data_universe.candles,
        BacktestRoutingIgnoredModel(universe.reserve_assets[0].address),
        pairs=universe.data_universe.pairs, vault_state=early_state)
    assert unknown.can_deposit(datetime.datetime(2025, 3, 30), pair)
