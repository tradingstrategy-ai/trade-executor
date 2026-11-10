"""Integration checks for construction-time protocol data and its actual consumers."""

import datetime
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
from tradingstrategy.alternative_data.vault import convert_vault_prices_to_vault_state
from tradingstrategy.client import Client
from tradingstrategy.exchange import ExchangeUniverse
from tradingstrategy.timebucket import TimeBucket

import tradeexecutor.strategy.trading_strategy_universe as loader
from tradeexecutor.backtest.backtest_execution import BacktestExecution
from tradeexecutor.backtest.backtest_routing import BacktestRoutingIgnoredModel
from tradeexecutor.backtest.backtest_pricing import BacktestPricing
from tradeexecutor.backtest.backtest_runner import run_backtest_inline
from tradeexecutor.backtest.simulated_wallet import SimulatedWallet
from tradeexecutor.backtest.vault_fees import FEE_KEY
from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet
from tradeexecutor.strategy.vault_fee_data import validate_vault_fee_data
from tradeexecutor.strategy.vault_protocol import enrich_vault_pair
from tradeexecutor.strategy.universe_model import UniverseOptions
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
    assert atoma.get_vault_metadata().vault_protocol_rules is atoma.get_vault_protocol_rules()
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
    def decide(input: StrategyInput) -> list[TradeExecution]:
        """Open, request an exit, then verify pending proceeds remain unavailable."""
        manager = input.get_position_manager()
        if input.cycle == 1:
            return manager.open_spot(atoma, value=1000)
        if input.cycle == 5:
            assert input.state.portfolio.get_cash() == pytest.approx(9000)
            assert any(t.is_vault_settlement_in_flight() for t in input.state.portfolio.get_all_trades())
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


@pytest.mark.parametrize('with_protocol', [True, False])
def test_partial_loader_preserves_protocol_for_window_buckets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_protocol: bool) -> None:
    """The public parquet loader preserves D2 bucketing and accepts older schemas.

    1. Write hardcoded observations with and without the optional protocol field.
    2. Load through the actual partial-universe projection and parquet reader.
    3. Check D2 closes only in the next decision bucket; old files still load.
    """
    universe = create_vault_protocol_universe()
    pairs = universe.data_universe.pairs.df
    pair = next(p for p in universe.iterate_pairs() if p.get_vault_metadata().vault_name == 'HYPE++')
    pairs = pairs[pairs.address == pair.pool_address].copy()
    # 1. Write hardcoded observations with and without the optional protocol field.
    frame = pd.DataFrame([dict(chain=42161, address=pair.pool_address,
        protocol='D2 Finance', timestamp=pd.Timestamp('2026-02-11 02:32:56'),
        share_price=1., total_assets=100000., deposits_open=False, redemption_open=True)])
    if not with_protocol:
        frame = frame.drop(columns='protocol')
    path = tmp_path / 'prices.parquet'
    frame.to_parquet(path)
    # 2. Load through the actual partial-universe projection and parquet reader.
    # Stub metadata transport only: the test must not acquire external data.
    client = Mock(spec=Client)
    client.fetch_exchange_universe.return_value = ExchangeUniverse({})
    monkeypatch.setattr(loader, 'create_vault_data_client', lambda *args: object())
    monkeypatch.setattr(loader, 'load_multiple_vaults', lambda *args, **kwargs: (list(universe.data_universe.exchanges), pairs))
    options = UniverseOptions(start_at=datetime.datetime(2026, 2, 10), end_at=datetime.datetime(2026, 2, 14))
    options.vault_price_snapshot = path
    dataset = loader.load_partial_data(client, ExecutionContext(ExecutionMode.backtesting),
        TimeBucket.d1, pd.DataFrame(columns=['dex_type', 'pair_id', 'exchange_id']),
        options, vaults=[(42161, pair.pool_address)], vault_history_source='trading-strategy-website')
    # 3. Check D2 closes only in the next decision bucket; old files still load.
    expected = '2026-02-12' if with_protocol else '2026-02-11'
    assert dataset.vault_state.timestamp.iloc[0] == pd.Timestamp(expected)


def test_other_d2_unknown_history_keeps_existing_policy() -> None:
    """Keep the HYPE++ assumption from closing unrelated D2 histories.

    1. Load HYPE++ and a second D2 identity with no availability samples.
    2. Query deposits before and after the HYPE++ cutoff.
    3. Confirm only HYPE++ is refused after the cutoff.
    """
    # 1. Load HYPE++ and a second D2 identity with no availability samples.
    universe = create_vault_protocol_universe()
    hype = next(p for p in universe.iterate_pairs() if p.get_vault_metadata().vault_name == 'HYPE++')
    other = replace(hype, pool_address='0x' + '9' * 40)
    pricing = BacktestPricing(universe.data_universe.candles,
        BacktestRoutingIgnoredModel(universe.reserve_assets[0].address), pairs=universe.data_universe.pairs)
    # 2. Query deposits before and after the HYPE++ cutoff.
    assert pricing.can_deposit(datetime.datetime(2025, 1, 1), other)
    assert pricing.can_deposit(datetime.datetime(2027, 1, 1), other)
    # 3. Confirm only HYPE++ is refused after the cutoff.
    assert not pricing.can_deposit(datetime.datetime(2027, 1, 1), hype)


def test_invalid_fee_enrichment_does_not_add_global_assertion() -> None:
    """Keep invalid fee refusal in explicit universe policy rather than translation.

    1. Load a vault and replace its entry fee with an invalid value.
    2. Resolve capabilities without a global construction assertion.
    3. Check the selected-universe validator refuses the unsupported fee.
    """
    # 1. Load a vault and replace its entry fee with an invalid value.
    universe = create_vault_protocol_universe()
    pair = next(p for p in universe.iterate_pairs() if p.get_vault_metadata().vault_name == 'Generic')
    pair.other_data['token_metadata'] = replace(pair.get_vault_metadata(), deposit_fee=float('nan'))
    # 2. Resolve capabilities without a global construction assertion.
    enrich_vault_pair(pair)
    assert not pair.get_vault_protocol_rules().deposit_fee_supported
    # 3. Check the selected-universe validator refuses the unsupported fee.
    with pytest.raises(AssertionError, match='Unsupported vault deposit fee'):
        validate_vault_fee_data(universe)


def test_atoma_missing_entry_fee_override_is_data_only() -> None:
    """Preserve supported Atoma exits without treating unknown entry fees as zero.

    1. Load Atoma metadata with an unknown entry fee.
    2. Check strict fee validation refuses that missing field.
    3. Enable the override and confirm the pair is data-only.
    """
    # 1. Load Atoma metadata with an unknown entry fee.
    universe = create_vault_protocol_universe()
    pair = next(p for p in universe.iterate_pairs() if p.get_vault_metadata().vault_name == 'Atoma')
    pair.other_data['token_metadata'] = replace(pair.get_vault_metadata(), deposit_fee=None)
    enrich_vault_pair(pair)
    # 2. Check strict fee validation refuses that missing field.
    with pytest.raises(AssertionError, match='deposit fee'):
        validate_vault_fee_data(universe)
    # 3. Enable the override and confirm the pair is data-only.
    validate_vault_fee_data(universe, allow_missing=True)
    assert pair.get_ignore_reason() == 'lacks_fee_data'


def test_protocol_rules_use_other_data_and_survive_state_json() -> None:
    """Retain fee and settlement rules when State drops full vault metadata.

    1. Capture loaded Atoma rules and the expected redemption due time.
    2. Round-trip the pair through its JSON encoder, as used by State.
    3. Check restored fees and timing, then the absent-rules result.
    """
    # 1. Capture loaded Atoma rules and the expected redemption due time.
    universe = create_vault_protocol_universe()
    pair = next(p for p in universe.iterate_pairs() if p.get_vault_metadata().vault_name == 'Atoma')
    assert 'vault_protocol_rules' not in {f.name for f in fields(TradingPairIdentifier)}
    assert pair.other_data['vault_protocol_rules']['redemption_fee'] == 0.005
    ts = datetime.datetime(2025, 4, 1)
    execution = BacktestExecution(SimulatedWallet())
    expected_due = execution._get_settlement_due(pair, ts)
    expected_rules = pair.get_vault_protocol_rules()
    assert pair.get_vault_metadata().vault_protocol_rules is expected_rules
    # 2. Round-trip the pair through its JSON encoder, as used by State.
    restored = TradingPairIdentifier.from_json(pair.to_json())
    # 3. Check restored fees and timing, then the absent-rules result.
    assert restored.get_vault_protocol_rules() == expected_rules
    assert restored.get_sell_tax() == pytest.approx(0.005)
    assert restored.get_buy_tax() == 0
    assert execution._get_settlement_due(restored, ts) == expected_due == ts + datetime.timedelta(days=7)
    restored.other_data.pop('vault_protocol_rules')
    assert restored.get_vault_protocol_rules() is None
