"""Real metadata and independent monetary oracles for automatic vault fees."""

import datetime
from dataclasses import replace
from types import SimpleNamespace
import hashlib
import json
from pathlib import Path
from decimal import Decimal

import pytest

from tradeexecutor.backtest.backtest_runner import run_backtest, run_backtest_inline, setup_backtest_for_universe
from tradeexecutor.backtest.vault_fees import FEE_KEY, calculate_vault_fees, vault_exit_rate
from tradeexecutor.backtest.backtest_execution import BacktestExecution
from tradeexecutor.backtest.simulated_wallet import SimulatedWallet
from tradeexecutor.backtest.backtest_pricing import BacktestPricing
from tradeexecutor.cli.log import setup_custom_log_levels
from tradeexecutor.state.trade import TradeStatus
from tradeexecutor.state.state import State
from tradeexecutor.strategy.strategy_module import read_strategy_module
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode
from tradeexecutor.strategy.pandas_trader.indicator import MemoryIndicatorStorage
from tradeexecutor.testing.synthetic_exchange_data import generate_simple_routing_model
from tradeexecutor.testing.vault_fee_sample import load_fee_sample, PendingFeeSnapshot


@pytest.mark.parametrize('protocol, routing', [
    ('lagoon', TradeRouting.default),
    ('lagoon', TradeRouting.user_supplied_routing_model),
    ('ipor', TradeRouting.default),
    ('lighter', TradeRouting.default),
    ('hyperliquid', TradeRouting.default),
])
def test_real_vault_fee_strategy(protocol: str, routing: TradeRouting) -> None:
    """Real metadata drives net proceeds without strategy fee code.

    1. Verify frozen input hashes and load a constructed strategy normally.
    2. Run real share prices through deposit, top-up and partial/full exits.
    3. Independently check each fee and reload the final state.
    4. Check saved pre-exit NAV, including the legacy pricing factory path.
    """
    # 1. Verify frozen input hashes and load a constructed strategy normally.
    root = Path(__file__).parent / 'fixtures/vault_externalised_fees'
    receipt = json.loads((root / 'provenance.json').read_text())
    for filename, key in [('metadata.json', 'metadata_sha256'), ('prices.parquet', 'prices_sha256')]:
        assert hashlib.sha256((root / filename).read_bytes()).hexdigest() == receipt[key]
    universe = load_fee_sample(root, protocol)
    module = read_strategy_module(Path('strategies/test_only/vault_externalised_fees.py'))
    pair = next(universe.iterate_pairs())
    metadata = pair.get_vault_metadata()
    candles = universe.data_universe.candles.df
    start = candles.timestamp.min().to_pydatetime()
    end = candles.timestamp.max().to_pydatetime()
    # Exercise the accepted boundary where execution ends at the requested load end.
    universe.data_load_end_at = end
    # 2. Run real share prices through deposit, top-up and partial/full exits.
    if routing == TradeRouting.user_supplied_routing_model:
        setup_custom_log_levels()
        setup = setup_backtest_for_universe(module, start, end, module.parameters.cycle_duration,
            10_000, universe, routing_model=generate_simple_routing_model(universe),
            create_indicators=module.create_indicators, parameters=module.parameters,
            indicator_storage=MemoryIndicatorStorage(universe.get_cache_key()),
            vault_settlement_delay=datetime.timedelta(days=1), allow_missing_fees=True)
        # Exercise the normal factory on every cycle, without a pre-created pricing model.
        setup.pricing_model = None
        result = run_backtest(setup, allow_missing_fees=True,
            execution_context=ExecutionContext(ExecutionMode.unit_testing, engine_version='0.5'))
    else:
        result = run_backtest_inline(start_at=start, end_at=end, client=None,
            decide_trades=module.decide_trades, create_indicators=module.create_indicators,
            universe=universe, cycle_duration=module.parameters.cycle_duration,
            initial_deposit=10_000, trade_routing=routing,
            routing_model=generate_simple_routing_model(universe), engine_version='0.5',
            parameters=module.parameters, mode=ExecutionMode.unit_testing,
            vault_settlement_delay=datetime.timedelta(days=1), allow_missing_fees=True)
    state = result.state
    trades = list(state.portfolio.get_all_trades())
    assert len(trades) == 4
    assert all(t.is_success() for t in trades)
    # 3. Independently check each fee and reload the final state.
    deposits = []
    for trade in trades:
        quantity = abs(trade.executed_quantity)
        if trade.is_buy():
            deposits.append([quantity, trade.executed_reserve, trade.executed_at])
            continue
        held = sum(d[0] for d in deposits)
        fraction = quantity / held
        basis = sum(d[1] for d in deposits) * fraction
        mgmt = sum(d[1] * Decimal(str((trade.executed_at-d[2]).total_seconds()/31557600)) for d in deposits) * fraction * Decimal(str(metadata.management_fee))
        record = trade.other_data.get(FEE_KEY)
        if metadata.fee_internalised:
            assert record is None
        else:
            fees = record['realised']
            gross = Decimal(fees['gross'])
            perf = max(gross-basis-mgmt, Decimal(0)) * Decimal(str(metadata.performance_fee))
            assert float(fees['management']) == pytest.approx(float(mgmt), abs=1e-6)
            assert float(fees['performance']) == pytest.approx(float(perf), abs=1e-6)
            assert float(trade.executed_reserve) == pytest.approx(float(gross-mgmt-perf), abs=1e-6)
        for deposit in deposits:
            deposit[0] *= 1-fraction
            deposit[1] *= 1-fraction
    restored = State.from_json(state.to_json())
    assert restored.portfolio.get_total_equity() == pytest.approx(state.portfolio.get_cash(), abs=1e-6)
    # 4. Check saved pre-exit NAV, including the legacy pricing factory path.
    first_sell = next(t for t in trades if t.is_sell())
    valuation = max((s for s in state.stats.positions[first_sell.position_id]
                     if s.calculated_at < first_sell.opened_at), key=lambda s: s.calculated_at)
    buys = [t for t in trades if t.is_buy() and t.executed_at <= valuation.calculated_at]
    gross_price = float(candles.loc[candles.timestamp == valuation.calculated_at, 'close'].iloc[0])
    gross = sum(float(t.executed_quantity) for t in buys) * gross_price
    principal = sum(float(t.executed_reserve) for t in buys)
    management = sum(float(t.executed_reserve) * (valuation.calculated_at - t.executed_at).total_seconds()
                     / 31557600 * metadata.management_fee for t in buys)
    performance = max(gross - principal - management, 0) * metadata.performance_fee
    expected_nav = gross if metadata.fee_internalised else gross - management - performance
    assert valuation.value == pytest.approx(expected_nav, abs=1e-6)


def test_fee_arithmetic_and_refusals() -> None:
    """Check an independent fee oracle and refuse invalid inputs before accounting.

    1. Use minimal ledger doubles to isolate arithmetic from execution machinery.
    2. Verify a half withdrawal and later settlement against hand-calculated amounts.
    3. Check losses, invalid metadata and the legacy-wrapper guard.
    """
    # 1. Use minimal ledger doubles to isolate arithmetic from execution machinery.
    root = Path(__file__).parent / 'fixtures/vault_externalised_fees'
    pair = next(load_fee_sample(root, 'lagoon').iterate_pairs())
    pair.other_data['token_metadata'] = replace(pair.get_vault_metadata(), management_fee=.02, performance_fee=.20)
    start = datetime.datetime(2024, 1, 1)
    end = start + datetime.timedelta(days=365, hours=6)
    buy = SimpleNamespace(executed_at=start, executed_price=1., pair=pair,
        get_position_quantity=lambda: Decimal(1000), is_buy=lambda: True)
    position = SimpleNamespace(pair=pair, get_successful_trades=lambda: [buy])
    # 2. Verify a half withdrawal and later settlement against hand-calculated amounts.
    result = calculate_vault_fees(position, end, Decimal(500), 1.2)
    assert float(result['management']) == pytest.approx(10)
    assert float(result['performance']) == pytest.approx(18)
    assert float(result['net']) == pytest.approx(572)
    settlement = calculate_vault_fees(position, end + datetime.timedelta(days=365, hours=6), Decimal(500), 1.4, snapshot=result)
    assert float(settlement['management']) == pytest.approx(20)
    assert float(settlement['performance']) == pytest.approx(36)
    assert float(settlement['net']) == pytest.approx(644)
    universe = load_fee_sample(root, 'lagoon')
    pricing = BacktestPricing(universe, generate_simple_routing_model(universe),
        portfolio=SimpleNamespace(get_existing_open_position_by_trading_pair=lambda p: position),
        fixed_prices={pair: 1.2})
    quote = pricing.get_sell_price(end, pair, Decimal(1000))
    assert quote.price * 1000 == pytest.approx(1144, abs=1e-6)
    assert quote.get_total_lp_fees() == 0
    assert quote.mid_price == pytest.approx(1.2)
    assert quote.market_feed_delay is None
    assert pricing.get_sell_price(end, pair, None).price == pytest.approx(1.2)
    # Without acquisition history, standalone fixed-price quotes remain gross.
    unbound_pricing = BacktestPricing(universe, generate_simple_routing_model(universe), fixed_prices={pair: 1.2})
    assert unbound_pricing.get_sell_price(end, pair, Decimal(1000)).price == pytest.approx(1.2)
    # An existing fixed exit cost is collected once after management/performance.
    pair.other_data['token_metadata'] = replace(pair.get_vault_metadata(), withdrawal_fee=.01)
    assert vault_exit_rate(pair) == 0
    pair.base.other_data['sell_tax'] = .005
    quote = pricing.get_sell_price(end, pair, Decimal(1000))
    assert quote.price * 1000 == pytest.approx(1138.28, abs=1e-6)
    # 3. Check losses, invalid metadata and the legacy-wrapper guard.
    loss = calculate_vault_fees(position, end, Decimal(500), 0)
    assert Decimal(loss['net']) == 0
    assert Decimal(loss['performance']) == 0
    metadata = pair.get_vault_metadata()
    # Old bundles omitted the mode: known adapter evidence can fill it, while
    # an unidentified protocol must still refuse rather than assume zero fees.
    pair.other_data['token_metadata'] = replace(metadata, fee_internalised=None, protocol_name='Morpho')
    assert calculate_vault_fees(position, end, Decimal(500), 1.2) is None
    pair.other_data['token_metadata'] = replace(metadata, fee_internalised=None, protocol_name='IPOR', features=['ipor_like'])
    assert calculate_vault_fees(position, end, Decimal(500), 1.2) is None
    pair.other_data['token_metadata'] = replace(metadata, fee_internalised=None, protocol_name='unidentified', features=[])
    with pytest.raises(AssertionError, match='Unknown vault fee mode'):
        calculate_vault_fees(position, end, Decimal(500), 1.2)
    pair.other_data['token_metadata'] = metadata
    pair.other_data['token_metadata'] = replace(pair.get_vault_metadata(), management_fee=float('nan'))
    with pytest.raises(AssertionError, match='Invalid vault fees'):
        calculate_vault_fees(position, end, Decimal(500), 1.2)
    execution = BacktestExecution(SimulatedWallet())
    with pytest.raises(AssertionError, match='Remove the strategy-local'):
        execution.bind_vault_fee_pricing(SimpleNamespace(_vault_redemption_fee_by_pair_id={}), None)


def test_pending_fee_recovery() -> None:
    """Reload an unresolved withdrawal and settle it twice without duplicate cash.

    1. Capture a pending sell during the constructed Lagoon strategy.
    2. Restore only normal state and wallet balances, then settle at a changed price.
    3. Re-run recovery and verify fees and cash remain unchanged.
    4. Restore a legacy pending rate snapshot without applying automatic fees again.
    """
    # 1. Capture a pending sell during the constructed Lagoon strategy.
    root = Path(__file__).parent / 'fixtures/vault_externalised_fees'
    universe = load_fee_sample(root, 'lagoon')
    module = read_strategy_module(Path('strategies/test_only/vault_externalised_fees.py'))
    candles = universe.data_universe.candles.df
    hook = PendingFeeSnapshot()
    run_backtest_inline(start_at=candles.timestamp.min().to_pydatetime(), end_at=candles.timestamp.max().to_pydatetime(),
        client=None, decide_trades=module.decide_trades, create_indicators=module.create_indicators,
        universe=universe, cycle_duration=module.parameters.cycle_duration, initial_deposit=10_000,
        trade_routing=TradeRouting.user_supplied_routing_model, routing_model=generate_simple_routing_model(universe),
        engine_version='0.5', parameters=module.parameters, mode=ExecutionMode.unit_testing,
        vault_settlement_delay=datetime.timedelta(days=2), execution_test_hook=hook, allow_missing_fees=True)
    assert hook.state_json is not None
    # 2. Restore only normal state and wallet balances, then settle at a changed price.
    state = State.from_json(hook.state_json)
    wallet = SimulatedWallet()
    wallet.balances = hook.balances.copy()
    execution = BacktestExecution(wallet)
    pending = next(t for t in state.portfolio.get_all_trades() if t.is_sell() and t.get_status() == TradeStatus.vault_settlement_pending)
    request = pending.other_data[FEE_KEY]['request'].copy()
    due = datetime.datetime.fromisoformat(pending.other_data['vault_settlement_estimated_at'])
    pricing = BacktestPricing(universe, generate_simple_routing_model(universe), portfolio=state.portfolio)
    before = wallet.get_balance(pending.pair.quote)
    assert execution.resolve_pending_vault_settlements(state, due - datetime.timedelta(days=1), pricing) == []
    assert wallet.get_balance(pending.pair.quote) == before
    resolved = execution.resolve_pending_vault_settlements(state, due, pricing)
    assert pending in resolved
    assert pending.other_data[FEE_KEY]['request'] == request
    assert float(pending.other_data[FEE_KEY]['realised']['net']) == pytest.approx(float(pending.executed_reserve), abs=1e-6)
    # 3. Re-run recovery and verify fees and cash remain unchanged.
    balance = wallet.get_balance(pending.pair.quote)
    saved = state.to_json()
    assert execution.resolve_pending_vault_settlements(state, due, pricing) == []
    assert wallet.get_balance(pending.pair.quote) == balance
    assert state.to_json() == saved

    # 4. Restore a legacy pending rate snapshot without applying automatic fees again.
    legacy = State.from_json(hook.state_json)
    old_trade = next(t for t in legacy.portfolio.get_all_trades() if t.trade_id == pending.trade_id)
    old_trade.other_data.pop(FEE_KEY)
    old_trade.other_data["backtest_vault_redemption_fee"] = .1
    old_trade.pair.other_data.pop("vault_fee_internalised", None)
    # Reloaded position and trade descriptors are separate objects.
    legacy.portfolio.get_position_by_id(old_trade.position_id).pair.other_data.pop("vault_fee_internalised", None)
    old_wallet = SimulatedWallet()
    old_wallet.balances = hook.balances.copy()
    old_pricing = BacktestPricing(universe, generate_simple_routing_model(universe), portfolio=legacy.portfolio)
    # The runner values holdings before recovery, so resuming it needs fee-mode
    # metadata. Direct recovery can still honour the saved redemption rate.
    with pytest.raises(AssertionError, match="Unknown vault fee mode"):
        old_pricing.get_sell_price(due, old_trade.pair, abs(old_trade.planned_quantity))
    old_execution = BacktestExecution(old_wallet)
    old_execution.resolve_pending_vault_settlements(legacy, due, old_pricing)
    assert old_trade.is_success()
    assert FEE_KEY not in old_trade.other_data
    assert float(old_trade.executed_reserve) == pytest.approx(float(pending.other_data[FEE_KEY]['realised']['gross']) * .9, abs=1e-6)
