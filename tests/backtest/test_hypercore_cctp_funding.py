"""Offline integration of native exits, cash parking and CCTP funding."""
import datetime
from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

from tradeexecutor.state.state import State
from tradeexecutor.state.lighter import LighterIdleCollateral
from tradeexecutor.state.portfolio import NotEnoughMoney
from tradeexecutor.state.trade import TradeType, TradeStatus
from tradeexecutor.ethereum.cctp.planner import inject_cctp_bridge_trades
from tradeexecutor.ethereum.execution import EthereumExecution
from tradeexecutor.strategy.execution_model import ExecutionHaltableIssue
from tradeexecutor.backtest.backtest_execution import BacktestExecution
from tradeexecutor.backtest.simulated_wallet import SimulatedWallet
from tradeexecutor.backtest.backtest_routing import BacktestRoutingState

import pytest
from eth_defi.erc_4626.core import ERC4626Feature
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe

from tradeexecutor.backtest.backtest_runner import run_backtest_inline
from tradeexecutor.backtest.backtest_routing import BacktestRoutingIgnoredModel
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier, TradingPairKind
from tradeexecutor.strategy.alpha_model import AlphaModel
from tradeexecutor.strategy.strategy_module import StrategyParameters
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.execution_context import ExecutionMode
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.pandas_trader.yield_manager import YieldManager, YieldRuleset, YieldWeightingRule, YieldDecisionInput
from tradeexecutor.strategy.reserve_currency import ReserveCurrency
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, create_pair_universe_from_code
from tradeexecutor.strategy.weighting import weight_passthrouh
from tradeexecutor.testing.synthetic_exchange_data import generate_exchange
from tradeexecutor.testing.synthetic_price_data import generate_fixed_price_candles


START = datetime.datetime(2025, 5, 2)
END = datetime.datetime(2025, 5, 7)


def _universe() -> TradingStrategyUniverse:
    """Create four fixed-price vaults and two bridges without external data."""
    arb = AssetIdentifier(42161, '0xaf88d065e77c8cc2239327c5edb3a432268e5831', 'USDC', 6)
    hyper = AssetIdentifier(999, '0xb88339cb7199b77e23db6e890353e22632ba630f', 'USDC', 6)
    base = AssetIdentifier(8453, '0x833589fcd6edb6e08f4c7c32d4f71b54bda02913', 'USDC', 6)
    synthetic = AssetIdentifier(9999, '0x0000000000000000000000000000000000000999', 'USDC', 6)
    exchange = generate_exchange(1, ChainId.arbitrum, address='0x0000000000000000000000000000000000000001')
    pairs = []
    for i, (symbol, quote) in enumerate([('native', synthetic), ('cash', arb), ('hub', arb), ('satellite', base), ('physical', hyper)], 10):
        address = f'0x{i:040x}'
        pair = TradingPairIdentifier(
            AssetIdentifier(quote.chain_id, address, symbol, 18), quote, address, exchange.address,
            internal_id=i, internal_exchange_id=1, fee=0, kind=TradingPairKind.vault,
            other_data={'vault_protocol': 'hypercore' if i == 10 else 'erc_4626',
                        'vault_features': {ERC4626Feature.hypercore_native} if i == 10 else {ERC4626Feature.morpho_like}},
        )
        pairs.append(pair)
    for i, asset in enumerate([hyper, base], 20):
        pairs.append(TradingPairIdentifier(asset, arb, f'0x{i:040x}', exchange.address,
                     internal_id=i, internal_exchange_id=1, fee=0, kind=TradingPairKind.cctp_bridge, other_data={'bridge_protocol': 'cctp'}))
    pu = create_pair_universe_from_code(ChainId.arbitrum, pairs)
    candles = generate_fixed_price_candles(TimeBucket.d1, START, END, {p: 1.0 for p in pairs})
    data = Universe(time_bucket=TimeBucket.d1, chains={ChainId.arbitrum, ChainId.hypercore, ChainId.hyperliquid, ChainId.base},
                    exchanges={exchange}, pairs=pu, candles=GroupedCandleUniverse(candles), liquidity=None)
    result = TradingStrategyUniverse(data_universe=data, reserve_assets=[arb], primary_chain=ChainId.arbitrum)
    result.data_universe.pairs.exchange_universe = data.exchange_universe
    return result


def _decide(input: StrategyInput):
    """Open the native holding, then replace it through normal allocation and yield release."""
    pairs = {p.base.token_symbol: p for p in input.strategy_universe.iterate_pairs() if p.is_vault()}
    pm = input.get_position_manager()
    if input.timestamp == START:
        return pm.open_spot(pairs['native'], 600) + pm.open_spot(pairs['cash'], 300)
    if input.timestamp != datetime.datetime(2025, 5, 5):
        return []
    alpha = AlphaModel(input.timestamp)
    alpha.set_signal(pairs['hub'], 4)
    alpha.set_signal(pairs['satellite'], 5)
    alpha.select_top_signals(2)
    alpha.assign_weights(method=weight_passthrouh)
    alpha.normalise_weights(max_weight=1.0)
    alpha.update_old_weights(input.state.portfolio, portfolio_pairs=[pairs[k] for k in ('native', 'hub', 'satellite')], ignore_credit=False)
    alpha.calculate_target_positions(pm, investable_equity=900)
    trades = alpha.generate_rebalance_trades_and_triggers(pm, min_trade_threshold=0.01,
        individual_rebalance_min_threshold=0.01, sell_rebalance_min_threshold=0.01,
        execution_context=input.execution_context)
    # YieldRuleset requires a positive buffer; a negligible value isolates cash topology.
    manager = YieldManager(pm, YieldRuleset(0.90, 1e-12, 0.01, [YieldWeightingRule(pairs['cash'], 1.0)]))
    released = manager.calculate_yield_management_safe(YieldDecisionInput(
        input.execution_context.mode, input.cycle, input.timestamp,
        input.state.portfolio.get_net_asset_value(), 0, trades))
    assert sum(float(t.planned_reserve) for t in released.trades) == pytest.approx(300)
    return trades + released.trades


def _indicators(parameters, indicators, strategy_universe, execution_context):
    """Fixed schedule needs no calculated indicators."""


def test_native_exit_funds_hub_and_satellite(monkeypatch: pytest.MonkeyPatch):
    """Reproduce the omitted native sale through the real runner and reconcile the wallet.

    1. Build an entirely local universe and execute the initial native allocation.
    2. Run the mixed native exit, cash release and destination purchases.
    3. Verify bridge ordering, target holdings and whole-wallet equity.
    """
    # 1. Build an entirely local universe and execute the initial native allocation.
    universe = _universe()
    routing = BacktestRoutingIgnoredModel(universe.get_reserve_asset().address)
    observed = []
    original = BacktestExecution.simulate_spot

    def observe(execution, state, trade):
        # Observe real simulation; do not replace any funding or settlement logic.
        hub_before = execution.wallet.get_balance(universe.get_reserve_asset().address)
        hyper_address = '0xb88339cb7199b77e23db6e890353e22632ba630f'
        hyper_before = execution.wallet.get_balance(hyper_address)
        result = original(execution, state, trade)
        if trade.pair.is_hyperliquid_vault():
            observed.append((float(execution.wallet.get_balance(hyper_address) - hyper_before),
                             float(execution.wallet.get_balance(universe.get_reserve_asset().address) - hub_before)))
            assert execution.wallet.get_balance(trade.pair.quote.address) == 0
        return result

    monkeypatch.setattr(BacktestExecution, 'simulate_spot', observe)
    # 2. Run the mixed native exit, cash release and destination purchases.
    result = run_backtest_inline(start_at=START, end_at=END, client=None, decide_trades=_decide, create_indicators=_indicators,
        universe=universe, cycle_duration=CycleDuration.cycle_1d, initial_deposit=1000,
        reserve_currency=ReserveCurrency.usdc, trade_routing=TradeRouting.user_supplied_routing_model,
        routing_model=routing, three_leg_resolution=False, allow_missing_fees=True, engine_version='0.5', mode=ExecutionMode.unit_testing, parameters=StrategyParameters({'cycle_duration': CycleDuration.cycle_1d, 'initial_cash': 1000}))
    # 3. Verify bridge ordering, target holdings and whole-wallet equity.
    state = result.state
    assert observed == [(-600.0, 0.0), (600.0, 0.0)]
    assert state.portfolio.get_cash() == pytest.approx(100)
    assert state.portfolio.get_net_asset_value() == pytest.approx(1000)
    holdings = {p.pair.base.token_symbol: p.get_value() for p in state.portfolio.open_positions.values() if p.pair.is_vault()}
    assert holdings == pytest.approx({'hub': 400, 'satellite': 500})
    trades = list(state.portfolio.get_all_trades())
    assert all(t.is_success() for t in trades)
    rebalance = sorted([t for t in trades if t.opened_at == datetime.datetime(2025, 5, 5)], key=lambda t: t.get_execution_sort_position())
    bridges = [t for t in rebalance if t.pair.is_cctp_bridge()]
    assert [(t.pair.get_destination_chain_id(), t.is_sell()) for t in bridges] == [(999, True), (8453, False)]
    assert [abs(float(t.planned_quantity)) for t in bridges] == pytest.approx([600, 500])


@pytest.mark.parametrize('primary', [999, 9999])
def test_native_primary_keeps_existing_planner_behaviour(primary: int):
    """Preserve Hyper AI and the synthetic historical convention.

    1. Build native partial-sale and buy requests with an empty primary reserve.
    2. Verify the bridge planner leaves standalone native funding to its existing checks.
    """
    # 1. Build native partial-sale and buy requests with an empty primary reserve.
    universe = _universe()
    native = next(p for p in universe.iterate_pairs() if p.is_hyperliquid_vault())
    reserve = AssetIdentifier(primary, native.quote.address, 'USDC', 6)
    state = State()
    state.portfolio.initialise_reserves(reserve, reserve_token_price=1.0)
    _, sell, _ = state.create_trade(START, native, Decimal(-100), None, 1.0, TradeType.rebalance, reserve, 1.0)
    _, buy, _ = state.create_trade(START, native, None, Decimal(100), 1.0, TradeType.rebalance, reserve, 1.0)
    # 2. Verify standalone native trades are unchanged, with no new liquidity rejection.
    assert inject_cctp_bridge_trades(state, [sell, buy], SimpleNamespace(iterate_pairs=lambda: [native]),
                                    primary, START, reserve) == [sell, buy]


@pytest.mark.parametrize('exit_price,performance,net', [(0.8, 0.0, 480), (1.2, 0.1, 708)])
def test_reduced_native_payout_refuses_bridge_burn(exit_price: float, performance: float, net: int):
    """Never burn planned proceeds that the native withdrawal did not actually return.

    1. Execute a bridge, native buy and lower-price native exit using real simulation.
    2. Request the original bridge-back and dependent Base transfer through live execution.
    3. Verify refusal before signing/broadcast and durable expired-trade state.
    """
    # 1. Execute the funding and native round trip; only the transport is simulated.
    universe = _universe()
    pairs = list(universe.iterate_pairs())
    reserve = universe.get_reserve_asset()
    native = next(p for p in pairs if p.is_hyperliquid_vault())
    native.other_data.update(vault_fee_internalised=False, vault_management_fee=0.0, vault_performance_fee=performance)
    physical = next(p for p in pairs if p.base.token_symbol == 'physical')
    bridge = next(p for p in pairs if p.is_cctp_bridge() and p.get_destination_chain_id() == 999)
    base_bridge = next(p for p in pairs if p.is_cctp_bridge() and p.get_destination_chain_id() == 8453)
    state = State()
    state.portfolio.initialise_reserves(reserve, reserve_token_price=1.0)
    state.portfolio.get_default_reserve_position().quantity = Decimal(1000)
    wallet = SimulatedWallet()
    wallet.set_balance(reserve, Decimal(1000))
    execution = BacktestExecution(wallet)
    routing = BacktestRoutingIgnoredModel(reserve.address)
    routing_state = BacktestRoutingState(universe.data_universe.pairs, wallet)
    for pair, quantity, amount, price in [(bridge, None, Decimal(800), 1.0),
                                        (physical, None, Decimal(200), 1.0),
                                        (native, None, Decimal(600), 1.0),
                                        (native, Decimal(-600), None, exit_price)]:
        _, trade, _ = state.create_trade(START, pair, quantity, amount, price, TradeType.rebalance, reserve, 1.0)
        if pair.is_hyperliquid_vault() and trade.is_sell():
            forecast_state = State.from_json(state.to_json())
            forecast = inject_cctp_bridge_trades(forecast_state, [trade], universe, 42161, START, reserve)
            back_forecast = next(t for t in forecast if t.pair.is_cctp_bridge() and t.is_sell())
            assert abs(float(back_forecast.planned_quantity)) == pytest.approx(net)
        execution.execute_trades(START, state, [trade], routing, routing_state)
    assert state.portfolio.get_bridge_position_for_chain(999).get_available_bridge_capital() == pytest.approx(net)
    assert state.portfolio.get_cash() == pytest.approx(200)
    # 2. Only external signing/transport is mocked; run the real live sequential guard.
    _, back, _ = state.create_trade(START, bridge, -Decimal(str(max(600, 600 * exit_price))), None, 1.0, TradeType.rebalance, reserve, 1.0)
    _, out, _ = state.create_trade(START, base_bridge, None, Decimal(500), 1.0, TradeType.rebalance, reserve, 1.0)
    executor = MagicMock(spec=EthereumExecution)
    routing_live = MagicMock()
    routing_live.check_trade_before_execution.return_value = None
    with pytest.raises(ExecutionHaltableIssue, match='insufficient source cash'):
        EthereumExecution._execute_trades_sequentially(executor, START, state, [back, out],
            routing_live, MagicMock(), check_balances=False, rebroadcast=False, triggered=False)
    # 3. No burn or destination spend is signed, and the refusal can be saved/reloaded.
    routing_live.setup_trades.assert_not_called()
    executor._execute_trade_batch.assert_not_called()
    executor.sync_state_before_broadcast.assert_called_once()
    assert back.get_status() == out.get_status() == TradeStatus.expired
    restored = State.from_json(state.to_json())
    restored.perform_integrity_check()
    assert restored.portfolio.get_net_asset_value() == pytest.approx(400 + net)


def test_native_legacy_ledger_diagnosis_is_atomic():
    """Reject old native holdings before constructing a fictitious satellite bridge-back.

    1. Create the old-style planned native sell without a HyperEVM accounting ledger.
    2. Verify an explicit refusal leaves all existing state untouched.
    """
    # 1. Create the old-style planned native sell.
    universe = _universe()
    native = next(p for p in universe.iterate_pairs() if p.is_hyperliquid_vault())
    reserve = universe.get_reserve_asset()
    state = State()
    state.portfolio.initialise_reserves(reserve, reserve_token_price=1.0)
    _, trade, _ = state.create_trade(START, native, Decimal(-600), None, 1.0, TradeType.rebalance, reserve, 1.0)
    before = state.to_json()
    # 2. Verify refusal is atomic.
    with pytest.raises(NotEnoughMoney, match='no funding ledger'):
        inject_cctp_bridge_trades(state, [trade], universe, 42161, START, reserve)
    assert state.to_json() == before


@pytest.mark.parametrize('idle', [0, 40])
def test_lighter_synthetic_quote_uses_ethereum_cash(idle: int):
    """Keep Lighter's dataset quote out of the physical simulated wallet.

    1. Build a Lighter pool with a synthetic quote and an Ethereum bridge ledger.
    2. Execute its funding, purchase and exit using real state and simulation.
    3. Verify Ethereum cash is debited/credited while synthetic cash stays zero.
    """
    # 1. Build the synthetic pool and its physical funding route.
    universe = _universe()
    native = next(p for p in universe.iterate_pairs() if p.is_hyperliquid_vault())
    reserve = universe.get_reserve_asset()
    ethereum = AssetIdentifier(1, '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48', 'USDC', 6)
    pool = replace(native,
        base=AssetIdentifier(9998, 'lighter-pool-123', 'Lighter', 18),
        quote=AssetIdentifier(9998, '0x0000000000000000000000000000000000000998', 'USDC', 6),
        pool_address='lighter-pool-123')
    bridge = replace(next(p for p in universe.iterate_pairs() if p.is_cctp_bridge()), base=ethereum)
    state = State()
    state.portfolio.initialise_reserves(reserve, reserve_token_price=1.0)
    state.portfolio.get_default_reserve_position().quantity = Decimal(1000 - idle)
    observation = LighterIdleCollateral(1, '0x' + '22' * 20, 123, Decimal(idle), START)
    state.portfolio.lighter_idle_collateral[observation.get_identifier()] = observation
    wallet = SimulatedWallet()
    wallet.set_balance(reserve, Decimal(1000 - idle))
    execution = BacktestExecution(wallet)
    routing = BacktestRoutingIgnoredModel(reserve.address)
    routing_state = BacktestRoutingState(None, wallet)
    # 2. Execute real capital booking and wallet movements, without API calls.
    for pair, quantity, amount in [(bridge, None, Decimal(600)), (pool, None, Decimal(100)), (pool, Decimal(-100), None)]:
        _, trade, _ = state.create_trade(START, pair, quantity, amount, 1.0, TradeType.rebalance, reserve, 1.0)
        execution.execute_trades(START, state, [trade], routing, routing_state)
        if pair is pool and trade.is_buy():
            assert wallet.get_balance(ethereum.address) == pytest.approx(500 + idle)
    # 3. Native and physical ledger cash agree after the round trip.
    assert wallet.get_balance(pool.quote.address) == 0
    assert wallet.get_balance(ethereum.address) == pytest.approx(600 + idle)
    assert state.portfolio.get_net_asset_value() == pytest.approx(1000)
