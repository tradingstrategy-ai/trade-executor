"""Backtest coverage for two-stage async (ERC-7540) vault deposit/redeem.

Exercises :py:class:`~tradeexecutor.backtest.backtest_execution.BacktestExecution`
simulated settlement delay: a vault deposit/redeem request does not settle on the
same cycle but on a later cycle once the configured delay has elapsed, with the
capital correctly accounted as pending in between.

The strategy logic mirrors the example module
``strategies/test_only/async_vault_backtest_example.py`` (kept inline here because
the synthetic universe is supplied by the test rather than a create_trading_universe()).
"""

import datetime
from decimal import Decimal
from itertools import chain as ichain

import pandas as pd
import pytest

from eth_defi.erc_4626.core import ERC4626Feature

from tradeexecutor.backtest.backtest_pricing import BacktestPricing
from tradeexecutor.backtest.backtest_runner import run_backtest_inline
from tradeexecutor.cli.loop import ExecutionTestHook
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier, TradingPairKind
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.reserve import ReservePosition
from tradeexecutor.state.state import State
from tradeexecutor.state.statistics import PortfolioStatistics
from tradeexecutor.state.trade import TradeExecution, TradeStatus, TradeType
from tradeexecutor.strategy.alpha_model import AlphaModel, TradingPairSignalFlags, format_signals
from tradeexecutor.strategy.asset import get_asset_amounts
from tradeexecutor.strategy.chart.definition import ChartInput
from tradeexecutor.strategy.chart.standard.vault import pending_vault_settlements
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.execution_context import ExecutionMode, unit_test_execution_context
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.reserve_currency import ReserveCurrency
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, create_pair_universe_from_code
from tradeexecutor.strategy.weighting import weight_passthrouh
from tradeexecutor.testing.synthetic_ethereum_data import generate_random_ethereum_address
from tradeexecutor.testing.synthetic_exchange_data import generate_exchange, generate_simple_routing_model
from tradeexecutor.testing.synthetic_price_data import generate_fixed_price_candles, generate_ohlcv_candles
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe
from tradingstrategy.vault import VaultMetadata


#: Fixed synthetic vault address; also the key for the per-vault settlement delay override.
VAULT_ADDRESS = "0x" + "11" * 20

#: Async settlement delay applied to the vault in these backtests.
SETTLEMENT_DELAY = datetime.timedelta(days=2)

#: Hold the position this long after opening before redeeming.
HOLD_PERIOD = datetime.timedelta(days=4)

START_AT = datetime.datetime(2024, 1, 1)
END_AT = datetime.datetime(2024, 1, 13)
INITIAL_DEPOSIT = 10_000
FIXED_PRICE = 100.0


class Parameters:
    """Minimal backtest parameters for the synthetic async vault strategy."""

    cycle_duration = CycleDuration.cycle_1d
    initial_cash = INITIAL_DEPOSIT


def create_indicators(parameters, indicators: IndicatorSet, strategy_universe, execution_context) -> None:
    """No indicators needed."""


def decide_trades(input: StrategyInput) -> list[TradeExecution]:
    """Deposit half the cash into the async vault once, hold, then redeem after HOLD_PERIOD.

    Mirrors strategies/test_only/async_vault_backtest_example.py.

    1. Deposit half the cash on the first cycle (only once ever), keeping a buffer.
    2. Do not re-deposit while the deposit is pending (position already open, cash debited).
    3. Redeem the whole position once held long enough and fully settled.
    """

    state = input.state
    timestamp = input.timestamp
    position_manager = input.get_position_manager()
    pair = next(input.strategy_universe.iterate_pairs())

    trades: list[TradeExecution] = []

    # 1. Deposit half the cash into the vault on the first cycle (only once ever).
    if not position_manager.is_any_open() and len(state.portfolio.closed_positions) == 0:
        cash = state.portfolio.get_cash()
        if cash > 1.0:
            trades += position_manager.open_spot(pair, value=cash * 0.5)
        return trades

    # 3. Redeem once held long enough and fully settled.
    if position_manager.is_any_open():
        position = next(iter(state.portfolio.open_positions.values()))
        # 2. Skip while a deposit/redeem request is still pending settlement.
        has_pending_settlement = any(
            t.get_status() == TradeStatus.vault_settlement_pending
            for t in position.trades.values()
        )
        held_long_enough = (timestamp - position.opened_at) >= HOLD_PERIOD
        if held_long_enough and not has_pending_settlement:
            trades += position_manager.close_all()

    return trades


class _SnapshotHook(ExecutionTestHook):
    """Capture portfolio + simulated-wallet state at the start of every cycle."""

    def __init__(self, reserve_address: str, share_address: str):
        self.reserve_address = reserve_address
        self.share_address = share_address
        self.snapshots: list[dict] = []

    def on_before_cycle(self, cycle: int, cycle_st: datetime.datetime, state: State, sync_model):
        wallet = sync_model.wallet

        # Capture the (single) open position's quantity, the on-chain owner balance
        # that check-accounts would expect (get_asset_amounts), and whether a request
        # is still pending. get_asset_amounts subtracts escrowed pending-redeem shares.
        position_quantity = None
        expected_base = None
        available_quantity = None
        has_pending_settlement = False
        open_positions = list(state.portfolio.open_positions.values())
        if open_positions:
            position = open_positions[0]
            position_quantity = float(position.get_quantity())
            available_quantity = float(position.get_available_trading_quantity())
            expected_base = float(sum(
                amount for asset, amount in get_asset_amounts(position)
                if asset == position.pair.base
            ))
            has_pending_settlement = any(
                t.get_status() == TradeStatus.vault_settlement_pending
                for t in position.trades.values()
            )

        self.snapshots.append({
            "cycle": cycle,
            "ts": cycle_st,
            "cash": float(state.portfolio.get_cash()),
            "pending_value": float(state.portfolio.get_vault_settlement_pending_value()),
            "equity": float(state.portfolio.calculate_total_equity()),
            "wallet_reserve": float(wallet.balances.get(self.reserve_address, Decimal(0))),
            "wallet_shares": float(wallet.balances.get(self.share_address, Decimal(0))),
            "open_positions": len(state.portfolio.open_positions),
            "closed_positions": len(state.portfolio.closed_positions),
            "position_quantity": position_quantity,
            "available_quantity": available_quantity,
            "expected_base": expected_base,
            "has_pending_settlement": has_pending_settlement,
        })


def _make_universe(rising_price: bool) -> tuple[TradingStrategyUniverse, TradingPairIdentifier]:
    """Build a single-pair synthetic universe with an async ERC-7540 vault pair."""

    chain_id = ChainId.ethereum
    exchange = generate_exchange(
        exchange_id=1,
        chain_id=chain_id,
        address=generate_random_ethereum_address(),
    )
    reserve_asset = AssetIdentifier(chain_id.value, generate_random_ethereum_address(), "USDC", 6, 1)
    share = AssetIdentifier(chain_id.value, generate_random_ethereum_address(), "vToken", 18, 2)

    vault_pair = TradingPairIdentifier(
        share,
        reserve_asset,
        VAULT_ADDRESS,
        exchange.address,
        internal_id=555,
        internal_exchange_id=exchange.exchange_id,
        fee=0,
        kind=TradingPairKind.vault,
    )
    # Features travel via token_metadata — the pairs DataFrame round-trip drops the
    # raw "vault_features" key but preserves VaultMetadata (production shape).
    vault_pair.other_data["token_metadata"] = VaultMetadata(
        vault_name="Test async vault",
        protocol_name="Test async vault",
        protocol_slug="test_async_vault",
        features=[ERC4626Feature.erc_7540_like],
        performance_fee=None,
        management_fee=None,
    )
    vault_pair.other_data["vault_protocol"] = "test_async_vault"

    pair_universe = create_pair_universe_from_code(chain_id, [vault_pair])

    if rising_price:
        candles = generate_ohlcv_candles(
            TimeBucket.d1, START_AT, END_AT,
            start_price=FIXED_PRICE,
            daily_drift=(1.01, 1.03),   # always rising
            high_drift=1.0, low_drift=1.0,
            pair_id=vault_pair.internal_id,
        )
    else:
        candles = generate_fixed_price_candles(
            TimeBucket.d1, START_AT, END_AT, {vault_pair: FIXED_PRICE},
        )

    candle_universe = GroupedCandleUniverse(candles)
    universe = Universe(
        time_bucket=TimeBucket.d1,
        chains={chain_id},
        exchanges={exchange},
        pairs=pair_universe,
        candles=candle_universe,
        liquidity=None,
    )
    strategy_universe = TradingStrategyUniverse(data_universe=universe, reserve_assets=[reserve_asset])
    strategy_universe.data_universe.pairs.exchange_universe = strategy_universe.data_universe.exchange_universe
    return strategy_universe, vault_pair


def _make_multi_vault_universe(
    vault_specs: list[tuple[str, str, set | None]],
    drifting_prices: bool = False,
) -> tuple[TradingStrategyUniverse, dict[str, TradingPairIdentifier]]:
    """Build a synthetic universe with several vault pairs.

    :param vault_specs:
        List of (share token symbol, vault address, vault features set or None).
        Passing None features creates a vault detectable as async only via a
        settlement-delay override.

    :param drifting_prices:
        Generate a gently rising share price path per vault instead of flat
        fixed prices, so request-day and settlement-day prices differ and
        settlement-time pricing is observable in tests.
    """
    chain_id = ChainId.ethereum
    exchange = generate_exchange(
        exchange_id=1,
        chain_id=chain_id,
        address=generate_random_ethereum_address(),
    )
    reserve_asset = AssetIdentifier(chain_id.value, generate_random_ethereum_address(), "USDC", 6, 1)

    pairs: dict[str, TradingPairIdentifier] = {}
    for idx, (symbol, address, features) in enumerate(vault_specs):
        share = AssetIdentifier(chain_id.value, generate_random_ethereum_address(), symbol, 18, 10 + idx)
        pair = TradingPairIdentifier(
            share,
            reserve_asset,
            address,
            exchange.address,
            internal_id=600 + idx,
            internal_exchange_id=exchange.exchange_id,
            fee=0,
            kind=TradingPairKind.vault,
        )
        if features:
            # Features must travel via token_metadata: the pairs DataFrame round-trip
            # (create_pair_universe_from_code -> translate_trading_pair) drops the raw
            # "vault_features" key but preserves VaultMetadata, matching the production
            # shape produced by translate_vault_to_trading_pair().
            pair.other_data["token_metadata"] = VaultMetadata(
                vault_name=f"Test vault {symbol}",
                protocol_name="Test async vault",
                protocol_slug="test_async_vault",
                features=list(features),
                performance_fee=None,
                management_fee=None,
            )
        pair.other_data["vault_protocol"] = "test_async_vault"
        pairs[symbol] = pair

    pair_universe = create_pair_universe_from_code(chain_id, list(pairs.values()))
    if drifting_prices:
        candles = pd.concat([
            generate_ohlcv_candles(
                TimeBucket.d1, START_AT, END_AT,
                start_price=FIXED_PRICE,
                daily_drift=(1.001, 1.003),  # always gently rising
                high_drift=1.0, low_drift=1.0,
                pair_id=p.internal_id,
            )
            for p in pairs.values()
        ])
    else:
        candles = generate_fixed_price_candles(
            TimeBucket.d1, START_AT, END_AT, {p: FIXED_PRICE for p in pairs.values()},
        )
    candle_universe = GroupedCandleUniverse(candles)
    universe = Universe(
        time_bucket=TimeBucket.d1,
        chains={chain_id},
        exchanges={exchange},
        pairs=pair_universe,
        candles=candle_universe,
        liquidity=None,
    )
    strategy_universe = TradingStrategyUniverse(data_universe=universe, reserve_assets=[reserve_asset])
    strategy_universe.data_universe.pairs.exchange_universe = strategy_universe.data_universe.exchange_universe
    return strategy_universe, pairs


def _run(
    strategy_universe: TradingStrategyUniverse,
    hook=None,
    *,
    decide=None,
    delay: datetime.timedelta = SETTLEMENT_DELAY,
    delay_overrides: dict[str, datetime.timedelta] | None = None,
):
    """Run an inline strategy against the synthetic universe with async vault settlement delays."""

    if delay_overrides is None:
        delay_overrides = {VAULT_ADDRESS: SETTLEMENT_DELAY}

    routing_model = generate_simple_routing_model(strategy_universe)
    return run_backtest_inline(
        start_at=START_AT,
        end_at=END_AT,
        client=None,
        decide_trades=decide or decide_trades,
        create_indicators=create_indicators,
        universe=strategy_universe,
        cycle_duration=CycleDuration.cycle_1d,
        initial_deposit=INITIAL_DEPOSIT,
        reserve_currency=ReserveCurrency.usdc,
        trade_routing=TradeRouting.user_supplied_routing_model,
        routing_model=routing_model,
        engine_version="0.5",
        parameters=Parameters,
        mode=ExecutionMode.unit_testing,
        allow_missing_fees=True,
        vault_settlement_delay=delay,
        vault_settlement_delay_overrides=delay_overrides,
        execution_test_hook=hook,
        name="async-vault-backtest",
    )


@pytest.mark.timeout(300)
def test_backtest_async_vault_two_stage_lifecycle():
    """Two-stage async vault deposit and redeem settle after a delay, not instantly.

    1. Build a single async vault pair with a flat price and a 2-day settlement delay.
    2. Run a deposit-hold-redeem strategy and snapshot state before every cycle.
    3. Assert the deposit and redeem each settled a full delay after their request.
    4. Assert the pending-window accounting: cash debited, request asset moved
       to the vault queue immediately, claim output credited at settlement,
       equity continuous.
    5. Assert exactly one deposit and one redeem (no double-trading) and a clean close.
    """

    # 1. Build a single async vault pair with a flat price and a 2-day settlement delay.
    strategy_universe, vault_pair = _make_universe(rising_price=False)
    hook = _SnapshotHook(vault_pair.quote.address, vault_pair.base.address)

    # 2. Run a deposit-hold-redeem strategy and snapshot state before every cycle.
    state, _, _ = _run(strategy_universe, hook=hook)

    trades = list(state.portfolio.get_all_trades())
    buys = [t for t in trades if t.is_buy()]
    sells = [t for t in trades if t.is_sell()]

    # 5. Exactly one deposit and one redeem — no double-trading.
    assert len(trades) == 2, f"Expected one deposit + one redeem, got {trades}"
    assert len(buys) == 1 and len(sells) == 1
    assert buys[0].is_success() and sells[0].is_success()

    # 3. Each request settled a full delay after it was opened (not instantly).
    deposit_delay = buys[0].executed_at - buys[0].opened_at
    redeem_delay = sells[0].executed_at - sells[0].opened_at
    assert deposit_delay >= SETTLEMENT_DELAY, f"Deposit settled too fast: {deposit_delay}"
    assert redeem_delay >= SETTLEMENT_DELAY, f"Redeem settled too fast: {redeem_delay}"

    # 4. Pending-window accounting. A pending deposit: cash and simulated wallet
    #    reserve are already debited, value is counted as pending, and shares are
    #    credited only at settlement. We deposit half the initial cash, keeping the
    #    other half as a buffer.
    deposit_amount = INITIAL_DEPOSIT * 0.5
    deposit_pending = [
        s for s in hook.snapshots
        if s["pending_value"] == pytest.approx(deposit_amount, abs=1e-6)
    ]
    assert deposit_pending, "Expected at least one cycle with the deposit pending settlement"
    for s in deposit_pending:
        assert s["cash"] == pytest.approx(deposit_amount, abs=1e-6)  # cash ledger debited at request
        assert s["wallet_shares"] == pytest.approx(0.0, abs=1e-9)    # shares not credited until claim
        assert s["wallet_reserve"] == pytest.approx(deposit_amount, abs=1e-6)  # request reserve moved to queue
        assert s["open_positions"] == 1

    # After the deposit settles: shares credited, wallet reserve partially spent, nothing pending.
    settled = [
        s for s in hook.snapshots
        if s["wallet_shares"] > 0 and s["open_positions"] == 1 and s["pending_value"] == 0
    ]
    assert settled, "Expected at least one cycle after the deposit settled"
    assert settled[0]["wallet_reserve"] == pytest.approx(deposit_amount, abs=1e-6)

    # Equity is continuous across the settlement boundaries (flat price -> ~constant).
    # Skip cycle 1: its on_before_cycle snapshot is taken before the initial deposit
    # is synced into the portfolio (funding happens inside that first tick).
    funded_snapshots = [s for s in hook.snapshots if s["cycle"] >= 2]
    assert funded_snapshots
    for s in funded_snapshots:
        assert s["equity"] == pytest.approx(INITIAL_DEPOSIT, abs=1.0), f"Equity discontinuity: {s}"

    # Part C: while a redeem is pending, get_asset_amounts() must subtract the
    # escrowed shares so check-accounts does not flag a false mismatch. During the
    # redeem-pending window the position still holds its full quantity but the
    # expected on-chain owner balance is zero (shares moved to vault escrow).
    redeem_pending = [
        s for s in hook.snapshots
        if s["has_pending_settlement"] and s["position_quantity"] and s["position_quantity"] > 0
    ]
    assert redeem_pending, "Expected a redeem-pending cycle (position held but settlement pending)"
    for s in redeem_pending:
        assert s["expected_base"] == pytest.approx(0.0, abs=1e-9), f"Escrowed shares not subtracted: {s}"
        assert s["wallet_shares"] == pytest.approx(0.0, abs=1e-9), f"Redeem request did not debit wallet shares: {s}"
    # While merely holding (settled, no pending), expected on-chain balance == full quantity.
    holding = [
        s for s in hook.snapshots
        if not s["has_pending_settlement"] and s["position_quantity"] and s["position_quantity"] > 0
    ]
    assert holding
    for s in holding:
        assert s["expected_base"] == pytest.approx(s["position_quantity"], abs=1e-9)

    # 5. Clean close: position closed, capital returned, nothing pending.
    assert len(state.portfolio.open_positions) == 0
    assert len(state.portfolio.closed_positions) == 1
    assert state.portfolio.get_cash() == pytest.approx(INITIAL_DEPOSIT, abs=1.0)
    assert state.portfolio.get_vault_settlement_pending_value() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.timeout(300)
def test_backtest_async_vault_settles_at_current_price():
    """Async settlement values deposit/redeem at the settlement-cycle price, realising drift.

    1. Build the same vault with a steadily rising price.
    2. Run the deposit-hold-redeem strategy.
    3. Assert the redeem executed at a higher price than the deposit (current, not stale price).
    4. Assert the held position therefore realised a gain over the cycle.
    """

    # 1. Build the same vault with a steadily rising price.
    strategy_universe, vault_pair = _make_universe(rising_price=True)

    # 2. Run the deposit-hold-redeem strategy.
    state, _, _ = _run(strategy_universe)

    trades = list(state.portfolio.get_all_trades())
    buys = [t for t in trades if t.is_buy()]
    sells = [t for t in trades if t.is_sell()]
    assert len(buys) == 1 and len(sells) == 1
    assert buys[0].is_success() and sells[0].is_success()

    # 3. Redeem executed at a higher price than the deposit (settlement-time pricing).
    assert sells[0].executed_price > buys[0].executed_price, (
        f"Redeem price {sells[0].executed_price} should exceed deposit price {buys[0].executed_price}"
    )

    # 4. The hold realised a gain — final equity exceeds the initial deposit.
    assert state.portfolio.get_cash() > INITIAL_DEPOSIT
    assert state.portfolio.calculate_total_equity() > INITIAL_DEPOSIT


class _PendingValueHook(ExecutionTestHook):
    """Capture portfolio pending-settlement value at the start of every cycle."""

    def __init__(self):
        self.pending_by_ts: dict[datetime.datetime, float] = {}

    def on_before_cycle(self, cycle: int, cycle_st: datetime.datetime, state: State, sync_model):
        self.pending_by_ts[cycle_st] = float(state.portfolio.get_vault_settlement_pending_value())


VAULT_A_ADDRESS = "0x" + "aa" * 20
VAULT_B_ADDRESS = "0x" + "bb" * 20
VAULT_C_ADDRESS = "0x" + "cc" * 20


@pytest.mark.timeout(300)
def test_backtest_async_vault_multiple_vaults_and_settlement_boundaries():
    """Concurrent pending operations across vaults with distinct delays settle independently.

    Also covers the settlement boundary semantics and override-only async detection:
    a zero delay settles on the next resolver run (one-cycle minimum because the request
    is created after the resolver step within a tick), a delay of exactly one cycle
    settles on its boundary (inclusive comparison), and a vault with no async features
    is still treated as async when it has a delay override.

    1. Build three vaults: A has async features and uses the global zero delay,
       B has async features and a one-cycle override, C has NO features and a
       one-day-one-hour override (async detection purely via the override).
    2. Deposit into all three on the first cycle.
    3. Assert A and B settled exactly one cycle after request, C two cycles after.
    4. Assert the pending-value trajectory shrinks as each vault settles.
    5. Assert all three positions are open and equity is continuous.
    """

    # 1. Build three vaults with distinct delay configurations.
    strategy_universe, pairs = _make_multi_vault_universe([
        ("VA", VAULT_A_ADDRESS, {ERC4626Feature.erc_7540_like}),
        ("VB", VAULT_B_ADDRESS, {ERC4626Feature.erc_7540_like}),
        ("VC", VAULT_C_ADDRESS, None),
    ])

    # 2. Deposit into all three on the first cycle.
    def decide(input: StrategyInput) -> list[TradeExecution]:
        if input.timestamp != START_AT:
            return []
        pm = input.get_position_manager()
        trades = []
        for pair in input.strategy_universe.iterate_pairs():
            trades += pm.open_spot(pair, value=2000.0)
        return trades

    hook = _PendingValueHook()
    state, _, _ = _run(
        strategy_universe,
        hook=hook,
        decide=decide,
        delay=datetime.timedelta(0),
        delay_overrides={
            VAULT_B_ADDRESS: datetime.timedelta(days=1),
            VAULT_C_ADDRESS: datetime.timedelta(days=1, hours=1),
        },
    )

    trades_by_symbol = {t.pair.base.token_symbol: t for t in state.portfolio.get_all_trades()}
    assert set(trades_by_symbol) == {"VA", "VB", "VC"}
    assert all(t.is_success() for t in trades_by_symbol.values())

    # 3. A and B settled one cycle after request, C two cycles after.
    one_cycle = datetime.timedelta(days=1)
    assert trades_by_symbol["VA"].executed_at - trades_by_symbol["VA"].opened_at == one_cycle
    assert trades_by_symbol["VB"].executed_at - trades_by_symbol["VB"].opened_at == one_cycle
    assert trades_by_symbol["VC"].executed_at - trades_by_symbol["VC"].opened_at == 2 * one_cycle

    # 4. Pending value shrinks as each vault settles (snapshots run before the resolver).
    assert hook.pending_by_ts[START_AT + one_cycle] == pytest.approx(6000.0, abs=1e-6)
    assert hook.pending_by_ts[START_AT + 2 * one_cycle] == pytest.approx(2000.0, abs=1e-6)
    assert hook.pending_by_ts[START_AT + 3 * one_cycle] == pytest.approx(0.0, abs=1e-6)

    # 5. All three positions open, equity continuous, nothing left pending.
    assert len(state.portfolio.open_positions) == 3
    assert state.portfolio.calculate_total_equity() == pytest.approx(INITIAL_DEPOSIT, abs=1.0)
    assert state.portfolio.get_vault_settlement_pending_value() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.timeout(300)
def test_backtest_async_vault_partial_redeem_and_increase():
    """A position can be increased while its deposit is pending and partially redeemed later.

    1. Deposit 4000 on cycle 1; increase by 1000 on cycle 2 while the first deposit is
       still pending (two independent pending deposits on the same position).
    2. Both deposits settle independently one delay after their own request.
    3. Partially redeem half the position; during the pending window the escrowed
       shares are excluded from both the expected on-chain balance and the available
       trading quantity, while the position still holds its full quantity.
    4. After the partial redeem settles the position stays open with the residual.
    """

    strategy_universe, vault_pair = _make_universe(rising_price=False)
    caps: dict = {}

    def decide(input: StrategyInput) -> list[TradeExecution]:
        state = input.state
        ts = input.timestamp
        pm = input.get_position_manager()
        pair = next(input.strategy_universe.iterate_pairs())

        # 1. Deposit, then increase while the first deposit is pending.
        if ts == START_AT:
            return pm.open_spot(pair, value=4000.0)
        if ts == START_AT + datetime.timedelta(days=1):
            return pm.adjust_position(pair, dollar_delta=1000.0, quantity_delta=10.0, weight=1.0)

        # 3. Partially redeem half once both deposits have settled.
        if ts == START_AT + datetime.timedelta(days=4):
            position = next(iter(state.portfolio.open_positions.values()))
            quantity = position.get_quantity()
            caps["pre_redeem_quantity"] = float(quantity)
            return pm.adjust_position(pair, dollar_delta=-2500.0, quantity_delta=-(quantity / 2), weight=0.5)
        return []

    hook = _SnapshotHook(vault_pair.quote.address, vault_pair.base.address)
    state, _, _ = _run(strategy_universe, hook=hook, decide=decide)

    trades = list(state.portfolio.get_all_trades())
    buys = [t for t in trades if t.is_buy()]
    sells = [t for t in trades if t.is_sell()]
    assert len(buys) == 2 and len(sells) == 1
    assert all(t.is_success() for t in trades)

    # 2. Both deposits settled independently, each one delay after its own request.
    for buy in buys:
        assert buy.executed_at - buy.opened_at == SETTLEMENT_DELAY

    # Position quantity before the partial redeem: 4000/100 + 1000/100 shares.
    assert caps["pre_redeem_quantity"] == pytest.approx(50.0, abs=1e-9)

    # 3. During the redeem-pending window: full quantity held, escrowed half excluded
    #    from both expected on-chain balance and available trading quantity.
    redeem_pending = [s for s in hook.snapshots if s["has_pending_settlement"] and s["position_quantity"] == pytest.approx(50.0)]
    assert redeem_pending, "Expected a snapshot during the partial redeem pending window"
    for s in redeem_pending:
        assert s["expected_base"] == pytest.approx(25.0, abs=1e-9)
        assert s["available_quantity"] == pytest.approx(25.0, abs=1e-9)
        assert s["wallet_shares"] == pytest.approx(25.0, abs=1e-9)

    # 4. Partial redeem settled: position stays open with the residual half.
    assert len(state.portfolio.open_positions) == 1
    position = next(iter(state.portfolio.open_positions.values()))
    assert position.get_quantity() == pytest.approx(Decimal(25), abs=Decimal(1e-9))
    assert state.portfolio.get_cash() == pytest.approx(INITIAL_DEPOSIT - 5000 + 2500, abs=1.0)
    assert state.portfolio.calculate_total_equity() == pytest.approx(INITIAL_DEPOSIT, abs=1.0)
    assert state.portfolio.get_vault_settlement_pending_value() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.timeout(300)
def test_backtest_async_vault_double_open_rejected():
    """Opening a second position on a vault whose deposit is pending is rejected.

    1. Deposit on cycle 1; the deposit is pending on cycle 2.
    2. A second open_spot() on the same vault pair fails: the position already
       exists, so the strategy cannot accidentally double-deposit the committed capital.
    """

    strategy_universe, _ = _make_universe(rising_price=False)

    def decide(input: StrategyInput) -> list[TradeExecution]:
        ts = input.timestamp
        pm = input.get_position_manager()
        pair = next(input.strategy_universe.iterate_pairs())
        # 1. Deposit on cycle 1.
        if ts == START_AT:
            return pm.open_spot(pair, value=4000.0)
        # 2. Second open on the same pair while pending must fail.
        if ts == START_AT + datetime.timedelta(days=1):
            return pm.open_spot(pair, value=1000.0)
        return []

    with pytest.raises(AssertionError, match="Opening a new position failed"):
        _run(strategy_universe, decide=decide)


@pytest.mark.timeout(300)
def test_backtest_async_vault_alpha_model_rebalance():
    """Alpha-model rebalancing waits for pending settlements and never double-trades.

    Exercises the documented pattern for async vaults in portfolio construction:
    skip rebalancing while any vault settlement is in flight, and size targets off
    equity excluding the pending value.

    1. Cycle 1: alpha model allocates 50% into vault A (async deposit, pending).
    2. Cycle 2: the same target weights are requested again — the pending guard
       must produce zero trades (no double-deposit of committed capital).
    3. Cycle 4: targets switch to vault B — the rebalance closes A (async redeem)
       and opens B (async deposit) in the same cycle, funded by the cash buffer.
    4. Cycle 5: targets requested again while both operations are pending — zero trades.
    5. Both settle; final portfolio holds only B with equity preserved.
    """

    strategy_universe, pairs = _make_multi_vault_universe([
        ("VA", VAULT_A_ADDRESS, {ERC4626Feature.erc_7540_like}),
        ("VB", VAULT_B_ADDRESS, {ERC4626Feature.erc_7540_like}),
    ])

    one_day = datetime.timedelta(days=1)
    schedule = {
        START_AT: {"VA": 1.0},
        START_AT + one_day: {"VA": 1.0},          # guard cycle: deposit pending
        START_AT + 3 * one_day: {"VB": 1.0},      # rotate A -> B
        START_AT + 4 * one_day: {"VB": 1.0},      # guard cycle: redeem + deposit pending
    }

    def decide(input: StrategyInput) -> list[TradeExecution]:
        state = input.state
        weights = schedule.get(input.timestamp)
        if weights is None:
            return []

        # 2./4. Guard: wait for in-flight vault settlements before rebalancing.
        has_pending = any(
            t.get_status() == TradeStatus.vault_settlement_pending
            for p in ichain(state.portfolio.open_positions.values(), state.portfolio.pending_positions.values())
            for t in p.trades.values()
        )
        if has_pending:
            return []

        pm = input.get_position_manager()
        portfolio = state.portfolio
        # Size off equity excluding capital still committed to pending deposits.
        investable = (portfolio.calculate_total_equity() - portfolio.get_vault_settlement_pending_value()) * 0.5

        alpha_model = AlphaModel(input.timestamp, close_position_weight_epsilon=0.001)
        pair_by_symbol = {p.base.token_symbol: p for p in input.strategy_universe.iterate_pairs()}
        for symbol, weight in weights.items():
            alpha_model.set_signal(pair_by_symbol[symbol], weight)
        alpha_model.select_top_signals(count=2)
        alpha_model.assign_weights(method=weight_passthrouh)
        alpha_model.normalise_weights(max_weight=1.0)
        alpha_model.update_old_weights(portfolio, ignore_credit=False)
        alpha_model.calculate_target_positions(pm, investable_equity=investable)
        return alpha_model.generate_rebalance_trades_and_triggers(
            pm,
            min_trade_threshold=0.01,
            individual_rebalance_min_threshold=0.01,
            sell_rebalance_min_threshold=0.01,
            execution_context=input.execution_context,
        )

    state, _, _ = _run(strategy_universe, decide=decide, delay=SETTLEMENT_DELAY, delay_overrides={})

    trades = list(state.portfolio.get_all_trades())
    assert all(t.is_success() for t in trades), f"Unsettled trades: {trades}"

    # 1./3. Exactly three trades: buy A, then sell A + buy B on the rotation cycle.
    assert len(trades) == 3, f"Expected 3 trades, got {trades}"
    by_kind = {(t.pair.base.token_symbol, t.is_buy()) for t in trades}
    assert by_kind == {("VA", True), ("VA", False), ("VB", True)}

    # 2./4. The guard cycles produced no trades: everything opened on cycles 1 and 4 only.
    opened_at = {t.opened_at for t in trades}
    assert opened_at == {START_AT, START_AT + 3 * one_day}, f"Trades opened at unexpected cycles: {opened_at}"

    # 5. Final portfolio: A closed, only B open, equity preserved (flat prices).
    open_symbols = {p.pair.base.token_symbol for p in state.portfolio.open_positions.values()}
    assert open_symbols == {"VB"}
    assert len(state.portfolio.closed_positions) == 1
    assert state.portfolio.calculate_total_equity() == pytest.approx(INITIAL_DEPOSIT, abs=1.0)
    assert state.portfolio.get_cash() == pytest.approx(INITIAL_DEPOSIT * 0.5, abs=1.0)
    assert state.portfolio.get_vault_settlement_pending_value() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.timeout(300)
def test_backtest_async_vault_ends_with_pending():
    """A backtest that finishes while a deposit is still pending leaves a sane final state.

    1. Deposit half the cash with a settlement delay longer than the whole backtest.
    2. Run to the end without the settlement ever resolving (the strategy's pending
       guard also prevents any redeem attempt).
    3. Final state: the trade is still pending, the committed capital is counted as
       pending value, equity is continuous and statistics generation did not crash.
    """

    strategy_universe, _ = _make_universe(rising_price=False)

    # 1.-2. Module-level strategy deposits half on cycle 1; a 30-day delay never elapses.
    state, _, _ = _run(strategy_universe, delay=datetime.timedelta(days=30), delay_overrides={})

    # 3. The deposit never settled but the final state is coherent.
    trades = list(state.portfolio.get_all_trades())
    assert len(trades) == 1
    assert trades[0].get_status() == TradeStatus.vault_settlement_pending

    assert len(state.portfolio.open_positions) == 1
    position = next(iter(state.portfolio.open_positions.values()))
    assert position.get_quantity() == 0

    deposit_amount = INITIAL_DEPOSIT * 0.5
    assert state.portfolio.get_cash() == pytest.approx(deposit_amount, abs=1e-6)
    assert state.portfolio.get_vault_settlement_pending_value() == pytest.approx(deposit_amount, abs=1e-6)
    assert state.portfolio.calculate_total_equity() == pytest.approx(INITIAL_DEPOSIT, abs=1.0)


@pytest.mark.timeout(300)
def test_backtest_async_vault_settlement_due_defaults():
    """Settlement due-time defaults and precedence for async vault backtesting.

    Async simulation switches on automatically for any vault whose metadata
    carries async features, so the defaults must be realistic rather than a
    silent next-cycle settlement.

    1. The global default delay is non-zero (two days, so the pending window
       spans a full one-day decision cycle).
    2. An ERC-7540 vault with no override settles two days after the request.
    3. An Ostium vault with no override settles the next day at the epoch
       settlement hour, regardless of the global delay.
    4. A per-vault override beats both the global default and the Ostium schedule.
    5. Override-only vaults (no features) are detected as async; plain vaults are not.
    """
    from tradeexecutor.backtest.backtest_execution import (
        BacktestExecution,
        DEFAULT_VAULT_SETTLEMENT_DELAY,
        OSTIUM_BACKTEST_SETTLEMENT_HOUR,
    )
    from tradeexecutor.backtest.simulated_wallet import SimulatedWallet

    chain_id = ChainId.ethereum
    usdc = AssetIdentifier(chain_id.value, generate_random_ethereum_address(), "USDC", 6, 1)
    exchange_address = generate_random_ethereum_address()

    def _make_pair(symbol: str, address: str, features: set | None, internal_id: int) -> TradingPairIdentifier:
        share = AssetIdentifier(chain_id.value, generate_random_ethereum_address(), symbol, 18, internal_id)
        pair = TradingPairIdentifier(
            share,
            usdc,
            address,
            exchange_address,
            internal_id=internal_id,
            internal_exchange_id=1,
            fee=0,
            kind=TradingPairKind.vault,
        )
        if features:
            pair.other_data["vault_features"] = features
        return pair

    erc_7540_pair = _make_pair("V7540", VAULT_A_ADDRESS, {ERC4626Feature.erc_7540_like}, 700)
    ostium_pair = _make_pair("VOST", VAULT_B_ADDRESS, {ERC4626Feature.ostium_like}, 701)
    override_only_pair = _make_pair("VOVR", VAULT_C_ADDRESS, None, 702)
    plain_pair = _make_pair("VPLAIN", "0x" + "dd" * 20, None, 703)

    execution = BacktestExecution(
        SimulatedWallet(),
        vault_settlement_delay_overrides={VAULT_C_ADDRESS: datetime.timedelta(hours=2)},
    )

    # 1. The global default delay is non-zero (two days).
    assert DEFAULT_VAULT_SETTLEMENT_DELAY == datetime.timedelta(days=2)
    assert execution.vault_settlement_delay == DEFAULT_VAULT_SETTLEMENT_DELAY

    ts = datetime.datetime(2024, 1, 1, 9, 30)

    # 2. ERC-7540 vault: two days after the request.
    assert execution._get_settlement_due(erc_7540_pair, ts) == datetime.datetime(2024, 1, 3, 9, 30)

    # 3. Ostium vault: next day at the epoch settlement hour.
    assert execution._get_settlement_due(ostium_pair, ts) == datetime.datetime(2024, 1, 2, OSTIUM_BACKTEST_SETTLEMENT_HOUR, 0)

    # 4. A per-vault override beats both the global default and the Ostium schedule.
    assert execution._get_settlement_due(override_only_pair, ts) == ts + datetime.timedelta(hours=2)
    execution_ostium_override = BacktestExecution(
        SimulatedWallet(),
        vault_settlement_delay_overrides={VAULT_B_ADDRESS: datetime.timedelta(hours=6)},
    )
    assert execution_ostium_override._get_settlement_due(ostium_pair, ts) == ts + datetime.timedelta(hours=6)

    # 5. Override-only vaults are async; plain vaults without features are not.
    assert execution._is_async_vault(erc_7540_pair)
    assert execution._is_async_vault(ostium_pair)
    assert execution._is_async_vault(override_only_pair)
    assert not execution._is_async_vault(plain_pair)


def _get_pending_windows(state: State) -> dict[int, list[tuple[datetime.datetime, datetime.datetime | None]]]:
    """Collect per-position half-open pending settlement windows from async vault trades.

    Settlement clears ``vault_settlement_pending_at``, so the request time comes
    from the durable ``other_data["vault_settlement_requested_at"]`` copy.

    :return:
        Mapping of position id to list of ``(requested_at, settled_at)``,
        ``settled_at`` None when still pending at the end of the backtest.
    """
    windows: dict[int, list[tuple[datetime.datetime, datetime.datetime | None]]] = {}
    for position in state.portfolio.get_all_positions(pending=True):
        for trade in position.trades.values():
            if not trade.other_data.get("vault_async_flow"):
                continue
            requested_at = datetime.datetime.fromisoformat(trade.other_data["vault_settlement_requested_at"])
            settled_at = trade.executed_at if trade.is_success() else None
            windows.setdefault(position.position_id, []).append((requested_at, settled_at))
    return windows


def _assert_no_overlapping_windows(windows: dict[int, list[tuple[datetime.datetime, datetime.datetime | None]]], end_at: datetime.datetime):
    """Assert per-position pending windows never intersect (half-open [start, end))."""
    for position_id, position_windows in windows.items():
        resolved = [(start, settled if settled is not None else end_at) for start, settled in position_windows]
        resolved.sort()
        for (start_a, end_a), (start_b, _end_b) in zip(resolved, resolved[1:]):
            assert end_a <= start_b, \
                f"Position #{position_id} has overlapping pending settlement windows: {resolved}"


@pytest.mark.timeout(300)
def test_backtest_async_vault_alpha_model_native_settlement_handling():
    """AlphaModel natively handles pending vault settlements with no manual guard in decide_trades.

    Reproduces the xchain-master-vault crash scenario in miniature: an alpha-model
    strategy rebalancing every cycle over async vaults on different settlement
    schedules, gradually ramping the allocation up and back down so deposit and
    redeem requests keep overlapping the pending windows. Before the framework
    became settlement-aware this double-allocated in-flight capital and crashed
    the simulated wallet (OutOfSimulatedBalance).

    1. Build a drifting-price universe with two async vaults on different
       settlement schedules - VA flagged erc_7540_like (2-day global delay) and
       VB an override-only async vault (3-day delay override, NO async feature
       flags in metadata) - plus one synchronous vault VS.
    2. Run an alpha-model strategy that follows the xchain pattern
       (carry_forward + locked-value subtraction) but has NO manual pending guard,
       ramping target allocation 20% -> 60% -> 30%, then rotating into VS.
    3. The backtest completes; every async trade settles after its vault's delay
       at the settlement-day candle open price (drift makes it differ from the
       request-day price).
    4. Per position, pending settlement windows never overlap (no double deposits
       or double redeems) and each async vault entered/exited over multiple cycles.
    5. Per cycle, new buys never exceed the free cash at cycle start, and equity
       stays continuous across request/settle boundaries.
    6. Pinned cycles carry the settlement_pending flag with direction-specific
       pending_deposit_usd / pending_redemption_usd diagnostics, locked value
       covers the pinned positions, and format_signals() shows the pending columns.
    7. The pending_vault_settlements chart reports deposit and redemption buffers
       only inside the pending windows.
    """

    # 1. Universe: VA flagged async, VB async only via the delay override
    # (no async metadata), VS synchronous, rising prices.
    strategy_universe, pairs = _make_multi_vault_universe(
        [
            ("VA", VAULT_A_ADDRESS, {ERC4626Feature.erc_7540_like}),
            ("VB", VAULT_B_ADDRESS, None),
            ("VS", VAULT_C_ADDRESS, None),
        ],
        drifting_prices=True,
    )
    assert not pairs["VB"].is_async_vault(), "VB must be async only via the settlement-delay override"

    one_day = datetime.timedelta(days=1)
    va_delay = SETTLEMENT_DELAY                  # 2 days, from vault features + global delay
    vb_delay = datetime.timedelta(days=3)        # per-vault override, no async feature flags

    # Target allocation per cycle: gradual entry, hold, gradual exit, rotation to VS.
    allocation_schedule = {
        START_AT + 0 * one_day: ({"VA": 1.0, "VB": 1.0}, 0.20),
        START_AT + 1 * one_day: ({"VA": 1.0, "VB": 1.0}, 0.30),   # both pending -> pinned
        START_AT + 2 * one_day: ({"VA": 1.0, "VB": 1.0}, 0.45),   # VA settled, tops up; VB pinned
        START_AT + 3 * one_day: ({"VA": 1.0, "VB": 1.0}, 0.60),   # VB settled, tops up; VA pinned
        START_AT + 4 * one_day: ({"VA": 1.0, "VB": 1.0}, 0.60),
        START_AT + 5 * one_day: ({"VA": 1.0, "VB": 1.0}, 0.40),   # partial redeems start
        START_AT + 6 * one_day: ({"VA": 1.0, "VB": 1.0}, 0.40),
        START_AT + 7 * one_day: ({"VA": 1.0, "VB": 1.0}, 0.30),
        START_AT + 8 * one_day: ({"VA": 1.0, "VB": 1.0}, 0.30),
        START_AT + 9 * one_day: ({"VS": 1.0}, 0.20),              # rotate: close VA/VB, open sync VS
        START_AT + 10 * one_day: ({"VS": 1.0}, 0.20),
        START_AT + 11 * one_day: ({"VS": 1.0}, 0.20),
    }

    # 6. Per-cycle diagnostics capture: the alpha model object is not retrievable
    # per cycle after the run, so record what the asserts below need.
    records: list[dict] = []

    def decide(input: StrategyInput) -> list[TradeExecution]:
        state = input.state
        entry = allocation_schedule.get(input.timestamp)
        if entry is None:
            return []
        weights, allocation = entry

        pm = input.get_position_manager()
        portfolio = state.portfolio

        alpha_model = AlphaModel(input.timestamp, close_position_weight_epsilon=0.001)
        pair_by_symbol = {p.base.token_symbol: p for p in input.strategy_universe.iterate_pairs()}
        for symbol, weight in weights.items():
            alpha_model.set_signal(pair_by_symbol[symbol], weight)

        # The xchain pattern: pin untradeable positions, subtract their value from the target.
        locked_position_value = alpha_model.carry_forward_non_redeemable_positions(pm)
        portfolio_target_value = portfolio.calculate_total_equity() * allocation
        deployable_target_value = max(portfolio_target_value - locked_position_value, 0.0)

        alpha_model.select_top_signals(count=3)
        alpha_model.assign_weights(method=weight_passthrouh)
        alpha_model.normalise_weights(max_weight=1.0)
        alpha_model.update_old_weights(portfolio, ignore_credit=False)
        alpha_model.calculate_target_positions(pm, investable_equity=deployable_target_value)
        trades = alpha_model.generate_rebalance_trades_and_triggers(
            pm,
            min_trade_threshold=1.0,
            individual_rebalance_min_threshold=1.0,
            sell_rebalance_min_threshold=1.0,
            execution_context=input.execution_context,
        )

        records.append({
            "ts": input.timestamp,
            "locked": locked_position_value,
            "pinned": {
                s.pair.base.token_symbol: dict(s.other_data)
                for s in alpha_model.raw_signals.values()
                if TradingPairSignalFlags.settlement_pending in s.flags
            },
            "signals_df": format_signals(alpha_model, signal_type="all"),
        })
        return trades

    reserve_address = strategy_universe.reserve_assets[0].address
    hook = _SnapshotHook(reserve_address, pairs["VA"].base.address)

    # 2.-3. Run; completion alone proves no OutOfSimulatedBalance double-allocation crash.
    state, _, _ = _run(
        strategy_universe,
        hook,
        decide=decide,
        delay=va_delay,
        delay_overrides={VAULT_B_ADDRESS: vb_delay},
    )

    trades = list(state.portfolio.get_all_trades())
    delay_by_symbol = {"VA": va_delay, "VB": vb_delay}

    # 3. Every async trade settled after its vault's delay at the settlement-day price.
    # The final VB close may legitimately end the backtest still in the queue.
    unsettled = [t for t in trades if not t.is_success()]
    assert len(unsettled) <= 1, f"Expected at most the last VB close pending, got {unsettled}"
    for t in unsettled:
        assert t.get_status() == TradeStatus.vault_settlement_pending

    async_settled = [t for t in trades if t.other_data.get("vault_async_flow") and t.is_success()]
    assert len(async_settled) >= 6, f"Expected several async settlements, got {len(async_settled)}"
    candle_universe = strategy_universe.data_universe.candles
    for t in async_settled:
        symbol = t.pair.base.token_symbol
        requested_at = datetime.datetime.fromisoformat(t.other_data["vault_settlement_requested_at"])
        assert t.executed_at - requested_at == delay_by_symbol[symbol], \
            f"Trade #{t.trade_id} ({symbol}) settled after {t.executed_at - requested_at}, expected {delay_by_symbol[symbol]}"
        # Rising drift: the settlement-day price always exceeds the request-day price.
        assert t.executed_price > t.planned_price, \
            f"Trade #{t.trade_id} did not use settlement-time pricing: {t.executed_price} vs {t.planned_price}"
        # The settlement realises exactly the settlement-day candle open price
        # (vault pairs trade fee-free, backtest pricing reads the candle open).
        settlement_day_open = float(candle_universe.get_candles_by_pair(t.pair.internal_id).loc[t.executed_at]["open"])
        assert t.executed_price == pytest.approx(settlement_day_open, rel=1e-6), \
            f"Trade #{t.trade_id} settled at {t.executed_price}, settlement-day candle open is {settlement_day_open}"

    # Synchronous vault trades settled instantly, untouched by the async flow.
    vs_trades = [t for t in trades if t.pair.base.token_symbol == "VS"]
    assert vs_trades, "Rotation into the synchronous vault never happened"
    for t in vs_trades:
        assert not t.other_data.get("vault_async_flow")
        assert t.is_success()

    # 4. No overlapping pending windows per position; gradual entry and exit.
    windows = _get_pending_windows(state)
    _assert_no_overlapping_windows(windows, END_AT)
    for symbol in ("VA", "VB"):
        symbol_trades = [t for t in trades if t.pair.base.token_symbol == symbol]
        buys = [t for t in symbol_trades if t.is_buy()]
        sells = [t for t in symbol_trades if t.is_sell()]
        assert len(buys) >= 2, f"{symbol} should have entered gradually over multiple deposits, got {len(buys)}"
        assert len(sells) >= 2, f"{symbol} should have exited gradually over multiple redeems, got {len(sells)}"

    # 5. Per cycle: new buys never exceed the free cash at cycle start; equity continuous.
    # The first snapshot is taken before the initial treasury deposit is credited,
    # so the opening cycle measures against the initial cash instead.
    cash_by_cycle = {s["ts"]: s["cash"] for s in hook.snapshots}
    cash_by_cycle[START_AT] = max(cash_by_cycle.get(START_AT, 0.0), float(INITIAL_DEPOSIT))
    for ts, cash in cash_by_cycle.items():
        cycle_buy_volume = sum(float(t.planned_reserve) for t in trades if t.is_buy() and t.opened_at == ts)
        assert cycle_buy_volume <= cash + 1e-6, \
            f"Cycle {ts} allocated {cycle_buy_volume} USD of buys with only {cash} USD free cash"
    # Drop the opening snapshot taken before the initial treasury deposit credited.
    equities = [s["equity"] for s in hook.snapshots if s["equity"] > 0]
    for previous, current in zip(equities, equities[1:]):
        assert abs(current - previous) <= previous * 0.01, \
            f"Equity jumped {previous} -> {current}; in-flight capital dropped out of the books"

    # 6. Pinned-cycle diagnostics: settlement_pending flag, direction-specific values,
    # locked value covering the pins, and the format_signals() pending columns
    # showing the exact same dollar amounts.
    pinned_records = [r for r in records if r["pinned"]]
    assert pinned_records, "No cycle ever pinned a settlement-pending position"
    deposit_pins = 0
    redemption_pins = 0
    for record in pinned_records:
        assert record["locked"] > 0
        for symbol, other_data in record["pinned"].items():
            assert "pending_deposit_usd" in other_data or "pending_redemption_usd" in other_data
            assert other_data.get("pending_deposit_usd", 1.0) > 0
            assert other_data.get("pending_redemption_usd", 1.0) > 0
            deposit_pins += "pending_deposit_usd" in other_data
            redemption_pins += "pending_redemption_usd" in other_data
            row = record["signals_df"].loc[record["signals_df"].index.str.startswith(symbol)]
            assert len(row) == 1, f"Expected one diagnostics row for {symbol}, got {row}"
            for column, key in (("Pending deposit USD", "pending_deposit_usd"), ("Pending redemption USD", "pending_redemption_usd")):
                table_value = row.iloc[0][column]
                if key in other_data:
                    assert table_value == pytest.approx(other_data[key]), \
                        f"{symbol} {column} shows {table_value}, signal diagnostics say {other_data[key]}"
                else:
                    assert table_value == "-", f"{symbol} {column} should be blank, got {table_value}"
    assert deposit_pins > 0, "No cycle pinned a pending deposit"
    assert redemption_pins > 0, "No cycle pinned a pending redemption"

    # Unpinned cycles carry no settlement flags and no pending diagnostics anywhere.
    unpinned_records = [r for r in records if not r["pinned"]]
    assert unpinned_records, "Expected some cycles with nothing pending"
    for record in unpinned_records:
        df_ = record["signals_df"]
        if df_.empty:
            continue
        assert not df_["Flags"].str.contains("settlement_pending").any(), \
            f"Unpinned cycle {record['ts']} has settlement_pending flags: {df_['Flags']}"
        assert (df_["Pending deposit USD"] == "-").all(), \
            f"Unpinned cycle {record['ts']} shows pending deposit values: {df_['Pending deposit USD']}"
        assert (df_["Pending redemption USD"] == "-").all(), \
            f"Unpinned cycle {record['ts']} shows pending redemption values: {df_['Pending redemption USD']}"

    # 7. The chart shows buffers only inside the pending windows.
    chart_input = ChartInput(execution_context=unit_test_execution_context, state=state)
    _fig, chart_df = pending_vault_settlements(chart_input)
    assert not chart_df.empty
    deposit_columns = [c for c in chart_df.columns if c.endswith("deposit")]
    redeem_columns = [c for c in chart_df.columns if c.endswith("redeem")]
    assert deposit_columns and redeem_columns, f"Chart misses buffer series: {list(chart_df.columns)}"
    assert (chart_df[deposit_columns] >= 0).all().all()
    assert (chart_df[redeem_columns] <= 0).all().all()
    all_windows = [w for ws in windows.values() for w in ws]
    for ts, row in chart_df.iterrows():
        inside_any_window = any(
            start <= ts < (settled if settled is not None else END_AT)
            for start, settled in all_windows
        )
        if not inside_any_window:
            assert row.abs().sum() == pytest.approx(0.0, abs=1e-6), \
                f"Chart shows a buffer at {ts} outside every pending window: {row}"


@pytest.mark.timeout(300)
def test_backtest_async_vault_naive_alpha_model_guarded():
    """A naive alpha-model strategy without carry-forward cannot double-trade pending positions.

    The belt-and-braces skip in trade generation protects strategies that never
    call carry_forward_non_redeemable_positions(): a position with an in-flight
    settlement produces no trades, full stop.

    1. Run a deliberately naive strategy: rebalance every cycle, size off total
       equity (which includes the pending deposit value), no pending guard and
       no carry-forward call.
    2. Cycle 2 raises the target while the first deposit is still pending - the
       framework must skip the signal instead of double-depositing.
    3. After settlement the raised target executes as a second deposit; windows
       never overlap and the backtest completes.
    """

    strategy_universe, pairs = _make_multi_vault_universe([
        ("VA", VAULT_A_ADDRESS, {ERC4626Feature.erc_7540_like}),
    ])

    allocation_schedule = {
        START_AT + n * datetime.timedelta(days=1): 0.5 if n == 0 else 0.8
        for n in range(6)
    }
    flags_by_cycle: dict[datetime.datetime, set] = {}

    def decide(input: StrategyInput) -> list[TradeExecution]:
        state = input.state
        allocation = allocation_schedule.get(input.timestamp)
        if allocation is None:
            return []

        pm = input.get_position_manager()
        portfolio = state.portfolio
        # 1. Naive sizing: includes capital already committed to the pending deposit.
        investable = portfolio.calculate_total_equity() * allocation

        alpha_model = AlphaModel(input.timestamp, close_position_weight_epsilon=0.001)
        alpha_model.set_signal(pairs["VA"], 1.0)
        alpha_model.select_top_signals(count=1)
        alpha_model.assign_weights(method=weight_passthrouh)
        alpha_model.normalise_weights(max_weight=1.0)
        alpha_model.update_old_weights(portfolio, ignore_credit=False)
        alpha_model.calculate_target_positions(pm, investable_equity=investable)
        trades = alpha_model.generate_rebalance_trades_and_triggers(
            pm,
            min_trade_threshold=1.0,
            individual_rebalance_min_threshold=1.0,
            sell_rebalance_min_threshold=1.0,
            execution_context=input.execution_context,
        )
        flags_by_cycle[input.timestamp] = set().union(*(s.flags for s in alpha_model.raw_signals.values()))
        return trades

    state, _, _ = _run(strategy_universe, decide=decide, delay=SETTLEMENT_DELAY, delay_overrides={})

    # 3. The backtest completed with exactly two non-overlapping deposits.
    trades = list(state.portfolio.get_all_trades())
    assert all(t.is_success() for t in trades), f"Unsettled trades: {trades}"
    assert len(trades) == 2, f"Expected the initial deposit and one post-settlement top-up, got {trades}"
    assert all(t.is_buy() for t in trades)
    _assert_no_overlapping_windows(_get_pending_windows(state), END_AT)

    # 2. The pending cycle was skipped by the belt-and-braces guard, with the flag set.
    one_day = datetime.timedelta(days=1)
    assert TradingPairSignalFlags.settlement_pending in flags_by_cycle[START_AT + one_day]
    opened_at = {t.opened_at for t in trades}
    assert START_AT + one_day not in opened_at, "A second deposit was created while the first was pending"


def _make_async_trade(
    pair: TradingPairIdentifier,
    trade_id: int,
    quantity: Decimal,
    async_flow: bool = True,
    **timestamps,
) -> TradeExecution:
    """Hand-build a vault trade in an arbitrary lifecycle shape for state-level tests."""
    trade = TradeExecution(
        trade_id=trade_id,
        position_id=1,
        trade_type=TradeType.rebalance,
        pair=pair,
        opened_at=timestamps.get("opened_at", START_AT),
        planned_quantity=quantity,
        planned_price=float(FIXED_PRICE),
        planned_reserve=abs(quantity) * Decimal(FIXED_PRICE),
        reserve_currency=pair.quote,
    )
    if async_flow:
        trade.other_data["vault_async_flow"] = True
    for field, value in timestamps.items():
        setattr(trade, field, value)
    return trade


@pytest.mark.timeout(60)
def test_position_has_pending_vault_settlement_shapes():
    """Trade lifecycle shapes that must (not) count as an in-flight async vault request.

    Live routing stamps vault_async_flow at transaction-build time but the
    vault_settlement_pending status appears only after confirmation parsing,
    so a request can be live on-chain while the trade still reads
    started/broadcasted. Trading the position in that window risks a
    duplicate request - has_pending_vault_settlement() must catch it.

    1. A trade in vault_settlement_pending counts as in flight.
    2. An async-flagged trade merely started or broadcasted counts as in flight.
    3. A planned async-flagged trade does not (the alpha model itself creates those).
    4. Settled and failed async trades do not; flag-less broadcasted trades do not.
    """
    strategy_universe, vault_pair = _make_universe(rising_price=False)

    def _position_with(trade: TradeExecution) -> TradingPosition:
        position = TradingPosition(
            position_id=1,
            pair=vault_pair,
            opened_at=START_AT,
            last_pricing_at=START_AT,
            last_token_price=FIXED_PRICE,
            last_reserve_price=1.0,
            reserve_currency=vault_pair.quote,
        )
        position.trades[trade.trade_id] = trade
        return position

    one = Decimal(1)

    # 1. Confirmed request awaiting settlement.
    pending = _make_async_trade(vault_pair, 1, one, started_at=START_AT, vault_settlement_pending_at=START_AT)
    assert _position_with(pending).has_pending_vault_settlement()

    # 2. Request built/broadcast, confirmation not yet parsed - may be live on-chain.
    started = _make_async_trade(vault_pair, 2, one, started_at=START_AT)
    assert _position_with(started).has_pending_vault_settlement()
    broadcasted = _make_async_trade(vault_pair, 3, one, started_at=START_AT, broadcasted_at=START_AT)
    assert _position_with(broadcasted).has_pending_vault_settlement()

    # 3. Planned trades are created by the alpha model mid-cycle - not in flight.
    planned = _make_async_trade(vault_pair, 4, one)
    assert not _position_with(planned).has_pending_vault_settlement()

    # 4. Terminal states and non-async trades are not in flight.
    settled = _make_async_trade(vault_pair, 5, one, started_at=START_AT, executed_at=START_AT + datetime.timedelta(days=2))
    assert not _position_with(settled).has_pending_vault_settlement()
    failed = _make_async_trade(vault_pair, 6, one, started_at=START_AT, failed_at=START_AT + datetime.timedelta(days=1))
    assert not _position_with(failed).has_pending_vault_settlement()
    sync_broadcasted = _make_async_trade(vault_pair, 7, one, async_flow=False, started_at=START_AT, broadcasted_at=START_AT)
    assert not _position_with(sync_broadcasted).has_pending_vault_settlement()


@pytest.mark.timeout(300)
def test_backtest_async_vault_sell_proceeds_cannot_fund_buys():
    """Buys cannot be funded by same-cycle redemption proceeds of an override-only async vault.

    A redemption requested from an async vault pays out only after settlement,
    so a rotation that sells the async vault and buys another vault in the same
    cycle must fit its buys in the actually-free cash. The vault here carries NO
    async feature flags - it is asynchronous only via the backtest delay
    override - so the cap must classify it from the position's own settlement
    history (vault_async_flow on its deposit), not from metadata.

    1. Deposit 90% of capital into the override-only async vault VO, leaving
       ~10% free cash, and let the deposit settle.
    2. Rotate everything into the synchronous vault VS: the VO close is an
       async redeem whose proceeds arrive two days later.
    3. On the rotation cycle the VS buy is capped to the free cash
       (capped_by_pending_settlement_cash flag), instead of assuming the
       redemption proceeds are spendable - which would crash the wallet.
    4. After the redemption settles, a later rebalance redeploys the proceeds
       into VS, and the backtest completes with everything settled.
    """
    strategy_universe, pairs = _make_multi_vault_universe([
        ("VO", VAULT_A_ADDRESS, None),
        ("VS", VAULT_B_ADDRESS, None),
    ])
    assert not pairs["VO"].is_async_vault(), "VO must be async only via the settlement-delay override"

    one_day = datetime.timedelta(days=1)
    allocation = 0.9
    weights_schedule = {
        START_AT + 0 * one_day: {"VO": 1.0},
        START_AT + 1 * one_day: {"VO": 1.0},
        START_AT + 2 * one_day: {"VO": 1.0},
        START_AT + 3 * one_day: {"VS": 1.0},   # rotation: async VO close + VS buy
        START_AT + 4 * one_day: {"VS": 1.0},
        START_AT + 5 * one_day: {"VS": 1.0},   # redeem settled: redeploy proceeds
        START_AT + 6 * one_day: {"VS": 1.0},
    }
    rotation_ts = START_AT + 3 * one_day
    flags_by_cycle: dict[datetime.datetime, set] = {}

    def decide(input: StrategyInput) -> list[TradeExecution]:
        state = input.state
        weights = weights_schedule.get(input.timestamp)
        if weights is None:
            return []

        pm = input.get_position_manager()
        portfolio = state.portfolio
        pair_by_symbol = {p.base.token_symbol: p for p in input.strategy_universe.iterate_pairs()}

        alpha_model = AlphaModel(input.timestamp, close_position_weight_epsilon=0.001)
        for symbol, weight in weights.items():
            alpha_model.set_signal(pair_by_symbol[symbol], weight)

        # The xchain pattern: pin in-flight positions, deploy the rest.
        locked = alpha_model.carry_forward_non_redeemable_positions(pm)
        deployable = max(portfolio.calculate_total_equity() * allocation - locked, 0.0)

        alpha_model.select_top_signals(count=2)
        alpha_model.assign_weights(method=weight_passthrouh)
        alpha_model.normalise_weights(max_weight=1.0)
        alpha_model.update_old_weights(portfolio, ignore_credit=False)
        alpha_model.calculate_target_positions(pm, investable_equity=deployable)
        trades = alpha_model.generate_rebalance_trades_and_triggers(
            pm,
            min_trade_threshold=1.0,
            individual_rebalance_min_threshold=1.0,
            sell_rebalance_min_threshold=1.0,
            execution_context=input.execution_context,
        )
        flags_by_cycle[input.timestamp] = set().union(*(s.flags for s in alpha_model.raw_signals.values()))
        return trades

    reserve_address = strategy_universe.reserve_assets[0].address
    hook = _SnapshotHook(reserve_address, pairs["VO"].base.address)

    # 1.-2. VO is async purely via the delay override; completion proves the cap fired
    # (an uncapped VS buy would overdraw the simulated wallet by ~80% of equity).
    state, _, _ = _run(
        strategy_universe,
        hook,
        decide=decide,
        delay=SETTLEMENT_DELAY,
        delay_overrides={VAULT_A_ADDRESS: SETTLEMENT_DELAY},
    )

    trades = list(state.portfolio.get_all_trades())
    assert all(t.is_success() for t in trades), f"Unsettled trades: {trades}"

    # 3. The rotation-cycle VS buy was capped to the free cash.
    assert TradingPairSignalFlags.capped_by_pending_settlement_cash in flags_by_cycle[rotation_ts], \
        f"Cap never fired on the rotation cycle: {flags_by_cycle[rotation_ts]}"
    cash_at_rotation = next(s["cash"] for s in hook.snapshots if s["ts"] == rotation_ts)
    rotation_buys = sum(float(t.planned_reserve) for t in trades if t.is_buy() and t.opened_at == rotation_ts)
    assert rotation_buys > 0, "Rotation cycle generated no capped buy at all"
    assert rotation_buys <= cash_at_rotation + 1e-6, \
        f"Rotation buys {rotation_buys} exceed free cash {cash_at_rotation}"

    # 4. The redemption proceeds were redeployed into VS on a later cycle.
    # Compare the final position value, not the buy volume: while the VO close
    # is pending its pinned value shrinks the deployable target, which can
    # cause benign sell-and-rebuy churn on VS in between.
    vs_buys = [t for t in trades if t.pair.base.token_symbol == "VS" and t.is_buy()]
    assert len(vs_buys) >= 2, f"Redemption proceeds never redeployed: {vs_buys}"
    vs_position = next(p for p in state.portfolio.open_positions.values() if p.pair.base.token_symbol == "VS")
    assert vs_position.get_value() == pytest.approx(INITIAL_DEPOSIT * allocation, rel=0.05)


def test_backtest_async_vault_same_cycle_cash_buffer():
    """The same-cycle cash buffer withholds extra headroom from the async cap.

    When async redemption proceeds force this cycle's buys to be funded from
    current cash plus synchronous sell proceeds, the sync proceeds are only
    mark-to-market; execution realises slightly less, so scaling buys to exactly
    the free cash can leave a (cross-chain) plan a few dollars short. Passing
    ``same_cycle_cash_buffer_usd`` keeps the rotation buys below the free cash by
    the buffer amount.

    1. Deposit 90% into an override-only async vault VO, leaving ~10% free cash.
    2. Rotate everything into VS in one cycle with a 200 USD same-cycle buffer;
       the VO close is an async redeem whose proceeds arrive later.
    3. Assert the rotation-cycle VS buy is capped to free cash minus the buffer
       (not just free cash), and the buffer flag fired.
    """
    # 1. Deposit 90% into the override-only async vault VO, leaving ~10% free cash.
    strategy_universe, pairs = _make_multi_vault_universe([
        ("VO", VAULT_A_ADDRESS, None),
        ("VS", VAULT_B_ADDRESS, None),
    ])
    one_day = datetime.timedelta(days=1)
    allocation = 0.9
    buffer_usd = 200.0
    weights_schedule = {
        START_AT + 0 * one_day: {"VO": 1.0},
        START_AT + 1 * one_day: {"VO": 1.0},
        START_AT + 2 * one_day: {"VO": 1.0},
        START_AT + 3 * one_day: {"VS": 1.0},   # rotation: async VO close + VS buy
        START_AT + 4 * one_day: {"VS": 1.0},
        START_AT + 5 * one_day: {"VS": 1.0},
    }
    rotation_ts = START_AT + 3 * one_day
    flags_by_cycle: dict[datetime.datetime, set] = {}

    def decide(input: StrategyInput) -> list[TradeExecution]:
        state = input.state
        weights = weights_schedule.get(input.timestamp)
        if weights is None:
            return []

        pm = input.get_position_manager()
        portfolio = state.portfolio
        pair_by_symbol = {p.base.token_symbol: p for p in input.strategy_universe.iterate_pairs()}

        alpha_model = AlphaModel(input.timestamp, close_position_weight_epsilon=0.001)
        for symbol, weight in weights.items():
            alpha_model.set_signal(pair_by_symbol[symbol], weight)

        locked = alpha_model.carry_forward_non_redeemable_positions(pm)
        deployable = max(portfolio.calculate_total_equity() * allocation - locked, 0.0)

        alpha_model.select_top_signals(count=2)
        alpha_model.assign_weights(method=weight_passthrouh)
        alpha_model.normalise_weights(max_weight=1.0)
        alpha_model.update_old_weights(portfolio, ignore_credit=False)
        alpha_model.calculate_target_positions(pm, investable_equity=deployable)
        # 2. Rotate with a same-cycle cash buffer.
        trades = alpha_model.generate_rebalance_trades_and_triggers(
            pm,
            min_trade_threshold=1.0,
            individual_rebalance_min_threshold=1.0,
            sell_rebalance_min_threshold=1.0,
            execution_context=input.execution_context,
            same_cycle_cash_buffer_usd=buffer_usd,
        )
        flags_by_cycle[input.timestamp] = set().union(*(s.flags for s in alpha_model.raw_signals.values()))
        return trades

    reserve_address = strategy_universe.reserve_assets[0].address
    hook = _SnapshotHook(reserve_address, pairs["VO"].base.address)

    state, _, _ = _run(
        strategy_universe,
        hook,
        decide=decide,
        delay=SETTLEMENT_DELAY,
        delay_overrides={VAULT_A_ADDRESS: SETTLEMENT_DELAY},
    )

    trades = list(state.portfolio.get_all_trades())
    assert all(t.is_success() for t in trades), f"Unsettled trades: {trades}"

    # 3. The rotation-cycle VS buy is capped to free cash MINUS the buffer.
    assert TradingPairSignalFlags.capped_by_pending_settlement_cash in flags_by_cycle[rotation_ts], \
        f"Cap never fired on the rotation cycle: {flags_by_cycle[rotation_ts]}"
    cash_at_rotation = next(s["cash"] for s in hook.snapshots if s["ts"] == rotation_ts)
    rotation_buys = sum(float(t.planned_reserve) for t in trades if t.is_buy() and t.opened_at == rotation_ts)
    assert rotation_buys > 0, "Rotation cycle generated no capped buy at all"
    assert rotation_buys <= cash_at_rotation - buffer_usd + 1e-6, \
        f"Rotation buys {rotation_buys} exceed free cash {cash_at_rotation} minus buffer {buffer_usd}"


@pytest.mark.timeout(60)
def test_pending_vault_settlements_chart_metadata_matrix():
    """The pending settlements chart handles every persisted async trade shape.

    The chart reconstructs pending windows from trade metadata because
    settlement clears the live pending marker. Old state files predate the
    durable vault_settlement_requested_at key, and failed (reclaimed) requests
    end their window at failure time.

    1. Build a state with four async trade shapes on one vault position:
       still pending (with the requested-at key), settled (marker cleared),
       legacy pending (no requested-at key), and failed/reclaimed.
    2. Render pending_vault_settlements.
    3. Settled redeem: buffer present inside [requested, settled), zero at the
       settlement timestamp (half-open window).
    4. Failed deposit: buffer ends at failed_at; legacy and current pending
       deposits run to the end of the statistics series.
    """
    strategy_universe, vault_pair = _make_universe(rising_price=False)
    one_day = datetime.timedelta(days=1)

    # 1. One position holding all four shapes.
    position = TradingPosition(
        position_id=1,
        pair=vault_pair,
        opened_at=START_AT,
        last_pricing_at=START_AT,
        last_token_price=FIXED_PRICE,
        last_reserve_price=1.0,
        reserve_currency=vault_pair.quote,
    )
    # Still pending deposit, current state shape: requested day 0, never settles.
    t1 = _make_async_trade(vault_pair, 1, Decimal(10), started_at=START_AT)
    t1.other_data["vault_settlement_requested_at"] = START_AT.isoformat()
    t1.vault_settlement_pending_at = START_AT
    # Settled redeem: requested day 1, settled day 3, pending marker cleared.
    t2 = _make_async_trade(vault_pair, 2, Decimal(-10), started_at=START_AT + one_day)
    t2.other_data["vault_settlement_requested_at"] = (START_AT + one_day).isoformat()
    t2.executed_at = START_AT + 3 * one_day
    # Legacy pending deposit: old state file without the requested-at key.
    t3 = _make_async_trade(vault_pair, 3, Decimal(10), started_at=START_AT + 2 * one_day)
    t3.vault_settlement_pending_at = START_AT + 2 * one_day
    # Failed (reclaimed) deposit: requested day 1, reclaimed day 4.
    t4 = _make_async_trade(vault_pair, 4, Decimal(10), started_at=START_AT + one_day)
    t4.other_data["vault_settlement_requested_at"] = (START_AT + one_day).isoformat()
    t4.failed_at = START_AT + 4 * one_day
    for t in (t1, t2, t3, t4):
        position.trades[t.trade_id] = t

    state = State()
    state.portfolio.open_positions[1] = position
    for n in range(8):
        state.stats.portfolio.append(
            PortfolioStatistics(calculated_at=START_AT + n * one_day, total_equity=10_000.0)
        )

    # 2. Render the chart.
    chart_input = ChartInput(execution_context=unit_test_execution_context, state=state)
    _fig, df = pending_vault_settlements(chart_input)
    deposit_column = next(c for c in df.columns if c.endswith("deposit"))
    redeem_column = next(c for c in df.columns if c.endswith("redeem"))
    deposit_unit = float(Decimal(10) * Decimal(FIXED_PRICE))

    # 3. Half-open settled redeem window [day 1, day 3).
    assert df.loc[START_AT + one_day, redeem_column] == pytest.approx(-deposit_unit)
    assert df.loc[START_AT + 2 * one_day, redeem_column] == pytest.approx(-deposit_unit)
    assert df.loc[START_AT + 3 * one_day, redeem_column] == pytest.approx(0.0)

    # 4. Deposits: t1 from day 0, t3 (legacy fallback) from day 2, t4 until failed_at day 4.
    assert df.loc[START_AT, deposit_column] == pytest.approx(deposit_unit)                      # t1
    assert df.loc[START_AT + one_day, deposit_column] == pytest.approx(2 * deposit_unit)        # t1 + t4
    assert df.loc[START_AT + 3 * one_day, deposit_column] == pytest.approx(3 * deposit_unit)    # t1 + t3 + t4
    assert df.loc[START_AT + 4 * one_day, deposit_column] == pytest.approx(2 * deposit_unit)    # t4 reclaimed
    assert df.loc[START_AT + 7 * one_day, deposit_column] == pytest.approx(2 * deposit_unit)    # t1 + t3 still pending


@pytest.mark.timeout(60)
def test_alpha_model_values_broadcasted_async_requests():
    """A live broadcast-but-unconfirmed async request is valued, pinned and diagnosed correctly.

    Live routing stamps vault_async_flow at transaction-build time, but the
    vault_settlement_pending status appears only after confirmation parsing.
    A deposit caught in that window has zero shares and zero pending-status
    value - if the alpha model valued it by status alone it would pin the
    position at $0, understate equity and lose its diagnostics, while the
    request may already be live on-chain.

    1. Build a live-like state: VA holds a broadcasted deposit request
       (vault_async_flow, no pending marker, quantity 0, reserve already
       debited from cash); VB holds settled shares plus a broadcasted redeem
       request for all of them.
    2. Run the xchain AlphaModel sequence: carry-forward -> locked subtraction
       -> update_old_weights(ignore_credit=False) -> calculate_target_positions
       -> trade generation.
    3. Both positions are pinned at full committed value: locked covers the
       in-flight deposit reserve and the settled share value, old values match,
       update_old_weights does not crash on an all-pending portfolio, and no
       duplicate trades are generated.
    4. Diagnostics are direction-specific (pending_deposit_usd for VA,
       pending_redemption_usd for VB) and appear in the format_signals() table.
    5. Equity counts the in-flight deposit; VB has no shares available to
       redeem twice.
    """
    strategy_universe, pairs = _make_multi_vault_universe([
        ("VA", VAULT_A_ADDRESS, {ERC4626Feature.erc_7540_like}),
        ("VB", VAULT_B_ADDRESS, {ERC4626Feature.erc_7540_like}),
    ])
    va_pair = pairs["VA"]
    vb_pair = pairs["VB"]
    reserve_asset = strategy_universe.reserve_assets[0]
    ts = START_AT + datetime.timedelta(days=1)
    committed_usd = 5_000.0
    share_quantity = Decimal(committed_usd) / Decimal(FIXED_PRICE)

    # 1. Live-like state: all cash committed, VA deposit broadcast, VB redeem broadcast.
    state = State()
    reserve = ReservePosition(reserve_asset, Decimal(0), START_AT, 1.0, START_AT)
    state.portfolio.reserves[reserve.get_identifier()] = reserve

    va_position = TradingPosition(
        position_id=1,
        pair=va_pair,
        opened_at=START_AT,
        last_pricing_at=START_AT,
        last_token_price=FIXED_PRICE,
        last_reserve_price=1.0,
        reserve_currency=reserve_asset,
    )
    va_deposit = _make_async_trade(va_pair, 1, share_quantity, started_at=START_AT, broadcasted_at=START_AT)
    va_position.trades[va_deposit.trade_id] = va_deposit

    vb_position = TradingPosition(
        position_id=2,
        pair=vb_pair,
        opened_at=START_AT,
        last_pricing_at=START_AT,
        last_token_price=FIXED_PRICE,
        last_reserve_price=1.0,
        reserve_currency=reserve_asset,
    )
    vb_buy = _make_async_trade(
        vb_pair, 2, share_quantity,
        started_at=START_AT,
        executed_at=START_AT,
        executed_quantity=share_quantity,
        executed_price=float(FIXED_PRICE),
        executed_reserve=Decimal(committed_usd),
    )
    vb_redeem = _make_async_trade(vb_pair, 3, -share_quantity, started_at=ts, broadcasted_at=ts)
    vb_position.trades[vb_buy.trade_id] = vb_buy
    vb_position.trades[vb_redeem.trade_id] = vb_redeem

    state.portfolio.open_positions = {1: va_position, 2: vb_position}

    pricing_model = BacktestPricing(
        strategy_universe.data_universe.candles,
        generate_simple_routing_model(strategy_universe),
        allow_missing_fees=True,
    )
    pm = PositionManager(ts, strategy_universe.data_universe, state, pricing_model)

    # 2. The xchain AlphaModel sequence with no manual pending guard.
    alpha_model = AlphaModel(ts, close_position_weight_epsilon=0.001)
    alpha_model.set_signal(va_pair, 1.0)
    alpha_model.set_signal(vb_pair, 1.0)
    locked = alpha_model.carry_forward_non_redeemable_positions(pm)
    portfolio = state.portfolio
    deployable = max(portfolio.calculate_total_equity() * 0.9 - locked, 0.0)
    alpha_model.select_top_signals(count=2)
    alpha_model.assign_weights(method=weight_passthrouh)
    alpha_model.normalise_weights(max_weight=1.0)
    alpha_model.update_old_weights(portfolio, ignore_credit=False)
    alpha_model.calculate_target_positions(pm, investable_equity=deployable)
    trades = alpha_model.generate_rebalance_trades_and_triggers(
        pm,
        min_trade_threshold=1.0,
        individual_rebalance_min_threshold=1.0,
        sell_rebalance_min_threshold=1.0,
        execution_context=unit_test_execution_context,
    )

    # 3. Pinned at committed value, sane old weights, no duplicate trades.
    assert locked == pytest.approx(2 * committed_usd)
    va_signal = alpha_model.raw_signals[va_pair.internal_id]
    vb_signal = alpha_model.raw_signals[vb_pair.internal_id]
    assert va_signal.old_value == pytest.approx(committed_usd)
    assert vb_signal.old_value == pytest.approx(committed_usd)
    assert trades == [], f"Broadcasted async requests must not produce new trades: {trades}"

    # 4. Direction-specific diagnostics, visible in the diagnostics table.
    assert TradingPairSignalFlags.settlement_pending in va_signal.flags
    assert TradingPairSignalFlags.settlement_pending in vb_signal.flags
    assert va_signal.other_data["pending_deposit_usd"] == pytest.approx(committed_usd)
    assert "pending_redemption_usd" not in va_signal.other_data
    assert vb_signal.other_data["pending_redemption_usd"] == pytest.approx(committed_usd)
    assert "pending_deposit_usd" not in vb_signal.other_data
    df = format_signals(alpha_model, signal_type="all")
    va_row = df.loc[df.index.str.startswith("VA")].iloc[0]
    vb_row = df.loc[df.index.str.startswith("VB")].iloc[0]
    assert va_row["Pending deposit USD"] == pytest.approx(committed_usd)
    assert va_row["Pending redemption USD"] == "-"
    assert vb_row["Pending redemption USD"] == pytest.approx(committed_usd)
    assert vb_row["Pending deposit USD"] == "-"

    # 5. Equity counts the in-flight deposit; escrow-bound shares are not re-sellable.
    assert portfolio.get_vault_settlement_pending_value() == pytest.approx(committed_usd)
    assert portfolio.calculate_total_equity() == pytest.approx(2 * committed_usd)
    assert float(vb_position.get_available_trading_quantity()) == pytest.approx(0.0)
