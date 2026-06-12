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

import pytest

from eth_defi.erc_4626.core import ERC4626Feature

from tradeexecutor.backtest.backtest_runner import run_backtest_inline
from tradeexecutor.cli.loop import ExecutionTestHook
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier, TradingPairKind
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution, TradeStatus
from tradeexecutor.strategy.alpha_model import AlphaModel
from tradeexecutor.strategy.asset import get_asset_amounts
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.execution_context import ExecutionMode
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet
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


def _make_multi_vault_universe(vault_specs: list[tuple[str, str, set | None]]) -> tuple[TradingStrategyUniverse, dict[str, TradingPairIdentifier]]:
    """Build a flat-price synthetic universe with several vault pairs.

    :param vault_specs:
        List of (share token symbol, vault address, vault features set or None).
        Passing None features creates a vault detectable as async only via a
        settlement-delay override.
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
    4. Assert the pending-window accounting: cash debited, value counted as pending,
       simulated wallet shares/reserve only move at claim, equity continuous.
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

    # 4. Pending-window accounting. A pending deposit: cash already debited, value
    #    counted as pending, but the simulated wallet has not moved yet. We deposit
    #    half the initial cash, keeping the other half as a buffer.
    deposit_amount = INITIAL_DEPOSIT * 0.5
    deposit_pending = [
        s for s in hook.snapshots
        if s["pending_value"] == pytest.approx(deposit_amount, abs=1e-6)
    ]
    assert deposit_pending, "Expected at least one cycle with the deposit pending settlement"
    for s in deposit_pending:
        assert s["cash"] == pytest.approx(deposit_amount, abs=1e-6)  # cash ledger debited at request
        assert s["wallet_shares"] == pytest.approx(0.0, abs=1e-9)    # shares not credited until claim
        assert s["wallet_reserve"] == pytest.approx(INITIAL_DEPOSIT, abs=1e-6)  # wallet reserve not debited yet
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
        assert s["wallet_shares"] == pytest.approx(50.0, abs=1e-9)

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

    1. The global default delay is non-zero (one day).
    2. An ERC-7540 vault with no override settles one day after the request.
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

    # 1. The global default delay is non-zero (one day).
    assert DEFAULT_VAULT_SETTLEMENT_DELAY == datetime.timedelta(days=1)
    assert execution.vault_settlement_delay == DEFAULT_VAULT_SETTLEMENT_DELAY

    ts = datetime.datetime(2024, 1, 1, 9, 30)

    # 2. ERC-7540 vault: one day after the request.
    assert execution._get_settlement_due(erc_7540_pair, ts) == datetime.datetime(2024, 1, 2, 9, 30)

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
