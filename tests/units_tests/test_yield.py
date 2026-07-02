"""Test yield manager calculations."""
import datetime
import random
from decimal import Decimal

import numpy as np
import pytest

from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.strategy.execution_context import ExecutionMode
from tradingstrategy.alternative_data.vault import load_multiple_vaults
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.pair import DEXPair
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe


from tradeexecutor.backtest.backtest_pricing import BacktestPricing
from tradeexecutor.backtest.backtest_routing import BacktestRoutingModel
from tradeexecutor.backtest.backtest_sync import BacktestSyncModel
from tradeexecutor.backtest.simulated_wallet import SimulatedWallet
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.trading_strategy_universe import create_pair_universe_from_code, TradingStrategyUniverse, translate_trading_pair
from tradeexecutor.testing.synthetic_ethereum_data import generate_random_ethereum_address
from tradeexecutor.testing.synthetic_exchange_data import generate_exchange, generate_vault_routing_model
from tradeexecutor.testing.synthetic_lending_data import generate_lending_universe, generate_lending_reserve
from tradeexecutor.testing.synthetic_price_data import generate_ohlcv_candles
from tradeexecutor.strategy.pandas_trader.yield_manager import YieldManager, YieldRuleset, YieldWeightingRule, YieldDecisionInput


@pytest.fixture()
def usdc() -> AssetIdentifier:
    """Create a mock USDC asset."""
    usdc = AssetIdentifier(ChainId.base.value, generate_random_ethereum_address(), "USDC", 6, 1)
    return usdc


@pytest.fixture()
def weth() -> AssetIdentifier:
    weth = AssetIdentifier(ChainId.base.value, generate_random_ethereum_address(), "WETH", 18, 2)
    return weth


@pytest.fixture()
def synthetic_universe(usdc, weth) -> TradingStrategyUniverse:
    """Create a universe with one directional pair, Aave, Ipor.

    - WETH-USDC, close price increase 1% every day
    """

    time_bucket = TimeBucket.d1
    start_at = datetime.datetime(2021, 6, 1)
    end_at = datetime.datetime(2022, 1, 1)

    # Set up fake assets
    chain_id = ChainId.base
    mock_exchange = generate_exchange(
        exchange_id=random.randint(1, 1000),
        chain_id=chain_id,
        address=generate_random_ethereum_address(),
        exchange_slug="my-dex",
    )

    weth_usdc = TradingPairIdentifier(
        weth,
        usdc,
        generate_random_ethereum_address(),
        mock_exchange.address,
        internal_id=random.randint(1, 1000),
        internal_exchange_id=mock_exchange.exchange_id,
        fee=0.0030,
    )

    usdc_reserve = generate_lending_reserve(usdc, chain_id, 1)
    weth_reserve = generate_lending_reserve(weth, chain_id, 2)

    # Create Aave pair
    _, lending_candle_universe = generate_lending_universe(
        time_bucket,
        start_at,
        end_at,
        reserves=[usdc_reserve, weth_reserve],
        aprs={
            "supply": 2,
            "variable": 5,
        }
    )

    # Create a vault pair.
    vaults = [(chain_id, "0x45aa96f0b3188d47a1dafdbefce1db6b37f58216")]
    vault_exchanges, vault_pairs_df = load_multiple_vaults(vaults)
    vault_exchange = next(iter(vault_exchanges))
    assert type(vault_exchange.exchange_id) == int
    vault_dex_pair = DEXPair.create_from_row(vault_pairs_df.iloc[0])
    assert type(vault_dex_pair.exchange_id) == np.int64
    ipor_pair = translate_trading_pair(vault_dex_pair)
    assert ipor_pair.is_vault()

    pair_universe = create_pair_universe_from_code(
        chain_id,
        pairs=[weth_usdc, ipor_pair]
    )

    candles = generate_ohlcv_candles(
        time_bucket,
        start_at,
        end_at,
        pair_id=weth_usdc.internal_id,
        daily_drift=(1.01, 1.01),  # Close price increase 1% every day
        high_drift=1.05,
        low_drift=0.90,
        random_seed=1,
    )
    candle_universe = GroupedCandleUniverse.create_from_single_pair_dataframe(candles)

    universe = Universe(
        time_bucket=time_bucket,
        chains={chain_id},
        exchanges={mock_exchange, vault_exchange},
        pairs=pair_universe,
        candles=candle_universe,
        liquidity=None,
        lending_candles=lending_candle_universe,
    )

    universe.pairs.exchange_universe = universe.exchange_universe
    strategy_universe = TradingStrategyUniverse(data_universe=universe, reserve_assets=[usdc])
    return strategy_universe



@pytest.fixture()
def routing_model(synthetic_universe) -> BacktestRoutingModel:
    return generate_vault_routing_model(synthetic_universe, expected_exchages=2)


@pytest.fixture()
def pricing_model(synthetic_universe, routing_model) -> BacktestPricing:

    # Work around lack of real data
    ipor_usdc = synthetic_universe.get_pair_by_smart_contract(
        "0x45aa96f0b3188d47a1dafdbefce1db6b37f58216",
    )
    fixed_prices = {
        ipor_usdc: 2.0,
    }

    pricing_model = BacktestPricing(
        synthetic_universe.data_universe.candles,
        routing_model,
        allow_missing_fees=True,
        fixed_prices=fixed_prices,
    )
    return pricing_model


@pytest.fixture()
def sync_model(usdc) -> BacktestSyncModel:
    """Read wallet balances back to the backtesting state."""
    wallet = SimulatedWallet()
    #wallet.set_balance(usdc, Decimal(10_000))
    sync_model = BacktestSyncModel(wallet, initial_deposit_amount=Decimal(10_000))
    return sync_model


@pytest.fixture()
def state(synthetic_universe, usdc, sync_model) -> State:
    """Create empty state with $10,000 USDC reserve."""
    state = State()
    # state.portfolio.initialise_reserves(usdc, reserve_token_price=1.0)

    # Inject cash for testing
    sync_model.sync_initial(state)
    start_at = datetime.datetime(2021, 6, 1)
    sync_model.sync_treasury(
        strategy_cycle_ts=start_at,
        state=state,
        supported_reserves=[usdc],
    )

    return state


@pytest.fixture()
def rules(
    synthetic_universe,
    usdc,
) -> YieldRuleset:
    """Create yield allocation rules.

    - Allocate safe portion to IPOR
    - Allocate the remaining to Aave
    """

    weth_usdc = synthetic_universe.get_pair_by_human_description(
        (ChainId.base, "my-dex", "WETH", "USDC"),
    )
    assert weth_usdc.is_spot()

    ipor_usdc = synthetic_universe.get_pair_by_smart_contract(
        "0x45aa96f0b3188d47a1dafdbefce1db6b37f58216",
    )
    assert ipor_usdc.is_vault(), f"Got type: {ipor_usdc.kind}"

    aave_usdc = synthetic_universe.get_credit_supply_pair()
    assert aave_usdc.is_credit_supply()

    return YieldRuleset(
        position_allocation=0.95,
        buffer_pct=0.01,
        cash_change_tolerance_usd=5.00,
        weights=[
            YieldWeightingRule(pair=ipor_usdc, max_concentration=0.33),
            YieldWeightingRule(pair=aave_usdc, max_concentration=1.0),
        ]
    )


def test_yield_manager_setup(
    synthetic_universe: TradingStrategyUniverse,
    state: State,
    pricing_model,
    routing_model,
    rules: YieldRuleset,
):
    """We can setup yield manager."""

    start_at = datetime.datetime(2021, 6, 1)
    position_manager = PositionManager(
        timestamp=start_at,
        universe=synthetic_universe,
        pricing_model=pricing_model,
        routing_model=routing_model,
        state=state,
    )

    yield_manager = YieldManager(
        position_manager=position_manager,
        rules=rules,
    )
    assert isinstance(yield_manager.portfolio, Portfolio)
    assert yield_manager.cash_pair.is_cash()



def test_yield_distribute_all(
    synthetic_universe: TradingStrategyUniverse,
    state: State,
    pricing_model,
    routing_model,
    rules: YieldRuleset,
):
    """Distribute all cash to yield positions.

    - No directional trades taken
    - We get one trade to open Aave position, another to open IPOR position
    """

    assert state.portfolio.get_total_equity() == 10_000.0

    start_at = datetime.datetime(2021, 6, 1)
    position_manager = PositionManager(
        timestamp=start_at,
        universe=synthetic_universe,
        pricing_model=pricing_model,
        routing_model=routing_model,
        state=state,
    )

    yield_manager = YieldManager(
        position_manager=position_manager,
        rules=rules,
    )

    input = YieldDecisionInput(
        execution_mode=ExecutionMode.backtesting,
        cycle=1,
        timestamp=start_at,
        total_equity=state.portfolio.get_total_equity(),
        directional_trades=[],
        pending_redemptions=0,

    )
    result = yield_manager.calculate_yield_management(input)
    assert len(result.trades) == 2, f"Got trades: {result.trades}"

    t = result.trades[0]
    assert t.is_vault()
    assert t.is_buy()
    assert t.planned_price == 2.0  # Fixed price
    assert t.planned_reserve == pytest.approx(Decimal(3135))

    t = result.trades[1]
    assert t.is_credit_supply()
    assert t.is_buy()
    assert t.planned_reserve == pytest.approx(Decimal(6365))


def test_yield_distribute_some_directional(
    synthetic_universe: TradingStrategyUniverse,
    state: State,
    pricing_model,
    routing_model,
    rules: YieldRuleset,
):
    """Distribute leftover cash to yield positions.

    - One directional trades taken
    - We get one trade to open Aave position, another to open IPOR position
    """

    assert state.portfolio.get_total_equity() == 10_000.0

    start_at = datetime.datetime(2021, 6, 1)
    position_manager = PositionManager(
        timestamp=start_at,
        universe=synthetic_universe,
        pricing_model=pricing_model,
        routing_model=routing_model,
        state=state,
    )

    yield_manager = YieldManager(
        position_manager=position_manager,
        rules=rules,
    )

    weth_usdc = synthetic_universe.get_pair_by_human_description(
        (ChainId.base, "my-dex", "WETH", "USDC"),
    )

    directional_trades = position_manager.open_spot(
        weth_usdc,
        value=1000.0,
    )

    input = YieldDecisionInput(
        total_equity=state.portfolio.get_total_equity(),
        execution_mode=ExecutionMode.backtesting,
        cycle=1,
        timestamp=start_at,
        directional_trades=directional_trades,
        pending_redemptions=0,
    )

    result = yield_manager.calculate_yield_management(input)
    assert len(result.trades) == 2, f"Got trades: {result.trades}"

    t = result.trades[0]
    assert t.is_vault()
    assert t.is_buy()
    assert t.planned_price == 2.0  # Fixed price
    assert t.planned_reserve == pytest.approx(Decimal(2805))

    t = result.trades[1]
    assert t.is_credit_supply()
    assert t.is_buy()
    assert t.planned_reserve == pytest.approx(Decimal(5695))


def test_yield_release_when_fully_deployed(
    synthetic_universe: TradingStrategyUniverse,
    state: State,
    pricing_model,
    routing_model,
    rules: YieldRuleset,
):
    """The zero-release-safe wrapper survives a fully-deployed cycle that trips the strict assert.

    ``calculate_yield_management`` asserts ``available_for_yield > 0``; when a directional buy plus
    the ``always_in_cash`` reserve consume all cash-like value, it goes negative. That is the
    phase-aware full-deployment / full-promotion boundary (invariant 5), so the safe wrapper must
    absorb it rather than crash the strategy.

    1. Build a YieldManager and a directional buy consuming ~98% of the cash.
    2. Assert the strict ``calculate_yield_management`` raises at that boundary.
    3. Assert ``calculate_yield_management_safe`` instead returns a YieldResult with
       ``available_for_yield <= 0`` and no sweep trades (no venue position exists to release yet).
    """
    # 1. Build a YieldManager and a directional buy consuming ~98% of the cash.
    assert state.portfolio.get_total_equity() == 10_000.0
    start_at = datetime.datetime(2021, 6, 1)
    position_manager = PositionManager(
        timestamp=start_at,
        universe=synthetic_universe,
        pricing_model=pricing_model,
        routing_model=routing_model,
        state=state,
    )
    yield_manager = YieldManager(position_manager=position_manager, rules=rules)
    weth_usdc = synthetic_universe.get_pair_by_human_description(
        (ChainId.base, "my-dex", "WETH", "USDC"),
    )
    directional_trades = position_manager.open_spot(weth_usdc, value=9_800.0)
    input = YieldDecisionInput(
        total_equity=state.portfolio.get_total_equity(),
        execution_mode=ExecutionMode.backtesting,
        cycle=1,
        timestamp=start_at,
        directional_trades=directional_trades,
        pending_redemptions=0,
    )

    # 2. The strict path trips its `available_for_yield > 0` assert at the full-deployment boundary.
    with pytest.raises(AssertionError):
        yield_manager.calculate_yield_management(input)

    # 3. The safe wrapper handles it: no sweep, and no release trades (no venue position exists yet).
    result = yield_manager.calculate_yield_management_safe(input)
    assert result.available_for_yield <= 0
    assert result.trades == []


def test_yield_safe_sweeps_reserve_withholding_pending_redemptions(
    synthetic_universe: TradingStrategyUniverse,
    state: State,
    pricing_model: BacktestPricing,
    routing_model: BacktestRoutingModel,
    rules: YieldRuleset,
):
    """The safe wrapper sweeps idle reserve into the venue while withholding pending redemptions.

    calculate_yield_management_safe's sweep path (available_for_yield > 0, delegating to the strict
    method) routes idle reserve above always_in_cash into the queue venue, so freed cash stays
    productive - settled redemption proceeds are indistinguishable reserve cash to YieldManager, so
    the same sweep covers them. The strategy's own pending redemptions are withheld from that sweep
    (available_for_yield subtracts them) so they can be paid out. This covers both the safe wrapper's
    sweep branch and the one redemption-specific term in the calc; it complements
    test_yield_release_when_fully_deployed (the release path) and test_yield_distribute_all (the
    no-pending strict path).

    1. Reserve holds $10,000, no directional trades, $2,000 of pending redemptions.
    2. available_for_yield = equity - always_in_cash - pending = $10,000 - $500 - $2,000 = $7,500.
    3. Exactly that $7,500 is swept into venue buys; the $2,000 is withheld for the redemption queue
       rather than idle-swept.
    """
    # 1. Idle reserve with a pending redemption obligation; nothing directional this cycle.
    assert state.portfolio.get_total_equity() == 10_000.0
    start_at = datetime.datetime(2021, 6, 1)
    position_manager = PositionManager(
        timestamp=start_at,
        universe=synthetic_universe,
        pricing_model=pricing_model,
        routing_model=routing_model,
        state=state,
    )
    yield_manager = YieldManager(position_manager=position_manager, rules=rules)
    input = YieldDecisionInput(
        execution_mode=ExecutionMode.backtesting,
        cycle=1,
        timestamp=start_at,
        total_equity=state.portfolio.get_total_equity(),
        directional_trades=[],
        pending_redemptions=2_000.0,
    )

    # 2. The safe wrapper takes the sweep path and subtracts the pending redemptions from the budget.
    result = yield_manager.calculate_yield_management_safe(input)
    always_in_cash = 10_000.0 * (1 - rules.position_allocation)  # 5% reserve = $500
    assert result.available_for_yield == pytest.approx(10_000.0 - always_in_cash - 2_000.0)  # 7,500

    # 3. Exactly the non-withheld excess is swept into venue buys; the redemption cash is not.
    assert result.trades, "Expected the swept reserve to open venue positions"
    assert all(t.is_buy() for t in result.trades)
    swept = sum(float(t.planned_reserve) for t in result.trades)
    assert swept == pytest.approx(result.available_for_yield, rel=0.01)
