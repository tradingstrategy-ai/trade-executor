"""Test yield manager calculations."""
import datetime
import random
from decimal import Decimal

import numpy as np
import pytest

from tradeexecutor.state.portfolio import Portfolio
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
            YieldWeightingRule(pair=ipor_usdc, max_weight=0.33),
            YieldWeightingRule(pair=aave_usdc, max_weight=1.0),
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
        total_equity=state.portfolio.get_total_equity(),
        directional_trades=[],

    )
    trades = yield_manager.calculate_yield_management(input)
    assert len(trades) == 2, f"Got trades: {trades}"

    t = trades[0]
    assert t.is_vault()
    assert t.is_buy()
    assert t.planned_price == 2.0  # Fixed price
    assert t.planned_reserve == pytest.approx(Decimal(3135))

    t = trades[1]
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
        directional_trades=directional_trades,

    )
    trades = yield_manager.calculate_yield_management(input)
    assert len(trades) == 2, f"Got trades: {trades}"

    t = trades[0]
    assert t.is_vault()
    assert t.is_buy()
    assert t.planned_price == 2.0  # Fixed price
    assert t.planned_reserve == pytest.approx(Decimal(2805))

    t = trades[1]
    assert t.is_credit_supply()
    assert t.is_buy()
    assert t.planned_reserve == pytest.approx(Decimal(5695))
