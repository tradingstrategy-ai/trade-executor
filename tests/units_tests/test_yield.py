"""Test yield manager calculations."""
import datetime
import random
from decimal import Decimal

import pytest

from tradeexecutor.backtest.backtest_pricing import BacktestPricing
from tradeexecutor.backtest.backtest_routing import BacktestRoutingModel
from tradeexecutor.backtest.backtest_sync import BacktestSyncModel
from tradeexecutor.backtest.simulated_wallet import SimulatedWallet
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.trading_strategy_universe import create_pair_universe_from_code, TradingStrategyUniverse, translate_trading_pair
from tradeexecutor.testing.synthetic_ethereum_data import generate_random_ethereum_address
from tradeexecutor.testing.synthetic_exchange_data import generate_exchange, generate_simple_routing_model
from tradeexecutor.testing.synthetic_lending_data import generate_lending_universe, generate_lending_reserve
from tradeexecutor.testing.synthetic_price_data import generate_ohlcv_candles
from tradingstrategy.alternative_data.vault import load_multiple_vaults
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.pair import DEXPair
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe


@pytest.fixture()
def usdc() -> AssetIdentifier:
    """Create a mock USDC asset."""
    usdc = AssetIdentifier(ChainId.ethereum.value, generate_random_ethereum_address(), "USDC", 6, 1)
    return usdc


@pytest.fixture()
def weth() -> AssetIdentifier:
    weth = AssetIdentifier(ChainId.ethereum.value, generate_random_ethereum_address(), "WETH", 18, 2)
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
    vault_dex_pair = DEXPair.create_from_row(vault_pairs_df.iloc[0])
    ipor_pair = translate_trading_pair(vault_dex_pair)

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
    return generate_simple_routing_model(synthetic_universe)


@pytest.fixture()
def pricing_model(synthetic_universe, routing_model) -> BacktestPricing:
    pricing_model = BacktestPricing(
        synthetic_universe.data_universe.candles,
        routing_model,
        allow_missing_fees=True,
    )
    return pricing_model


@pytest.fixture()
def sync_model(usdc) -> BacktestSyncModel:
    """Read wallet balances back to the backtesting state."""
    wallet = SimulatedWallet()
    wallet.set_balance(usdc, Decimal(10_000))
    sync_model = BacktestSyncModel(wallet)
    return sync_model


@pytest.fixture()
def state(synthetic_universe, usdc) -> State:
    """Create empty state."""
    state = State()
    state.portfolio.initialise_reserves(usdc, reserve_token_price=1.0)
    return state


def test_yield_manager_setup(
    synthetic_universe: TradingStrategyUniverse,
    state: State,
    pricing_model,
    routing_model,
):
    """We can setup yield manager."""

    weth_usdc = synthetic_universe.get_pair_by_human_description(
        (ChainId.ethereum, "my-dex", "WETH", "USDC"),
    )
    assert weth_usdc.is_spot()

    ipor_usdc = synthetic_universe.get_pair_by_smart_contract(
        "0x45aa96f0b3188d47a1dafdbefce1db6b37f58216",
    )
    assert ipor_usdc.is_vault()

    state = State()
    start_at = datetime.datetime(2021, 6, 1)
    position_manager = PositionManager(
        timestamp=start_at,
        universe=synthetic_universe,
        pricing_model=pricing_model,
        routing_model=routing_model,
    )




