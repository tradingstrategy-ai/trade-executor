"""Position manager API tests.

These functions are described are in PositionManager docstring documentation.
"""
import datetime
import random
from decimal import Decimal

import pytest

from tradeexecutor.backtest.backtest_pricing import BacktestPricing
from tradeexecutor.backtest.backtest_routing import BacktestRoutingModel
from tradeexecutor.state.identifier import TradingPairIdentifier, AssetIdentifier
from tradeexecutor.state.reserve import ReservePosition
from tradeexecutor.state.state import State
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager, NoSingleOpenPositionException
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, create_pair_universe_from_code
from tradeexecutor.testing.pairuniversetrader import PairUniverseTestTrader
from tradeexecutor.testing.simulated_trader import SimulatedTestTrader
from tradeexecutor.testing.synthetic_ethereum_data import generate_random_ethereum_address
from tradeexecutor.testing.synthetic_exchange_data import generate_exchange, generate_simple_routing_model
from tradeexecutor.testing.synthetic_price_data import generate_ohlcv_candles
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe


@pytest.fixture(scope="module")
def synthetic_universe() -> TradingStrategyUniverse:
    """Create ETH-USDC universe with random price data."""

    start_at = datetime.datetime(2021, 6, 1)
    end_at = datetime.datetime(2022, 1, 1)

    # Set up fake assets
    mock_chain_id = ChainId.ethereum
    mock_exchange = generate_exchange(
        exchange_id=random.randint(1, 1000),
        chain_id=mock_chain_id,
        address=generate_random_ethereum_address())
    usdc = AssetIdentifier(ChainId.ethereum.value, generate_random_ethereum_address(), "USDC", 6, 1)
    weth = AssetIdentifier(ChainId.ethereum.value, generate_random_ethereum_address(), "WETH", 18, 2)
    weth_usdc = TradingPairIdentifier(
        weth,
        usdc,
        generate_random_ethereum_address(),
        mock_exchange.address,
        internal_id=random.randint(1, 1000),
        internal_exchange_id=mock_exchange.exchange_id)

    time_bucket = TimeBucket.d1

    pair_universe = create_pair_universe_from_code(mock_chain_id, [weth_usdc])

    candles = generate_ohlcv_candles(time_bucket, start_at, end_at, pair_id=weth_usdc.internal_id)
    candle_universe = GroupedCandleUniverse.create_from_single_pair_dataframe(candles)

    universe = Universe(
        time_bucket=time_bucket,
        chains={mock_chain_id},
        exchanges={mock_exchange},
        pairs=pair_universe,
        candles=candle_universe,
        liquidity=None
    )

    return TradingStrategyUniverse(data_universe=universe, reserve_assets=[usdc])

@pytest.fixture()
def routing_model(synthetic_universe) -> BacktestRoutingModel:
    return generate_simple_routing_model(synthetic_universe)


@pytest.fixture()
def pricing_model(synthetic_universe, routing_model) -> BacktestRoutingModel:
    pricing_model = BacktestPricing(
        synthetic_universe.data_universe.candles,
        routing_model,
        allow_missing_fees=True,
    )
    return pricing_model


@pytest.fixture
def state(synthetic_universe: TradingStrategyUniverse):
    usdc = synthetic_universe.reserve_assets[0]
    state = State()
    ts = datetime.datetime(2022, 1, 1, tzinfo=None)
    state.update_reserves([ReservePosition(usdc, Decimal(1000), ts, 1.0, ts)])
    return state


@pytest.fixture
def position_manager(state, synthetic_universe, pricing_model):
    # Assume position manager is played on the second day
    ts = datetime.datetime(2021, 6, 2)
    p = PositionManager(ts, synthetic_universe.data_universe, state, pricing_model)
    return p


@pytest.fixture
def position_manager_with_open_position(
        synthetic_universe: TradingStrategyUniverse,
        position_manager,
        pricing_model,
):
    """Force in one executed position."""

    ts = datetime.datetime(2021, 6, 12)

    eth_usdc = synthetic_universe.get_single_pair()
    state = position_manager.state

    trader = SimulatedTestTrader(state, pricing_model)
    trade = trader.buy(ts, eth_usdc, Decimal(50))  # 50 USDC worth of ETH
    trader.simulate_execution(state, trade)

    assert len(state.portfolio.open_positions) == 1

    return position_manager


def test_get_current_position_none(position_manager):
    """"get_current_position() raises is no position is open."""
    with pytest.raises(NoSingleOpenPositionException):
        position_manager.get_current_position()


def test_get_current_position(position_manager_with_open_position: PositionManager):
    """"Check the latest price of the position."""
    pm = position_manager_with_open_position
    current_position = pm.get_current_position()

    # Quantity is the open amount in tokens.
    # This is expressed in Python Decimal class,
    # because Ethereum token balances are accurate up to 18 decimals
    # and this kind of accuracy cannot be expressed in floating point numbers.
    quantity = current_position.get_quantity()
    assert quantity == Decimal('0.03033456189136483055782720622')

    # The current price is the price of the trading pair
    # that was recorded on the last price feed sync.
    # This is a 64-bit floating point, as the current price
    # is always approximation based on market conditions.
    price = current_position.get_current_price()
    assert price == 1648.28488966024

    # The opening price is the price of the first trade
    # that was made for this position. This is the actual
    # executed price of the trade, expressed as floating
    # point for the convenience.
    price = current_position.get_opening_price()
    assert price == 1648.28488966024

    # Test setting the stop loss
    pm.update_stop_loss(current_position, 1600)
    assert current_position.stop_loss == 1600
    assert current_position.trigger_updates[0].timestamp == datetime.datetime(2021,6,2,0,0)
    assert current_position.trigger_updates[0].stop_loss_before is None
    assert current_position.trigger_updates[0].stop_loss_after == 1600
    assert current_position.trigger_updates[0].take_profit_before is None
    assert current_position.trigger_updates[0].take_profit_after is None
    assert len(current_position.trigger_updates) == 1


def test_estimate_fee_not_available(synthetic_universe, position_manager):
    """"Estimate the trading fee."""
    eth_usdc = synthetic_universe.get_single_pair()
    fee = position_manager.get_pair_fee(eth_usdc)
    assert fee is None


def test_estimate_fee_from_router(state, synthetic_universe, position_manager):
    """"Estimate the trading fee based on router data."""

    routing_model = generate_simple_routing_model(synthetic_universe, trading_fee=0.0025)
    pricing_model = BacktestPricing(
        synthetic_universe.data_universe.candles,
        routing_model
    )
    eth_usdc = synthetic_universe.get_single_pair()
    ts = datetime.datetime(2021, 6, 2)
    position_manager = PositionManager(ts, synthetic_universe.data_universe, state, pricing_model)
    fee = position_manager.get_pair_fee(eth_usdc)
    assert fee == 0.0025


def test_estimate_fee_from_pair(synthetic_universe, position_manager):
    """"Estimate the trading fee."""
    eth_usdc = synthetic_universe.get_single_pair()
    eth_usdc.fee = 0.020
    fee = position_manager.get_pair_fee(eth_usdc)
    assert fee == 0.020