"""Market limit order unit tests."""
import datetime
import random
from _decimal import Decimal

import pytest

from tradeexecutor.backtest.backtest_pricing import BacktestPricing
from tradeexecutor.backtest.backtest_routing import BacktestRoutingModel
from tradeexecutor.cli.log import setup_pytest_logging
from tradeexecutor.state.identifier import TradingPairIdentifier, AssetIdentifier
from tradeexecutor.state.reserve import ReservePosition
from tradeexecutor.state.state import State
from tradeexecutor.state.trigger import TriggerType
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.stop_loss import check_position_triggers
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, create_pair_universe_from_code
from tradeexecutor.testing.synthetic_ethereum_data import generate_random_ethereum_address
from tradeexecutor.testing.synthetic_exchange_data import generate_exchange, generate_simple_routing_model
from tradeexecutor.testing.synthetic_price_data import generate_ohlcv_candles
from tradeexecutor.testing.unit_test_trader import UnitTestTrader
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe


@pytest.fixture(scope="module")
def logger(request):
    """Setup test logger."""
    return setup_pytest_logging(request, mute_requests=False)



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
        address=generate_random_ethereum_address(),
        exchange_slug="my-dex",
    )
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
        chains={mock_chain_id},
        exchanges={mock_exchange},
        pairs=pair_universe,
        candles=candle_universe,
        liquidity=None
    )

    universe.pairs.exchange_universe = universe.exchange_universe

    return TradingStrategyUniverse(data_universe=universe, reserve_assets=[usdc])


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


def test_market_limit_executed(
    logger,
    synthetic_universe,
    pricing_model,
    state,
):
    """See that the market limit order triggers."""


    ts = datetime.datetime(2021, 6, 2)
    position_manager = PositionManager(ts, synthetic_universe.data_universe, state, pricing_model)
    portfolio = state.portfolio

    pair = position_manager.get_trading_pair((ChainId.ethereum, "my-dex", "WETH", "USDC"))

    trades = position_manager.open_spot(
        pair=pair,
        value=position_manager.get_current_cash(),
    )

    assert len(trades) == 1
    trade = trades[0]

    # Moves position from open to pending
    position_manager.set_market_limit_trigger(
        trades,
        TriggerType.market_limit,
        price=1900,
        expires_at=None,
    )

    assert len(trade.triggers) == 1
    assert len(trade.expired_triggers) == 0

    # Position moved to pending list when market limit set
    assert len(portfolio.open_positions) == 0
    assert len(portfolio.pending_positions) == 1

    # Check that position has its pending trades list updated
    position = portfolio.pending_positions[1]
    assert len(position.trades) == 0
    assert len(position.pending_trades) == 1
    assert len(position.expired_trades) == 0

    # We are not yet in trigger threshold
    assert position.last_token_price == 1818.0

    executed_trades = check_position_triggers(position_manager)
    assert len(executed_trades) == 0

    # Fast-forward time to the moment when market price
    # has exceeeded the trigger
    position_manager.timestamp = datetime.datetime(2021, 7, 4)
    assert position_manager.pricing_model.get_mid_price(position_manager.timestamp, pair) == pytest.approx(2499.642153569536)

    executed_trades = check_position_triggers(position_manager)
    assert len(executed_trades) == 1
    assert executed_trades[0] == trade

    assert trade.activated_trigger is not None
    assert trade.triggers == []
    assert trade.expired_triggers == []


def test_market_limit_expires(
    logger,
    synthetic_universe,
    pricing_model,
    state,
):
    """See that the market limit order expires."""

    ts = datetime.datetime(2021, 6, 2)
    position_manager = PositionManager(ts, synthetic_universe.data_universe, state, pricing_model)
    portfolio = state.portfolio

    pair = position_manager.get_trading_pair((ChainId.ethereum, "my-dex", "WETH", "USDC"))

    trades = position_manager.open_spot(
        pair=pair,
        value=position_manager.get_current_cash(),
    )

    assert len(trades) == 1
    trade = trades[0]

    # Moves position from open to pending
    position_manager.set_market_limit_trigger(
        trades,
        TriggerType.market_limit,
        price=5000,
        expires_at=datetime.datetime(2021, 6, 3),
    )

    # Check that position has its pending trades list updated
    position = portfolio.pending_positions[1]
    assert len(position.trades) == 0
    assert len(position.pending_trades) == 1
    assert len(position.expired_trades) == 0

    executed_trades = check_position_triggers(position_manager)
    assert len(executed_trades) == 0
    assert len(position.pending_trades) == 1  # Did not expire yet

    # Fast-forward time to the moment when the trigger has expired
    position_manager.timestamp = datetime.datetime(2021, 6, 4)
    assert position_manager.pricing_model.get_mid_price(position_manager.timestamp, pair) < 5000

    executed_trades = check_position_triggers(position_manager)
    assert len(executed_trades) == 0

    assert trade.activated_trigger is None
    assert len(trade.triggers) == 0
    assert len(trade.expired_triggers) == 1
    assert trade.expired_triggers[0].expired_at == datetime.datetime(2021, 6, 4)


def test_partial_take_profit(
    logger,
    synthetic_universe,
    pricing_model,
    state,
):
    """Partial take profit triggered."""

    ts = datetime.datetime(2021, 6, 2)
    position_manager = PositionManager(ts, synthetic_universe.data_universe, state, pricing_model)
    portfolio = state.portfolio

    pair = position_manager.get_trading_pair((ChainId.ethereum, "my-dex", "WETH", "USDC"))

    # Open a trade
    trades = position_manager.open_spot(
        pair=pair,
        value=position_manager.get_current_cash(),
    )

    position = position_manager.get_current_position_for_pair(pair)
    position_manager.set_take_profit_triggers(
        position,
        [
            (position.get_quantity() / 0.5, 1800),
            (position.get_quantity() / 0.5, 2500),
        ]
    )

    assert len(position.pending_trades) == 2
    assert len(position.expired_trades) == 0

    # Position moved to pending list when market limit set
    assert len(portfolio.open_positions) == 0
    assert len(portfolio.pending_positions) == 1

    # Check that position has its pending trades list updated
    position = portfolio.pending_positions[1]
    assert len(position.trades) == 0
    assert len(position.pending_trades) == 1
    assert len(position.expired_trades) == 0

    # Go to the firs trigger threshold
    assert position.last_token_price == 1818.0

    executed_trades = check_position_triggers(position_manager)
    assert len(executed_trades) == 0

    # Fast-forward time to the moment when market price
    # has exceeeded the trigger
    position_manager.timestamp = datetime.datetime(2021, 7, 4)
    assert position_manager.pricing_model.get_mid_price(position_manager.timestamp, pair) == pytest.approx(2499.642153569536)

    executed_trades = check_position_triggers(position_manager)
    assert len(executed_trades) == 1
    assert executed_trades[0] == trade

    assert trade.activated_trigger is not None
    assert trade.triggers == []
    assert trade.expired_triggers == []