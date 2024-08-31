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
from tradeexecutor.state.trade import TradeStatus
from tradeexecutor.state.trigger import TriggerType, TriggerCondition
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.parameters import StrategyParameters
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
        internal_exchange_id=mock_exchange.exchange_id,
        fee=0.0030,
    )

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
    """Partial take profit triggered.

    - Open a spot position
    - Set two partial take profits on it
    - Last partial take profit will close the position
    - See that take profits are orderly executed
    """

    ts = datetime.datetime(2021, 6, 2)
    position_manager = PositionManager(ts, synthetic_universe.data_universe, state, pricing_model)
    portfolio = state.portfolio

    pair = position_manager.get_trading_pair((ChainId.ethereum, "my-dex", "WETH", "USDC"))

    # Start to prepare a new position
    opening_trades = position_manager.open_spot(
        pair=pair,
        value=position_manager.get_current_cash(),
    )
    assert len(opening_trades) == 1

    # Set the two stage take profit for the position in preparation
    position = position_manager.get_current_position_for_pair(pair)
    total_quantity = position.get_quantity(planned=True)
    half_quantity = total_quantity / Decimal(2)
    prepared_trades = position_manager.prepare_take_profit_trades(
        position,
        [
            (1900.0, -half_quantity, False),
            (2500.0, -half_quantity, True),
        ]
    )

    # Check that the bunch of data structures are initialised correctly
    assert len(prepared_trades) == 2
    assert len(position.pending_trades) == 2
    assert len(position.expired_trades) == 0
    assert len(portfolio.open_positions) == 1
    assert len(portfolio.pending_positions) == 0
    t = prepared_trades[0]
    assert t.is_sell()  # Spot take profit is sell
    assert t.get_status() == TradeStatus.planned
    assert t.planned_quantity == pytest.approx(-half_quantity)
    assert len(t.triggers) == 1
    trigger = t.triggers[0]
    assert trigger.price == pytest.approx(1900.0)
    assert trigger.type == TriggerType.take_profit_partial
    assert trigger.condition == TriggerCondition.cross_above
    t = prepared_trades[1]
    trigger = t.triggers[0]
    assert trigger.price == pytest.approx(2500.0)

    # Execute the position open
    trader = UnitTestTrader(state)
    trader.set_perfectly_executed(opening_trades[0])
    assert opening_trades[0].is_success()
    assert position.get_quantity() == pytest.approx(total_quantity)

    # We are not within take profit level yet
    assert position.last_token_price == pytest.approx(1823.4539999999997)

    executed_trades = check_position_triggers(position_manager)
    assert len(executed_trades) == 0

    # Fast-forward time to the moment when market price
    # has exceeeded the first trigger
    position_manager.timestamp = datetime.datetime(2021, 6, 8)
    assert position_manager.pricing_model.get_mid_price(position_manager.timestamp, pair) == pytest.approx(1929.8436337926182)

    # Check that the first trade triggers, and execute it
    triggered_trades = check_position_triggers(position_manager)
    assert len(triggered_trades) == 1
    t = triggered_trades[0]
    assert t.planned_quantity == pytest.approx(-half_quantity)
    assert position.get_quantity() == pytest.approx(total_quantity)
    trader.set_perfectly_executed(t)
    assert t.is_sell()
    assert t.is_success()
    assert t.executed_quantity == pytest.approx(-half_quantity)
    assert position.is_open()

    # The position quantity should have decreased to half,
    # and we have only one pending trade left
    assert position.get_quantity() == pytest.approx(half_quantity)
    assert len(position.pending_trades) == 1

    # Trigger the remaining take profit and close the position
    position_manager.timestamp = datetime.datetime(2021, 8, 1)
    assert position_manager.pricing_model.get_mid_price(position_manager.timestamp, pair) == pytest.approx(3302.7545979895194)
    triggered_trades = check_position_triggers(position_manager)
    assert len(triggered_trades) == 1
    t = triggered_trades[0]
    assert t.planned_quantity == pytest.approx(-half_quantity)
    assert position.get_quantity() == pytest.approx(half_quantity)
    trader.set_perfectly_executed(t)
    assert t.is_sell()
    assert t.is_success()
    assert t.executed_quantity == pytest.approx(-half_quantity)
    assert position.is_closed()
