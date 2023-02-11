"""Test portfolio rebalancing.

- Tests low level functions

- Tests rebalancing trades using synthetic fixed price trading universe of
  WETH/USD and AAVE/USD

"""
import os
import datetime
import random
from decimal import Decimal

import pytest
from hexbytes import HexBytes

from tradeexecutor.backtest.backtest_pricing import BacktestSimplePricingModel
from tradeexecutor.backtest.backtest_routing import BacktestRoutingModel
from tradeexecutor.state.state import State, TradeType
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.blockhain_transaction import BlockchainTransaction
from tradeexecutor.state.reserve import ReservePosition
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.pandas_trader.rebalance import get_existing_portfolio_weights, rebalance_portfolio, \
    get_weight_diffs, clip_to_normalised, BadWeightsException
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, create_pair_universe_from_code
from tradeexecutor.testing.synthetic_ethereum_data import generate_random_ethereum_address
from tradeexecutor.testing.synthetic_exchange_data import generate_exchange, generate_simple_routing_model
from tradeexecutor.testing.synthetic_price_data import generate_fixed_price_candles
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import Exchange
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe


# https://docs.pytest.org/en/latest/how-to/skipping.html#skip-all-test-functions-of-a-class-or-module
pytestmark = pytest.mark.skipif(os.environ.get("TRADING_STRATEGY_API_KEY") is None, reason="Set TRADING_STRATEGY_API_KEY environment variable to run this test")


@pytest.fixture
def mock_exchange() -> Exchange:
    """Mock some assets"""
    mock_exchange = generate_exchange(
        exchange_id=random.randint(1, 1000),
        chain_id=ChainId.ethereum,
        address=generate_random_ethereum_address())
    return mock_exchange


@pytest.fixture
def usdc() -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(ChainId.ethereum.value, "0x0", "USDC", 6)


@pytest.fixture
def weth() -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(ChainId.ethereum.value, "0x1", "WETH", 18)


@pytest.fixture
def aave() -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(ChainId.ethereum.value, "0x3", "AAVE", 18)


@pytest.fixture
def weth_usdc(mock_exchange, usdc, weth) -> TradingPairIdentifier:
    """Mock some assets"""
    return TradingPairIdentifier(
        weth,
        usdc,
        "0x4",
        mock_exchange.address,
        internal_id=1,
        internal_exchange_id=mock_exchange.exchange_id)


@pytest.fixture
def aave_usdc(mock_exchange, usdc, aave) -> TradingPairIdentifier:
    """Mock some assets"""
    return TradingPairIdentifier(
        aave,
        usdc,
        "0x5",
        mock_exchange.address,
        internal_id=2,
        internal_exchange_id=mock_exchange.exchange_id
    )


@pytest.fixture
def start_ts(usdc, weth) -> datetime.datetime:
    """Timestamp of action started"""
    return datetime.datetime(2022, 1, 1, tzinfo=None)


@pytest.fixture()
def universe(
        mock_exchange,
        usdc,
        weth,
        weth_usdc,
        aave_usdc,
        start_ts) -> TradingStrategyUniverse:
    """Create a trading universe for tests.

    - WETH/USD and AAVE/USD pairs with fixed prices

    - Over several months
    """

    start_at = start_ts
    end_at = start_at + datetime.timedelta(days=365)

    # Set up fake assets
    mock_chain_id = ChainId.ethereum

    time_bucket = TimeBucket.d1

    pair_universe = create_pair_universe_from_code(mock_chain_id, [weth_usdc, aave_usdc])

    # Use these prices for the assets
    # throughout the tests
    price_map = {
        weth_usdc: 1670.0,
        aave_usdc: 100.0,
    }

    candles = generate_fixed_price_candles(
        time_bucket,
        start_at,
        end_at,
        price_map)

    candle_universe = GroupedCandleUniverse(
        candles,
        time_bucket
    )

    assert len(candle_universe.get_samples_by_pair(1)) > 0
    assert len(candle_universe.get_samples_by_pair(2)) > 0

    universe = Universe(
        time_bucket=time_bucket,
        chains={mock_chain_id},
        exchanges={mock_exchange},
        pairs=pair_universe,
        candles=candle_universe,
        liquidity=None
    )

    return TradingStrategyUniverse(universe=universe, reserve_assets=[usdc])


@pytest.fixture()
def routing_model(universe) -> BacktestRoutingModel:
    return generate_simple_routing_model(universe)


@pytest.fixture()
def pricing_model(routing_model, universe) -> BacktestSimplePricingModel:
    return BacktestSimplePricingModel(universe.universe.candles, routing_model)


@pytest.fixture
def single_asset_portfolio(start_ts, weth_usdc, weth, usdc) -> Portfolio:
    """Creates a mock portfolio that holds some reserve currency and WETH/USD pair."""
    p = Portfolio()

    reserve = ReservePosition(usdc, Decimal(500), start_ts, 1.0, start_ts)
    p.reserves[reserve.get_identifier()] = reserve

    trade = TradeExecution(
        trade_id = 1,
        position_id =1,
        trade_type = TradeType.rebalance,
        pair=weth_usdc,
        opened_at = start_ts,
        planned_quantity = Decimal(0.1),
        planned_price = 1670.0,
        planned_reserve=Decimal(167),
        reserve_currency = usdc,
        started_at = start_ts,
        reserve_currency_allocated = 167,
        broadcasted_at =start_ts,
        executed_at = start_ts,
        executed_price=1660,
        executed_quantity=Decimal(0.095),
        lp_fees_paid =2.5,
        native_token_price=1.9,
        planned_mid_price=None,
    )

    tx = BlockchainTransaction(
        tx_hash=HexBytes("0x01"),
        nonce=1,
        realised_gas_units_consumed=150_000,
        realised_gas_price=15,
    )
    trade.blockchain_transactions = [tx]

    assert trade.is_buy()
    assert trade.is_success()

    position = TradingPosition(
        position_id=1,
        pair=weth_usdc,
        opened_at=start_ts,
        last_token_price=1660,
        last_reserve_price=1,
        last_pricing_at=start_ts,
        reserve_currency=usdc,
        trades={1: trade},
    )

    p.open_positions = {1: position}
    p.next_trade_id = 2
    p.next_position_id = 2
    return p


def test_get_portfolio_weights(
    single_asset_portfolio: Portfolio,
    weth_usdc: TradingPairIdentifier,
):
    """"Get weights of exising portfolio."""

    portfolio = single_asset_portfolio

    weights = get_existing_portfolio_weights(portfolio)
    assert weights == {
        weth_usdc.internal_id: 1.0
    }


def test_weight_diffs(
    single_asset_portfolio: Portfolio,
    weth_usdc: TradingPairIdentifier,
    aave_usdc: TradingPairIdentifier,
):
    """"Create weight diffs between exising and new portfolio."""

    portfolio = single_asset_portfolio

    new_weights = {
        aave_usdc.internal_id: 1
    }

    existing = get_existing_portfolio_weights(portfolio)
    diffs = get_weight_diffs(existing, new_weights)

    assert diffs == {
        aave_usdc.internal_id: 1,
        weth_usdc.internal_id: -1
    }


def test_weight_diffs_partial(
    single_asset_portfolio: Portfolio,
    weth_usdc: TradingPairIdentifier,
    aave_usdc: TradingPairIdentifier,
):
    """"Create 50%/50% portfolio."""

    portfolio = single_asset_portfolio

    new_weights = {
        aave_usdc.internal_id: 0.5,
        weth_usdc.internal_id: 0.5,
    }

    existing = get_existing_portfolio_weights(portfolio)
    diffs = get_weight_diffs(existing, new_weights)

    assert diffs == {
        aave_usdc.internal_id: 0.5,
        weth_usdc.internal_id: -0.5
    }


def test_clip():
    """"Make sure normalised weights are summed precise 1."""

    weights = {
        0: 0.8,
        1: 0.21,
    }

    clipped = clip_to_normalised(weights)

    assert clipped == {
        0: 0.79,
        1: 0.21,
    }


def test_rebalance_trades_flip_position(
    single_asset_portfolio: Portfolio,
    universe,
    pricing_model,
    weth_usdc: TradingPairIdentifier,
    aave_usdc: TradingPairIdentifier,
    start_ts,
):
    """"Create trades for a portfolio to go from 100% ETH to 100% AAVE."""

    portfolio = single_asset_portfolio

    state = State(portfolio=portfolio)

    # Check we have WETH-USDC open
    position = state.portfolio.get_position_by_trading_pair(weth_usdc)
    assert position.get_quantity() > 0
    weth_quantity = position.get_quantity()

    # Create the PositionManager that
    # will create buy/sell trades based on our
    # trading universe and mock price feeds
    position_manager = PositionManager(
        start_ts + datetime.timedelta(days=1),  # Trade on t plus 1 day
        universe,
        state,
        pricing_model,
    )

    # Go all in to AAVE
    new_weights = {
        aave_usdc.internal_id: 1,
    }

    trades = rebalance_portfolio(
        position_manager,
        new_weights,
        portfolio.get_open_position_equity(),
    )

    assert len(trades) == 2, f"Got trades: {trades}"

    # Sells go first,
    # with 167 worth of sales
    t = trades[0]
    assert t.is_sell()
    assert t.is_planned()
    assert t.pair == weth_usdc
    assert t.planned_price == 1660
    assert t.planned_quantity == -weth_quantity
    assert t.get_planned_value() == 157.7

    # Buy comes next,
    # with approx the same value
    t = trades[1]
    assert t.is_buy()
    assert t.is_planned()
    assert t.planned_price == 100
    assert t.get_planned_value() ==  157.7


def test_rebalance_trades_flip_position_partial(
    single_asset_portfolio: Portfolio,
    universe,
    pricing_model,
    weth_usdc: TradingPairIdentifier,
    aave_usdc: TradingPairIdentifier,
    start_ts,
):
    """"Create trades for a portfolio to go to 70% ETH to 30% AAVE."""

    portfolio = single_asset_portfolio

    state = State(portfolio=portfolio)

    # Check we have WETH-USDC open
    position = state.portfolio.get_position_by_trading_pair(weth_usdc)
    assert position.get_quantity() > 0
    weth_quantity = position.get_quantity()

    # Create the PositionManager that
    # will create buy/sell trades based on our
    # trading universe and mock price feeds
    position_manager = PositionManager(
        start_ts + datetime.timedelta(days=1),  # Trade on t plus 1 day
        universe,
        state,
        pricing_model,
    )

    # Go all in to AAVE
    new_weights = {
        aave_usdc.internal_id: 0.3,
        weth_usdc.internal_id: 0.7,
    }

    trades = rebalance_portfolio(
        position_manager,
        new_weights,
        portfolio.get_open_position_equity(),
    )

    assert len(trades) == 2, f"Got trades: {trades}"

    # Sells go first,
    # with 167 worth of sales
    t = trades[0]
    assert t.is_sell()
    assert t.is_planned()
    assert t.pair == weth_usdc
    assert t.planned_price == 1660
    # Sell 30%
    assert t.planned_quantity == pytest.approx(-(weth_quantity * Decimal(0.3)))
    assert t.get_planned_value() == 47.31

    # Buy comes next,
    # with approx the same value
    t = trades[1]
    assert t.is_buy()
    assert t.is_planned()
    assert t.planned_price == 100
    assert t.get_planned_value() == pytest.approx(47.31)


def test_rebalance_bad_weights(
    single_asset_portfolio: Portfolio,
    universe,
    pricing_model,
    weth_usdc: TradingPairIdentifier,
    aave_usdc: TradingPairIdentifier,
    start_ts,
):
    """"Rebalance cannot be done if the sum of weights is not 1."""

    portfolio = single_asset_portfolio

    state = State(portfolio=portfolio)

    # Create the PositionManager that
    # will create buy/sell trades based on our
    # trading universe and mock price feeds
    position_manager = PositionManager(
        start_ts + datetime.timedelta(days=1),  # Trade on t plus 1 day
        universe,
        state,
        pricing_model,
    )

    # Go all in to AAVE
    new_weights = {
        aave_usdc.internal_id: 1.5,
        weth_usdc.internal_id: 1.5,
    }

    with pytest.raises(BadWeightsException):
        rebalance_portfolio(
            position_manager,
            new_weights,
            portfolio.get_open_position_equity(),
        )