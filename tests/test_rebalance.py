"""Test portfolio rebalancing.

- Tests low level functions

- Tests rebalancing trades using synthetic fixed price trading universe of
  WETH/USD and AAVE/USD

"""
import datetime
import random
from decimal import Decimal

import pytest
from hexbytes import HexBytes

from tradeexecutor.backtest.backtest_execution import BacktestExecution
from tradeexecutor.backtest.backtest_pricing import BacktestPricing
from tradeexecutor.backtest.backtest_routing import BacktestRoutingModel, BacktestRoutingState
from tradeexecutor.backtest.simulated_wallet import SimulatedWallet
from tradeexecutor.state.size_risk import SizeRisk
from tradeexecutor.state.state import State, TradeType
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.blockhain_transaction import BlockchainTransaction
from tradeexecutor.state.reserve import ReservePosition
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.strategy import size_risk_model
from tradeexecutor.strategy.alpha_model import AlphaModel
from tradeexecutor.strategy.fixed_size_risk import FixedSizeRiskModel
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.pandas_trader.rebalance import get_existing_portfolio_weights, rebalance_portfolio_old, \
    get_weight_diffs
from tradeexecutor.strategy.weighting import BadWeightsException, clip_to_normalised, weight_passthrouh
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, create_pair_universe_from_code
from tradeexecutor.testing.synthetic_ethereum_data import generate_random_ethereum_address
from tradeexecutor.testing.synthetic_exchange_data import generate_exchange, generate_simple_routing_model
from tradeexecutor.testing.synthetic_lending_data import generate_lending_reserve, generate_lending_universe
from tradeexecutor.testing.synthetic_price_data import generate_fixed_price_candles
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import Exchange
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe


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
        internal_exchange_id=mock_exchange.exchange_id,
        fee=0.0030,
    )


@pytest.fixture
def aave_usdc(mock_exchange, usdc, aave) -> TradingPairIdentifier:
    """Mock some assets"""
    return TradingPairIdentifier(
        aave,
        usdc,
        "0x5",
        mock_exchange.address,
        internal_id=2,
        internal_exchange_id=mock_exchange.exchange_id,
        fee=0.0030,
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

    # Generate lending data for shorting
    #
    # We can only short WETH, not AAVE
    #
    usdc_reserve = generate_lending_reserve(usdc, mock_chain_id, 1)
    weth_reserve = generate_lending_reserve(weth, mock_chain_id, 2)

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

    universe = Universe(
        time_bucket=time_bucket,
        chains={mock_chain_id},
        exchanges={mock_exchange},
        pairs=pair_universe,
        candles=candle_universe,
        liquidity=None,
        lending_candles=lending_candle_universe,
    )

    return TradingStrategyUniverse(
        data_universe=universe,
        reserve_assets=[usdc]
    )


@pytest.fixture()
def strategy_universe(universe):
    """Legacy alias. Use strategy_universe."""
    return universe


@pytest.fixture()
def routing_model(universe) -> BacktestRoutingModel:
    return generate_simple_routing_model(universe)


@pytest.fixture()
def pricing_model(routing_model, universe) -> BacktestPricing:
    return BacktestPricing(universe.data_universe.candles, routing_model)


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


@pytest.fixture()
def wallet(usdc, weth) -> SimulatedWallet:
    """Dummy blockchain accounting to see we do not try to use tokens we do not have."""
    wallet = SimulatedWallet()
    wallet.set_balance(usdc.address, Decimal(10_000))
    wallet.set_balance(weth.address, Decimal(10))
    return wallet


@pytest.fixture()
def execution_model(wallet) -> BacktestExecution:
    """Simulate trades using backtest execution model."""
    execution_model = BacktestExecution(wallet)
    return execution_model


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
        strategy_universe,
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
        strategy_universe.data_universe,
        state,
        pricing_model,
    )

    # Go all in to AAVE
    new_weights = {
        aave_usdc.internal_id: 1,
    }

    trades = rebalance_portfolio_old(
        position_manager,
        new_weights,
        portfolio.get_position_equity_and_loan_nav(),
    )

    assert len(trades) == 2, f"Got trades: {trades}"

    # Sells go first,
    # with 167 worth of sales
    t = trades[0]
    assert t.is_sell()
    assert t.is_planned()
    assert t.pair == weth_usdc
    assert t.planned_price == 1664.99
    assert t.planned_quantity == Decimal('-0.0944311377245508898337078562690294347703456878662109375')
    assert t.get_planned_value() == pytest.approx(157.2269) # 157.7

    # Buy comes next,
    # with approx the same value
    t = trades[1]
    assert t.is_buy()
    assert t.is_planned()
    assert t.planned_price == pytest.approx(100.29999999999998)
    assert t.get_planned_value() ==  157.7


def test_rebalance_trades_flip_position_partial(
    single_asset_portfolio: Portfolio,
        strategy_universe,
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
        strategy_universe.data_universe,
        state,
        pricing_model,
    )

    # Go all in to AAVE
    new_weights = {
        aave_usdc.internal_id: 0.3,
        weth_usdc.internal_id: 0.7,
    }

    trades = rebalance_portfolio_old(
        position_manager,
        new_weights,
        portfolio.get_position_equity_and_loan_nav(),
    )

    assert len(trades) == 2, f"Got trades: {trades}"

    # Sells go first,
    # with 167 worth of sales
    t = trades[0]
    assert t.is_sell()
    assert t.is_planned()
    assert t.pair == weth_usdc
    assert t.planned_price == pytest.approx(1664.99)
    # Sell 30%
    #assert t.planned_quantity == pytest.approx(-(weth_quantity * Decimal(0.3)))
    assert t.planned_quantity == pytest.approx(Decimal("-0.0283293413173652704195593088343230192549526691436767578125"))
    assert t.get_planned_value() == pytest.approx(47.16807)

    # Buy comes next,
    # with approx the same value
    t = trades[1]
    assert t.is_buy()
    assert t.is_planned()
    assert t.planned_price == pytest.approx(100.29999999999998)
    assert t.get_planned_value() == pytest.approx(47.31)


def test_rebalance_bad_weights(
    single_asset_portfolio: Portfolio,
        strategy_universe,
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
        strategy_universe.data_universe,
        state,
        pricing_model,
    )

    # Go all in to AAVE
    new_weights = {
        aave_usdc.internal_id: 1.5,
        weth_usdc.internal_id: 1.5,
    }

    with pytest.raises(BadWeightsException):
        rebalance_portfolio_old(
            position_manager,
            new_weights,
            portfolio.get_position_equity_and_loan_nav(),
        )


def test_alpha_model_trades_flip_position(
    single_asset_portfolio: Portfolio,
        strategy_universe,
    pricing_model,
    weth_usdc: TradingPairIdentifier,
    aave_usdc: TradingPairIdentifier,
    start_ts,
):
    """"Create trades for a portfolio to go from 100% ETH to 100% AAVE.

    Uses AlphaModel based approach
    """

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
        strategy_universe.data_universe,
        state,
        pricing_model,
    )

    # Go all in to AAVE
    alpha_model = AlphaModel()
    alpha_model.set_signal(aave_usdc, 1.0)
    alpha_model.select_top_signals(9999)
    alpha_model.assign_weights()
    alpha_model.normalise_weights()

    # Load in old weight for each trading pair signal,
    # so we can calculate the adjustment trade size
    alpha_model.update_old_weights(state.portfolio)

    # Calculate how much dollar value we want each individual position to be on this strategy cycle,
    # based on our total available equity
    portfolio = position_manager.get_current_portfolio()
    portfolio_target_value = portfolio.get_position_equity_and_loan_nav()
    alpha_model.calculate_target_positions(position_manager, portfolio_target_value)

    # Shift portfolio from current positions to target positions
    # determined by the alpha signals (momentum)
    trades = alpha_model.generate_rebalance_trades_and_triggers(position_manager)

    assert len(trades) == 2, f"Got trades: {trades}"

    # Sells go first,
    # with 167 worth of sales
    t = trades[0]
    assert t.is_sell()
    assert t.is_planned()
    assert t.pair == weth_usdc
    assert t.planned_price == pytest.approx(1664.99)
    # assert t.planned_quantity == -weth_quantity
    # assert t.get_planned_value() == 157.7
    assert t.planned_quantity == pytest.approx(Decimal('-0.0950000000000000011102230246'))
    assert t.get_planned_value() == pytest.approx(158.17405)

    # Buy comes next,
    # with approx the same value
    t = trades[1]
    assert t.is_buy()
    assert t.is_planned()
    assert t.planned_price == pytest.approx(100.29999999999998)
    assert t.get_planned_value() == 157.7


def test_alpha_model_flip_position_partially(
    single_asset_portfolio: Portfolio,
        strategy_universe,
    pricing_model,
    weth_usdc: TradingPairIdentifier,
    aave_usdc: TradingPairIdentifier,
    start_ts,
):
    """"Create trades for a portfolio to go from 100% ETH to 50% AAVE / 50% ETH.

    Uses AlphaModel based approach.
    """

    portfolio = single_asset_portfolio

    state = State(portfolio=portfolio)

    # Check we have WETH-USDC open
    position = state.portfolio.get_position_by_trading_pair(weth_usdc)
    assert position.get_quantity() > 0
    weth_quantity = position.get_quantity()
    assert position.get_value() == pytest.approx(157.7)

    # Create the PositionManager that
    # will create buy/sell trades based on our
    # trading universe and mock price feeds
    position_manager = PositionManager(
        start_ts + datetime.timedelta(days=1),  # Trade on t plus 1 day
        strategy_universe.data_universe,
        state,
        pricing_model,
    )

    # Go 50% in to AAVE
    alpha_model = AlphaModel()
    alpha_model.set_signal(aave_usdc, 0.5)
    alpha_model.set_signal(weth_usdc, 0.5)
    alpha_model.select_top_signals(9999)
    alpha_model.assign_weights(method=weight_passthrouh)
    alpha_model.normalise_weights()

    # Check we have 50% / 50%
    for signal in alpha_model.iterate_signals():
        assert signal.raw_weight == 0.5, f"Bad signal: {signal}"
        assert signal.normalised_weight == 0.5, f"Bad signal: {signal}"

    # Load in old weight for each trading pair signal,
    # so we can calculate the adjustment trade size
    alpha_model.update_old_weights(state.portfolio)
    assert alpha_model.get_signal_by_pair(weth_usdc).old_value == pytest.approx(157.7)

    # Calculate how much dollar value we want each individual position to be on this strategy cycle,
    # based on our total available equity
    portfolio = position_manager.get_current_portfolio()
    portfolio_target_value = portfolio.get_position_equity_and_loan_nav()
    alpha_model.calculate_target_positions(position_manager, portfolio_target_value)

    # Check we have 50% / 50%
    assert alpha_model.get_signal_by_pair(weth_usdc).position_adjust_usd < 0
    assert alpha_model.get_signal_by_pair(aave_usdc).position_adjust_usd > 0

    # Shift portfolio from current positions to target positions
    # determined by the alpha signals (momentum)
    trades = alpha_model.generate_rebalance_trades_and_triggers(position_manager)

    assert len(trades) == 2, f"Got trades: {trades}"

    # Sells go first,
    # sell 50% of ETH
    t = trades[0]
    assert t.is_sell()
    assert t.is_planned()
    assert t.pair == weth_usdc
    assert t.planned_price == pytest.approx(1664.99)
    # assert t.planned_quantity == pytest.approx(-weth_quantity / 2)
    # assert t.get_planned_value() == 78.85
    assert t.planned_quantity == pytest.approx(Decimal("-0.04721556886227544491685392813451471738517284393310546875"))
    assert t.get_planned_value() == pytest.approx(78.61345)

    # Buy comes next,
    # buy 50% of AAVE
    t = trades[1]
    assert t.is_buy()
    assert t.is_planned()
    assert t.planned_price == pytest.approx(100.29999999999998)
    assert t.get_planned_value() == 78.85


def test_alpha_model_open_short(
    single_asset_portfolio: Portfolio,
        strategy_universe,
    pricing_model,
    weth_usdc: TradingPairIdentifier,
    aave_usdc: TradingPairIdentifier,
    start_ts,
):
    """"Short asset on a negative signal.

    - On a rebalance short one asset, while longing other
    """

    # Check that shorting is enabled
    assert strategy_universe.can_open_short(start_ts, weth_usdc)
    assert not strategy_universe.can_open_short(start_ts, aave_usdc)

    portfolio = single_asset_portfolio

    state = State(portfolio=portfolio)

    # Check we have WETH-USDC open
    position = state.portfolio.get_position_by_trading_pair(weth_usdc)
    assert position.get_quantity() > 0
    weth_quantity = position.get_quantity()
    assert position.get_value() == pytest.approx(157.7)

    # Create the PositionManager that
    # will create buy/sell trades based on our
    # trading universe and mock price feeds
    position_manager = PositionManager(
        start_ts + datetime.timedelta(days=1),  # Trade on t plus 1 day
        strategy_universe,
        state,
        pricing_model,
    )

    alpha_model = AlphaModel()
    alpha_model.set_signal(aave_usdc, 0.5)  # 50% long AAVE
    alpha_model.set_signal(weth_usdc, -0.5, leverage=1.0)  # 505 short ETH
    alpha_model.select_top_signals(count=5)
    alpha_model.assign_weights(method=weight_passthrouh)
    alpha_model.normalise_weights()

    # Check we have 50% / 50%
    for signal in alpha_model.iterate_signals():
        assert signal.raw_weight == 0.5, f"Bad signal: {signal}"
        assert signal.normalised_weight == 0.5, f"Bad signal: {signal}"

    # Load in old weight for each trading pair signal,
    # so we can calculate the adjustment trade size
    alpha_model.update_old_weights(state.portfolio)
    weth_signal = alpha_model.get_signal_by_pair(weth_usdc)
    assert weth_signal.old_value == pytest.approx(157.7)
    assert weth_signal.old_pair == weth_usdc

    # Calculate how much dollar value we want each individual position to be on this strategy cycle,
    # based on our total available equity
    portfolio = position_manager.get_current_portfolio()
    portfolio_target_value = portfolio.get_position_equity_and_loan_nav()
    alpha_model.calculate_target_positions(position_manager, portfolio_target_value)

    assert weth_signal.position_adjust_usd == pytest.approx(78.85)  # For flipped position, we adjust from zero to size
    assert weth_signal.is_flipping()

    aave_signal = alpha_model.get_signal_by_pair(aave_usdc)
    assert aave_signal.position_adjust_usd > 0  # increase position
    assert not aave_signal.is_flipping()

    # Shift portfolio from current positions to target positions
    # determined by the alpha signals (momentum)
    trades = alpha_model.generate_rebalance_trades_and_triggers(position_manager)

    assert weth_signal.is_short()  # The position flips from spot to short
    assert aave_signal.is_spot()

    # Close spot ETH
    # Open ETH short
    # Open Aave spot
    assert len(trades) == 3, f"Got trades: {trades}"

    # Closing spot ETH comes first,
    # sell 100% of ETH spot
    t = trades[0]
    assert t.is_sell()
    assert t.is_spot()
    assert t.is_planned()
    assert t.pair == weth_usdc
    assert t.planned_price == pytest.approx(1664.99)
    assert t.planned_quantity == pytest.approx(Decimal("-0.09500000000000000111022302463"))  # Close all existing
    assert t.get_planned_value() == pytest.approx(158.17405)

    # Opening ETH short comes next
    # Use 50% of portfolio value to go short on ETH
    weth_usdc_short_pair = strategy_universe.get_shorting_pair(weth_usdc)
    t = trades[2]
    assert t.is_sell(), f"{t}"
    assert t.is_short(), f"{t}"
    assert t.is_planned(), f"{t}"
    assert t.pair == weth_usdc_short_pair
    assert t.planned_price == pytest.approx(1664.99)
    assert t.planned_quantity == pytest.approx(Decimal(-0.04721556886227544569799887059))

    # Spot buy comes next,
    # buy 50% of AAVE
    t = trades[1]
    assert t.is_buy()
    assert t.is_spot()
    assert t.is_planned()
    assert t.planned_price == pytest.approx(100.29999999999998)
    assert t.get_planned_value() == 78.85


def test_alpha_model_increase_short(
    single_asset_portfolio: Portfolio,
    universe: TradingStrategyUniverse,
    pricing_model,
    weth_usdc: TradingPairIdentifier,
    aave_usdc: TradingPairIdentifier,
    start_ts,
    execution_model: BacktestExecution,
    routing_model: BacktestRoutingModel,
    wallet: SimulatedWallet,
):
    """"Increase short position by going from 50% to 75% on the short signal.

    - Follow the steps from the ``test_alpha_model_open_short``,
      and do another cycle where short is increased

    """

    strategy_universe = universe

    portfolio = single_asset_portfolio

    state = State(portfolio=portfolio)

    # Create the PositionManager that
    # will create buy/sell trades based on our
    # trading universe and mock price feeds
    position_manager = PositionManager(
        start_ts + datetime.timedelta(days=1),  # Trade on t plus 1 day
        strategy_universe,
        state,
        pricing_model,
    )

    alpha_model = AlphaModel()
    alpha_model.set_signal(aave_usdc, 0.5)  # 50% long AAVE
    alpha_model.set_signal(weth_usdc, -0.5, leverage=1.0)  # 50% short ETH
    alpha_model.select_top_signals(count=5)
    alpha_model.assign_weights(method=weight_passthrouh)
    alpha_model.normalise_weights()

    # Load in old weight for each trading pair signal,
    # so we can calculate the adjustment trade size
    alpha_model.update_old_weights(state.portfolio)

    # Calculate how much dollar value we want each individual position to be on this strategy cycle,
    # based on our total available equity
    portfolio = position_manager.get_current_portfolio()
    portfolio_target_value = portfolio.get_position_equity_and_loan_nav()
    alpha_model.calculate_target_positions(position_manager, portfolio_target_value)

    # Shift portfolio from current positions to target positions
    # determined by the alpha signals (momentum)
    trades = alpha_model.generate_rebalance_trades_and_triggers(position_manager)

    # Close spot ETH
    # Open ETH short
    # Open Aave spot
    assert len(trades) == 3, f"Got trades: {trades}"

    # Closing spot ETH comes first,
    # sell 100% of ETH spot
    t = trades[0]
    assert t.is_sell()
    assert t.is_spot()

    # Opening ETH short comes next
    # Use 50% of portfolio value to go short on ETH
    t = trades[2]
    assert t.is_sell()
    assert t.is_short()

    # Spot buy comes next,
    # buy 50% of AAVE
    t = trades[1]
    assert t.is_buy()
    assert t.is_spot()

    #
    # Now do another cycle where we go even more short
    #
    balance_1_ts = start_ts + datetime.timedelta(days=1)

    execution_model.execute_trades(
        balance_1_ts,
        state,
        trades,
        routing_model,
        BacktestRoutingState(strategy_universe.data_universe.pairs, wallet),  # Create new routing state at the start of the cycle
    )
    assert any(t.is_success() for t in trades)

    # Check out position status
    p1 = portfolio.closed_positions[1]
    assert p1.is_closed()

    p2 = portfolio.open_positions[2]
    assert p2.pair == aave_usdc
    assert p2.pair.is_spot()
    assert p2.get_value() == pytest.approx(78.85)

    lending_reserves = strategy_universe.data_universe.lending_reserves
    for r in lending_reserves.reserves.items():
        print(r)

    vweth_ausdc = strategy_universe.get_shorting_pair(weth_usdc)
    p3 = portfolio.open_positions[3]
    assert p3.pair == vweth_ausdc
    assert p3.pair.is_short()
    assert p3.get_value() == pytest.approx(78.85)

    assert portfolio.get_net_asset_value() == pytest.approx(658.17405)

    # Increase short
    balance_1_ts = start_ts + datetime.timedelta(days=1)
    alpha_model = AlphaModel()
    alpha_model.set_signal(aave_usdc, 0.3)  # 30% long AAVE
    alpha_model.set_signal(weth_usdc, -0.7, leverage=1.0)  # 70% short ETH
    alpha_model.select_top_signals(count=5)
    alpha_model.assign_weights(method=weight_passthrouh)
    alpha_model.normalise_weights()

    # Load in old weight for each trading pair signal,
    # so we can calculate the adjustment trade size
    alpha_model.update_old_weights(state.portfolio)

    # Calculate how much dollar value we want each individual position to be on this strategy cycle,
    # based on our total available equity
    portfolio = position_manager.get_current_portfolio()
    portfolio_target_value = portfolio.get_position_equity_and_loan_nav()
    alpha_model.calculate_target_positions(position_manager, portfolio_target_value)

    # Shift portfolio from current positions to target positions
    # determined by the alpha signals (momentum)
    trades = alpha_model.generate_rebalance_trades_and_triggers(position_manager)

    # Increase ETH short
    # Decrease Aave spot
    assert len(trades) == 2, f"Got trades: {trades}"

    balance_2_ts = balance_1_ts + datetime.timedelta(days=1)

    execution_model.execute_trades(
        balance_2_ts,
        state,
        trades,
        routing_model,
        BacktestRoutingState(strategy_universe.data_universe.pairs, wallet),  # Create new routing state at the start of the cycle
    )
    assert any(t.is_success() for t in trades)

    assert len(portfolio.open_positions) == 2

    assert p2.pair == aave_usdc
    assert p2.is_spot()
    assert p2.get_value() == pytest.approx(47.21538)  # Down in value

    assert p3.pair == vweth_ausdc
    assert p3.is_short()
    assert p3.get_value() == pytest.approx(110.39)  # Up in value

    assert portfolio.get_net_asset_value() == pytest.approx(657.98481)


def test_alpha_model_decrease_short(
    single_asset_portfolio: Portfolio,
    universe: TradingStrategyUniverse,
    pricing_model,
    weth_usdc: TradingPairIdentifier,
    aave_usdc: TradingPairIdentifier,
    start_ts,
    execution_model: BacktestExecution,
    routing_model: BacktestRoutingModel,
    wallet: SimulatedWallet,
):
    """"Decrease short position by going from 50% to 25% on the short signal.

    - Follow the steps from the ``test_alpha_model_open_short``,
      and do another cycle where short is increased
    """

    strategy_universe = universe

    portfolio = single_asset_portfolio

    state = State(portfolio=portfolio)

    # Create the PositionManager that
    # will create buy/sell trades based on our
    # trading universe and mock price feeds
    position_manager = PositionManager(
        start_ts + datetime.timedelta(days=1),  # Trade on t plus 1 day
        strategy_universe,
        state,
        pricing_model,
    )

    alpha_model = AlphaModel()
    alpha_model.set_signal(aave_usdc, 0.5)  # 50% long AAVE
    alpha_model.set_signal(weth_usdc, -0.5, leverage=1.0)  # 50% short ETH
    alpha_model.select_top_signals(count=5)
    alpha_model.assign_weights(method=weight_passthrouh)
    alpha_model.normalise_weights()

    # Load in old weight for each trading pair signal,
    # so we can calculate the adjustment trade size
    alpha_model.update_old_weights(state.portfolio)

    # Calculate how much dollar value we want each individual position to be on this strategy cycle,
    # based on our total available equity
    portfolio = position_manager.get_current_portfolio()
    portfolio_target_value = portfolio.get_position_equity_and_loan_nav()
    alpha_model.calculate_target_positions(position_manager, portfolio_target_value)

    # Shift portfolio from current positions to target positions
    # determined by the alpha signals (momentum)
    trades = alpha_model.generate_rebalance_trades_and_triggers(position_manager)

    # Close spot ETH
    # Open ETH short
    # Open Aave spot
    assert len(trades) == 3, f"Got trades: {trades}"

    # Closing spot ETH comes first,
    # sell 100% of ETH spot
    t = trades[0]
    assert t.is_sell()
    assert t.is_spot()

    # Opening ETH short comes next
    # Use 50% of portfolio value to go short on ETH
    t = trades[2]
    assert t.is_sell()
    assert t.is_short()

    # Spot buy comes next,
    # buy 50% of AAVE
    t = trades[1]
    assert t.is_buy()
    assert t.is_spot()

    #
    # Now do another cycle where we go even more short
    #
    balance_1_ts = start_ts + datetime.timedelta(days=1)

    execution_model.execute_trades(
        balance_1_ts,
        state,
        trades,
        routing_model,
        BacktestRoutingState(strategy_universe.data_universe.pairs, wallet),  # Create new routing state at the start of the cycle
    )
    assert any(t.is_success() for t in trades)

    # Check out position status
    p1 = portfolio.closed_positions[1]
    assert p1.is_closed()

    p2 = portfolio.open_positions[2]
    assert p2.pair == aave_usdc
    assert p2.pair.is_spot()
    assert p2.get_value() == pytest.approx(78.85)

    lending_reserves = strategy_universe.data_universe.lending_reserves
    for r in lending_reserves.reserves.items():
        print(r)

    vweth_ausdc = strategy_universe.get_shorting_pair(weth_usdc)
    p3 = portfolio.open_positions[3]
    assert p3.pair == vweth_ausdc
    assert p3.pair.is_short()
    assert p3.get_value() == pytest.approx(78.85)
    assert p3.loan.get_leverage() == pytest.approx(0.997)

    assert portfolio.get_net_asset_value() == pytest.approx(658.17405)
    assert portfolio.get_cash() == pytest.approx(500.47405000000003)

    # Increase short
    alpha_model = AlphaModel()
    alpha_model.set_signal(aave_usdc, 0.75)  # 30% long AAVE
    alpha_model.set_signal(weth_usdc, -0.25, leverage=1.0)  # 70% short ETH
    alpha_model.select_top_signals(count=5)
    alpha_model.assign_weights(method=weight_passthrouh)
    alpha_model.normalise_weights()

    # Load in old weight for each trading pair signal,
    # so we can calculate the adjustment trade size
    alpha_model.update_old_weights(state.portfolio)

    # Calculate how much dollar value we want each individual position to be on this strategy cycle,
    # based on our total available equity
    portfolio = position_manager.get_current_portfolio()
    portfolio_target_value = portfolio.get_position_equity_and_loan_nav()
    alpha_model.calculate_target_positions(position_manager, portfolio_target_value)

    # Shift portfolio from current positions to target positions
    # determined by the alpha signals (momentum)
    trades = alpha_model.generate_rebalance_trades_and_triggers(position_manager)

    # Increase ETH short
    # Decrease Aave spot
    assert len(trades) == 2, f"Got trades: {trades}"

    balance_2_ts = balance_1_ts + datetime.timedelta(days=1)
    execution_model.execute_trades(
        balance_2_ts,
        state,
        trades,
        routing_model,
        BacktestRoutingState(strategy_universe.data_universe.pairs, wallet),  # Create new routing state at the start of the cycle
    )
    assert any(t.is_success() for t in trades)

    assert len(portfolio.open_positions) == 2
    assert portfolio.get_cash() == pytest.approx(500.47405000000003)

    assert p2.pair == aave_usdc
    assert p2.is_spot()
    assert p2.get_value() == pytest.approx(118.275)  # Up in value

    assert p3.pair == vweth_ausdc
    assert p3.is_short()
    assert p3.loan.get_leverage() == pytest.approx(0.991044785047922)
    assert p3.get_value() == pytest.approx(39.542920175)  # Down in value

    assert portfolio.get_net_asset_value() == pytest.approx(658.291970175)


def test_alpha_model_normalise_weight_capped(
    single_asset_portfolio: Portfolio,
    strategy_universe,
    pricing_model,
    weth_usdc: TradingPairIdentifier,
    aave_usdc: TradingPairIdentifier,
    start_ts,
):
    """"Cap the max allocation per asset."""

    portfolio = single_asset_portfolio

    state = State(portfolio=portfolio)

    # Check we have WETH-USDC open
    position = state.portfolio.get_position_by_trading_pair(weth_usdc)
    assert position.get_quantity() > 0
    weth_quantity = position.get_quantity()
    assert position.get_value() == pytest.approx(157.7)

    # Create the PositionManager that
    # will create buy/sell trades based on our
    # trading universe and mock price feeds
    position_manager = PositionManager(
        start_ts + datetime.timedelta(days=1),  # Trade on t plus 1 day
        strategy_universe.data_universe,
        state,
        pricing_model,
    )

    # Go 50% in to AAVE
    alpha_model = AlphaModel()
    alpha_model.set_signal(aave_usdc, 0.5)
    alpha_model.set_signal(weth_usdc, 0.5)
    alpha_model.select_top_signals(9999)
    alpha_model.assign_weights(method=weight_passthrouh)
    alpha_model.normalise_weights(max_weight=0.1)

    # Check we have 50% / 50%
    for signal in alpha_model.iterate_signals():
        assert signal.raw_weight == 0.5, f"Bad signal: {signal}"
        assert signal.normalised_weight == 0.1, f"Bad signal: {signal}"

    # Load in old weight for each trading pair signal,
    # so we can calculate the adjustment trade size
    alpha_model.update_old_weights(state.portfolio)
    assert alpha_model.get_signal_by_pair(weth_usdc).old_value == pytest.approx(157.7)

    # Calculate how much dollar value we want each individual position to be on this strategy cycle,
    # based on our total available equity
    portfolio = position_manager.get_current_portfolio()
    portfolio_target_value = portfolio.get_position_equity_and_loan_nav()
    alpha_model.calculate_target_positions(position_manager, portfolio_target_value)

    # Check we have 50% / 50%
    assert alpha_model.get_signal_by_pair(weth_usdc).position_adjust_usd < 0
    assert alpha_model.get_signal_by_pair(aave_usdc).position_adjust_usd > 0

    # Shift portfolio from current positions to target positions
    # determined by the alpha signals (momentum)
    trades = alpha_model.generate_rebalance_trades_and_triggers(position_manager)

    assert len(trades) == 2, f"Got trades: {trades}"

    # Sells go first,
    # sell 50% of ETH
    t = trades[0]
    assert t.is_sell()
    assert t.is_planned()
    assert t.pair == weth_usdc
    assert t.planned_price == pytest.approx(1664.99)
    # assert t.planned_quantity == pytest.approx(-weth_quantity / 2)
    # assert t.get_planned_value() == 78.85
    assert t.planned_quantity == pytest.approx(Decimal("-0.08498802395209580085033707064212649129331111907958984375"))
    assert t.get_planned_value() == pytest.approx(141.50421)

    # Buy comes next,
    # buy 50% of AAVE
    t = trades[1]
    assert t.is_buy()
    assert t.is_planned()
    assert t.planned_price == pytest.approx(100.29999999999998)
    assert t.get_planned_value() == pytest.approx(15.769999999999998)


def test_alpha_model_normalise_weight_size_risk(
    single_asset_portfolio: Portfolio,
    strategy_universe,
    pricing_model,
    weth_usdc: TradingPairIdentifier,
    aave_usdc: TradingPairIdentifier,
    start_ts,
):
    """"Cap the max allocation per asset using SizeRiskModel."""

    portfolio = single_asset_portfolio

    state = State(portfolio=portfolio)

    # Check we have WETH-USDC open
    position = state.portfolio.get_position_by_trading_pair(weth_usdc)
    assert position.get_quantity() > 0
    assert position.get_value() == pytest.approx(157.7)

    # Create the PositionManager that
    # will create buy/sell trades based on our
    # trading universe and mock price feeds
    position_manager = PositionManager(
        start_ts + datetime.timedelta(days=1),  # Trade on t plus 1 day
        strategy_universe.data_universe,
        state,
        pricing_model,
    )

    # Set the limits how large US-nominated positions can be.
    # We use artificially low 40 USD per position here and
    # both positions should be reduced to 40 USD.
    size_risker = FixedSizeRiskModel(
        pricing_model=pricing_model,
        per_trade_cap=5000,  # Limit individual trades to 5000 USD
        per_position_cap=40,  # Limit individual position sizes to 40 USD
    )

    # Calculate how much dollar value we want each individual position to be on this strategy cycle,
    # based on our total available equity
    portfolio = position_manager.get_current_portfolio()
    portfolio_target_value = portfolio.get_position_equity_and_loan_nav()

    # Try to Go 50% in to AAVE / 50% ETH
    # but we will be capped
    alpha_model = AlphaModel()
    alpha_model.set_signal(aave_usdc, 0.5)
    alpha_model.set_signal(weth_usdc, 0.5)
    alpha_model.select_top_signals(9999)
    alpha_model.assign_weights(method=weight_passthrouh)

    # Calculate weighted normals, capped by size risk
    alpha_model.normalise_weights(
        investable_equity=portfolio_target_value,
        size_risk_model=size_risker,
    )

    # Load in old weight for each trading pair signal,
    # so we can calculate the adjustment trade size
    alpha_model.update_old_weights(state.portfolio)
    assert alpha_model.get_signal_by_pair(weth_usdc).old_value == pytest.approx(157.7)

    alpha_model.calculate_target_positions(
        position_manager,
        portfolio_target_value,
    )

    # Check that signals are capped
    for signal in alpha_model.iterate_signals():
        assert isinstance(signal.position_size_risk, SizeRisk)
        assert signal.position_size_risk.asked_size > 40
        assert signal.position_size_risk.accepted_size == 40
        assert signal.position_size_risk.diagnostics_data["cap"] == 40
        assert signal.position_target == 40.0

    # Shift portfolio from current positions to target positions
    # determined by the alpha signals (momentum)
    trades = alpha_model.generate_rebalance_trades_and_triggers(position_manager)

    assert len(trades) == 2, f"Got trades: {trades}"

    # Sells go first,
    # sell ETH until we hit 40 USD
    t = trades[0]
    assert t.is_sell()
    assert t.is_planned()
    assert t.pair == weth_usdc
    assert t.planned_price == pytest.approx(1664.99)
    assert t.planned_quantity == pytest.approx(Decimal("-0.070479041916167661785408427022048272192478179931640625"))
    assert t.get_planned_value() == pytest.approx(117.3468999999999)  # 157 - 117 = 40
    assert isinstance(t.position_size_risk, SizeRisk)

    # Buy comes next,
    # buy Aave, enter max 40 USD
    t = trades[1]
    assert t.is_buy()
    assert t.is_planned()
    assert t.planned_price == pytest.approx(100.29999999999998)
    assert t.get_planned_value() == pytest.approx(40)
    assert isinstance(t.position_size_risk, SizeRisk)


def test_alpha_model_normalise_weight_size_risk_partial(
    single_asset_portfolio: Portfolio,
    strategy_universe,
    pricing_model,
    weth_usdc: TradingPairIdentifier,
    aave_usdc: TradingPairIdentifier,
    start_ts,
):
    """"Cap the max allocation per asset using SizeRiskModel, but only one of the asset is size risked."""

    portfolio = single_asset_portfolio

    state = State(portfolio=portfolio)

    # Check we have WETH-USDC open
    position = state.portfolio.get_position_by_trading_pair(weth_usdc)
    assert position.get_quantity() > 0
    assert position.get_value() == pytest.approx(157.7)

    # Create the PositionManager that
    # will create buy/sell trades based on our
    # trading universe and mock price feeds
    position_manager = PositionManager(
        start_ts + datetime.timedelta(days=1),  # Trade on t plus 1 day
        strategy_universe.data_universe,
        state,
        pricing_model,
    )

    # Set the limits how large US-nominated positions can be.
    # We use artificially low 50 USD per position here and
    # both positions should be reduced to 50 USD.
    size_risker = FixedSizeRiskModel(
        pricing_model=pricing_model,
        per_trade_cap=5000,  # Limit individual trades to 5000 USD
        per_position_cap=50,  # Limit individual position sizes to 40 USD
    )

    # Calculate how much dollar value we want each individual position to be on this strategy cycle,
    # based on our total available equity
    portfolio = position_manager.get_current_portfolio()
    portfolio_target_value = portfolio.get_position_equity_and_loan_nav()

    # Try to go 30% Aave, 70% ETH
    # Uncapped this would be
    # 70% * 157 USD for ETH
    # 30% * 157 USD for Aave = $47.1
    # Only ETH position will be capped.
    # The remaining equity from ETH position is distributed Aave position,
    # so that Aave will be $47.1 -> $50.
    alpha_model = AlphaModel()
    alpha_model.set_signal(aave_usdc, 0.3)
    alpha_model.set_signal(weth_usdc, 0.7)
    alpha_model.select_top_signals(9999)
    alpha_model.assign_weights(method=weight_passthrouh)

    # Calculate weighted normals, capped by size risk.
    alpha_model.normalise_weights(
        investable_equity=portfolio_target_value,
        size_risk_model=size_risker,
    )

    # Load in old weight for each trading pair signal,
    # so we can calculate the adjustment trade size
    alpha_model.update_old_weights(state.portfolio)
    assert alpha_model.get_signal_by_pair(weth_usdc).old_value == pytest.approx(157.7)

    alpha_model.calculate_target_positions(
        position_manager,
    )

    eth_signal = alpha_model.signals[weth_usdc.internal_id]
    aave_signal = alpha_model.signals[aave_usdc.internal_id]

    assert eth_signal.position_size_risk.capped
    assert eth_signal.position_size_risk.accepted_size == pytest.approx(50)
    assert aave_signal.position_size_risk.capped
    assert aave_signal.position_size_risk.accepted_size == pytest.approx(50)
    assert alpha_model.investable_equity == pytest.approx(157.7)
    assert alpha_model.accepted_investable_equity == pytest.approx(100)

    # Shift portfolio from current positions to target positions
    # determined by the alpha signals (momentum)
    trades = alpha_model.generate_rebalance_trades_and_triggers(position_manager)

    assert len(trades) == 2, f"Got trades: {trades}"

    # Sells go first,
    # sell ETH until we hit 50 USD
    t = trades[0]
    assert t.is_sell()
    assert t.is_planned()
    assert t.pair == weth_usdc
    assert t.planned_price == pytest.approx(1664.99)
    assert t.planned_quantity == pytest.approx(Decimal("-0.06449101796407184783443966580307460390031337738037109375"))
    assert t.get_planned_value() == pytest.approx(107.3768999999999)  # 157 - 107 = 50
    assert isinstance(t.position_size_risk, SizeRisk)

    # Buy comes next,
    # buy Aave, enter max 40 USD
    t = trades[1]
    assert t.is_buy()
    assert t.is_planned()
    assert t.planned_price == pytest.approx(100.29999999999998)
    assert t.get_planned_value() == pytest.approx(50)
    assert isinstance(t.position_size_risk, SizeRisk)