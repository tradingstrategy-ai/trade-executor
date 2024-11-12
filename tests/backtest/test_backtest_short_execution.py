"""Simulated trade execution tests for short positions."""

import datetime
import random
from decimal import Decimal
from typing import List

import pytest
from hexbytes import HexBytes

from tradeexecutor.backtest.backtest_execution import BacktestExecution
from tradeexecutor.backtest.backtest_pricing import BacktestPricing
from tradeexecutor.backtest.backtest_routing import BacktestRoutingModel, BacktestRoutingState
from tradeexecutor.backtest.backtest_sync import BacktestSyncModel
from tradeexecutor.backtest.simulated_wallet import SimulatedWallet
from tradeexecutor.state.state import State, TradeType
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.blockhain_transaction import BlockchainTransaction
from tradeexecutor.state.reserve import ReservePosition
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.strategy.alpha_model import AlphaModel
from tradeexecutor.strategy.interest import update_leveraged_position_interest, estimate_interest
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


@pytest.fixture(scope="module")
def mock_exchange() -> Exchange:
    """Mock some assets"""
    mock_exchange = generate_exchange(
        exchange_id=random.randint(1, 1000),
        chain_id=ChainId.ethereum,
        address=generate_random_ethereum_address())
    return mock_exchange


@pytest.fixture(scope="module")
def usdc() -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(ChainId.ethereum.value, "0x0", "USDC", 6)


@pytest.fixture(scope="module")
def weth() -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(ChainId.ethereum.value, "0x1", "WETH", 18)


@pytest.fixture(scope="module")
def aave() -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(ChainId.ethereum.value, "0x3", "AAVE", 18)


@pytest.fixture(scope="module")
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


@pytest.fixture(scope="module")
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


@pytest.fixture(scope="module")
def start_timestamp(usdc, weth) -> datetime.datetime:
    """Timestamp of action started"""
    return datetime.datetime(2022, 1, 1, tzinfo=None)


@pytest.fixture(scope="module")
def strategy_universe(
    mock_exchange,
    usdc: AssetIdentifier,
    weth: AssetIdentifier,
    aave: AssetIdentifier,
    weth_usdc,
    aave_usdc,
    start_timestamp
) -> TradingStrategyUniverse:
    """Create a trading universe for tests.

    - WETH/USD and AAVE/USD pairs with fixed prices

    - Over several months
    """

    start_at = start_timestamp
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
    aave_reserve = generate_lending_reserve(aave, mock_chain_id, 3)

    # Set super high interest rates to make a difference
    _, lending_candle_universe = generate_lending_universe(
        time_bucket,
        start_at,
        end_at,
        reserves=[usdc_reserve, weth_reserve, aave_reserve],
        aprs={
            "supply": 250,
            "variable": 1000,
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
def routing_model(strategy_universe) -> BacktestRoutingModel:
    return generate_simple_routing_model(strategy_universe)


@pytest.fixture()
def pricing_model(routing_model, strategy_universe) -> BacktestPricing:
    return BacktestPricing(strategy_universe.data_universe.candles, routing_model)


@pytest.fixture()
def wallet(usdc, weth) -> SimulatedWallet:
    """Dummy blockchain accounting to see we do not try to use tokens we do not have."""
    wallet = SimulatedWallet()
    wallet.set_balance(usdc, Decimal(10_000))
    return wallet


@pytest.fixture()
def execution_model(wallet) -> BacktestExecution:
    """Simulate trades using backtest execution model."""
    execution_model = BacktestExecution(wallet)
    return execution_model


@pytest.fixture()
def sync_model(wallet) -> BacktestSyncModel:
    """Read wallet balances back to the backtesting state."""
    sync_model = BacktestSyncModel(wallet)
    return sync_model


@pytest.fixture()
def state(
    usdc: AssetIdentifier,
    wallet: SimulatedWallet,
    start_timestamp,
) -> State:
    """Create a starting state."""
    state = State()
    state.update_reserves([ReservePosition(usdc, wallet.get_balance(usdc.address), start_timestamp, 1.0, start_timestamp)])
    return state


def test_estimate_interest():
    start = datetime.datetime(2020, 1, 1)
    end = datetime.datetime(2021, 1, 1)
    new_amount = estimate_interest(
        start,
        end,
        Decimal(100.0),
        1.02,
    )
    # Financial year = 360 days instead of 365
    assert new_amount == pytest.approx(Decimal("102.0055340350080275158006771"))


def test_open_and_close_one_short(
    state: State,
    strategy_universe: TradingStrategyUniverse,
    pricing_model,
    weth_usdc: TradingPairIdentifier,
    aave_usdc: TradingPairIdentifier,
    weth: AssetIdentifier,
    usdc: AssetIdentifier,
    start_timestamp,
    execution_model: BacktestExecution,
    routing_model: BacktestRoutingModel,
    wallet: SimulatedWallet,
):
    """Open and close one short position."""

    portfolio = state.portfolio
    trades = []
    simulated_time = start_timestamp + datetime.timedelta(days=1)

    position_manager = PositionManager(
        simulated_time,
        strategy_universe,
        state,
        pricing_model,
    )

    # Open WETH/USDC short
    trades += position_manager.open_short(
        weth_usdc,
        500.0,
    )

    execution_model.execute_trades(
        simulated_time,
        state,
        trades,
        routing_model,
        BacktestRoutingState(strategy_universe.data_universe.pairs, wallet),
    )

    assert any(t.is_success() for t in trades)
    assert len(state.portfolio.open_positions) == 1

    # Next day, close shots
    trades = []
    simulated_time += datetime.timedelta(days=1)
    position_manager = PositionManager(
        simulated_time,
        strategy_universe,
        state,
        pricing_model,
    )

    eth_shorting_pair = strategy_universe.get_shorting_pair(weth_usdc)
    eth_short_position = position_manager.get_current_position_for_pair(eth_shorting_pair)

    trades += position_manager.close_short(eth_short_position)

    execution_model.execute_trades(
        simulated_time,
        state,
        trades,
        routing_model,
        BacktestRoutingState(strategy_universe.data_universe.pairs, wallet),
    )

    assert any(t.is_success() for t in trades)
    assert len(state.portfolio.closed_positions) == 1

    assert portfolio.get_cash() == pytest.approx(9996.995486459378)  # Lost on fees
    assert portfolio.get_net_asset_value() == pytest.approx(9996.995486459378)  # All in cash

    # Check that we have cleared the wallet, including dust
    assert wallet.get_balance(weth) == 0
    assert wallet.get_balance(eth_shorting_pair.base) == 0
    assert wallet.get_balance(eth_shorting_pair.quote) == 0
    assert wallet.get_balance(usdc) == pytest.approx(Decimal(9996.995486459378))


def test_open_and_close_two_shorts(
    state: State,
    strategy_universe: TradingStrategyUniverse,
    pricing_model,
    usdc: AssetIdentifier,
    weth_usdc: TradingPairIdentifier,
    aave_usdc: TradingPairIdentifier,
    weth: AssetIdentifier,
    aave: AssetIdentifier,
    start_timestamp,
    execution_model: BacktestExecution,
    routing_model: BacktestRoutingModel,
    wallet: SimulatedWallet,
):
    """See that we can open and close two shorts sharing the same collateral simultaneously."""

    # Open WETH/USDC and AAVE/USDC shorts
    portfolio = state.portfolio
    trades = []
    simulated_time = start_timestamp + datetime.timedelta(days=1)

    position_manager = PositionManager(
        simulated_time,
        strategy_universe,
        state,
        pricing_model,
    )

    trades += position_manager.open_short(
        weth_usdc,
        500.0,
    )

    trades += position_manager.open_short(
        aave_usdc,
        300.0,
    )

    execution_model.execute_trades(
        simulated_time,
        state,
        trades,
        routing_model,
        BacktestRoutingState(strategy_universe.data_universe.pairs, wallet),
    )

    assert any(t.is_success() for t in trades)
    assert len(portfolio.open_positions) == 2

    # Next day, close shots
    trades = []
    simulated_time += datetime.timedelta(days=1)
    position_manager = PositionManager(
        simulated_time,
        strategy_universe,
        state,
        pricing_model,
    )

    # Close AAVE short
    aave_shorting_pair = strategy_universe.get_shorting_pair(aave_usdc)
    aave_short_position = position_manager.get_current_position_for_pair(aave_shorting_pair)
    trades += position_manager.close_short(aave_short_position)

    # Close ETH short
    eth_shorting_pair = strategy_universe.get_shorting_pair(weth_usdc)
    eth_short_position = position_manager.get_current_position_for_pair(eth_shorting_pair)
    trades += position_manager.close_short(eth_short_position)

    execution_model.execute_trades(
        simulated_time,
        state,
        trades,
        routing_model,
        BacktestRoutingState(strategy_universe.data_universe.pairs, wallet),
    )

    assert any(t.is_success() for t in trades)
    assert len(portfolio.closed_positions) == 2

    assert portfolio.get_cash() == pytest.approx(9995.192778335006)  # Lost on fees
    assert portfolio.get_net_asset_value() == pytest.approx(9995.192778335006)  # All in cash

    wallet = execution_model.wallet

    # Check that we have cleared the wallet, including dust
    assert wallet.get_balance(weth) == 0
    assert wallet.get_balance(aave) == 0
    assert wallet.get_balance(aave_shorting_pair.base) == 0
    assert wallet.get_balance(aave_shorting_pair.quote) == 0
    assert wallet.get_balance(eth_shorting_pair.base) == 0
    assert wallet.get_balance(eth_shorting_pair.quote) == 0
    assert wallet.get_balance(usdc) == pytest.approx(Decimal(9995.192778335006))


def test_open_and_close_one_short_with_interest(
    state: State,
    strategy_universe: TradingStrategyUniverse,
    pricing_model,
    weth_usdc: TradingPairIdentifier,
    aave_usdc: TradingPairIdentifier,
    weth: AssetIdentifier,
    usdc: AssetIdentifier,
    start_timestamp,
    execution_model: BacktestExecution,
    routing_model: BacktestRoutingModel,
    sync_model: BacktestSyncModel,
    wallet: SimulatedWallet,
):
    """Open and close one short position w/interest.

    - See that we correctly claim accrued interest.

    - Modified to include interest from :py:func:`test_open_and_close_one_short`
    """


    portfolio = state.portfolio
    trades = []
    simulated_time = start_timestamp + datetime.timedelta(days=1)
    wallet = execution_model.wallet

    position_manager = PositionManager(
        simulated_time,
        strategy_universe,
        state,
        pricing_model,
    )

    # Open WETH/USDC short
    trades += position_manager.open_short(
        weth_usdc,
        500.0,
    )

    execution_model.execute_trades(
        simulated_time,
        state,
        trades,
        routing_model,
        BacktestRoutingState(strategy_universe.data_universe.pairs, wallet),
    )

    assert any(t.is_success() for t in trades)
    assert len(state.portfolio.open_positions) == 1

    # Simulate accrued interest
    # by having the on-chain balances to update "by itself"
    eth_shorting_pair = strategy_universe.get_shorting_pair(weth_usdc)
    eth_short_position = position_manager.get_current_position_for_pair(eth_shorting_pair)

    # Next day, gain interest, close shorts
    trades = []
    simulated_time += datetime.timedelta(days=1)

    # Read on-chain token balances back to the state
    balance_updates = sync_model.sync_interests(
        simulated_time,
        state,
        strategy_universe,
        pricing_model,
    )
    assert len(balance_updates) == 2
    assert eth_short_position.loan.get_collateral_interest() == pytest.approx(6.839041095890411)
    assert eth_short_position.loan.get_borrow_interest() == pytest.approx(13.657534246575342)

    assert eth_short_position.get_loan_based_nav(include_interest=False) == pytest.approx(500)
    assert eth_short_position.get_loan_based_nav(include_interest=True) == pytest.approx(493.18150684931504)
    assert portfolio.get_cash() == pytest.approx(9500.0)

    # assert eth_short_position.get_accrued_interest() == pytest.approx(0)

    # Recreate position manager with changed timestamp
    position_manager = PositionManager(
        simulated_time,
        strategy_universe,
        state,
        pricing_model,
    )

    trades += position_manager.close_short(eth_short_position)

    execution_model.execute_trades(
        simulated_time,
        state,
        trades,
        routing_model,
        BacktestRoutingState(strategy_universe.data_universe.pairs, wallet),
    )

    assert any(t.is_success() for t in trades)
    assert len(state.portfolio.closed_positions) == 1

    #
    # TODO: %0.30 fee to close the position is covered from interest?
    #

    # Check that we have cleared the wallet, including dust
    assert wallet.get_balance(weth) == 0
    assert wallet.get_balance(eth_shorting_pair.base) == 0
    assert wallet.get_balance(eth_shorting_pair.quote) == 0
    assert wallet.get_balance(usdc) == pytest.approx(Decimal(9990.094677869224110624516323))

    assert portfolio.get_cash() == pytest.approx(9990.094677869224110624516323)
    assert portfolio.get_net_asset_value() == pytest.approx(9990.094677869224110624516323)


def test_open_and_close_two_shorts_with_interest(
    state: State,
    strategy_universe: TradingStrategyUniverse,
    pricing_model,
    weth_usdc: TradingPairIdentifier,
    aave_usdc: TradingPairIdentifier,
    weth: AssetIdentifier,
    usdc: AssetIdentifier,
    start_timestamp,
    execution_model: BacktestExecution,
    routing_model: BacktestRoutingModel,
    sync_model: BacktestSyncModel,
    wallet: SimulatedWallet,
):
    """Open and close one short position w/interest.

    - See that the shared collateral accrued interest is correctly distributed across positions.

    - See that we correctly claim accrued interest.

    - Modified to include interest from :py:func:`test_open_and_close_one_short`
    """

    portfolio = state.portfolio
    trades = []
    simulated_time = start_timestamp + datetime.timedelta(days=1)
    wallet = execution_model.wallet

    position_manager = PositionManager(
        simulated_time,
        strategy_universe,
        state,
        pricing_model,
    )

    # Open WETH/USDC short
    trades += position_manager.open_short(
        weth_usdc,
        500.0,
    )

    # Open WETH/USDC short
    trades += position_manager.open_short(
        aave_usdc,
        250.0,
    )

    execution_model.execute_trades(
        simulated_time,
        state,
        trades,
        routing_model,
        BacktestRoutingState(strategy_universe.data_universe.pairs, wallet),
    )

    assert any(t.is_success() for t in trades)
    assert len(state.portfolio.open_positions) == 2

    # Simulate accrued interest
    # by having the on-chain balances to update "by itself"
    eth_shorting_pair = strategy_universe.get_shorting_pair(weth_usdc)
    aave_shorting_pair = strategy_universe.get_shorting_pair(aave_usdc)
    eth_short_position = position_manager.get_current_position_for_pair(eth_shorting_pair)
    aave_short_position = position_manager.get_current_position_for_pair(aave_shorting_pair)

    # Check that we have a interest sync checkmark
    assert state.sync.interest.last_sync_at == simulated_time
    assert state.sync.interest.last_distribution is None

    # Next day, gain interest, close shorts
    trades = []
    simulated_time += datetime.timedelta(days=1)

    # Read on-chain token balances back to the state
    balance_updates = sync_model.sync_interests(
        simulated_time,
        state,
        strategy_universe,
        pricing_model,
    )
    assert len(balance_updates) == 4

    assert state.sync.interest.last_sync_at == simulated_time
    assert state.sync.interest.last_distribution is not None

    vweth = eth_short_position.pair.base
    ausdc = eth_short_position.pair.quote
    vaave = aave_shorting_pair.base

    interest_distribution = state.sync.interest.last_distribution
    assert interest_distribution.duration == datetime.timedelta(days=1)
    assert interest_distribution.assets == {ausdc, vweth, vaave}
    assert interest_distribution.asset_interest_data[vweth.get_identifier()].effective_rate == pytest.approx(10)  # Around ~1000% interest, minus few days + rounding errors
    assert interest_distribution.asset_interest_data[ausdc.get_identifier()].effective_rate == pytest.approx(2.5)  # Around ~250% interest, minus few days + rounding errors

    assert eth_short_position.loan.get_collateral_interest() == pytest.approx(6.839041095890411)
    assert eth_short_position.loan.get_borrow_interest() == pytest.approx(13.657534246575342)
    assert aave_short_position.loan.get_collateral_interest() == pytest.approx(3.4195205479452055)  # Should be 50% of ETH

    # Recreate position manager with changed timestamp
    position_manager = PositionManager(
        simulated_time,
        strategy_universe,
        state,
        pricing_model,
    )

    trades += position_manager.close_all()

    execution_model.execute_trades(
        simulated_time,
        state,
        trades,
        routing_model,
        BacktestRoutingState(strategy_universe.data_universe.pairs, wallet),
    )

    assert len(trades) == 2
    assert any(t.is_success() for t in trades)
    assert len(state.portfolio.closed_positions) == 2

    # Check that we have cleared the wallet, including dust
    assert wallet.get_balance(weth) == 0
    assert wallet.get_balance(eth_shorting_pair.base) == 0
    assert wallet.get_balance(eth_shorting_pair.quote) == 0
    assert wallet.get_balance(aave_shorting_pair.base) == 0
    assert wallet.get_balance(aave_shorting_pair.quote) == 0
    assert wallet.get_balance(usdc) == pytest.approx(Decimal(9985.142016803836165936774485))

    assert portfolio.get_cash() == pytest.approx(9985.142016803836165936774485)
    assert portfolio.get_net_asset_value() == pytest.approx(9985.142016803836165936774485)


def test_open_and_reduce_one_short_with_interest(
    state: State,
    strategy_universe: TradingStrategyUniverse,
    pricing_model,
    weth_usdc: TradingPairIdentifier,
    aave_usdc: TradingPairIdentifier,
    weth: AssetIdentifier,
    usdc: AssetIdentifier,
    start_timestamp,
    execution_model: BacktestExecution,
    routing_model: BacktestRoutingModel,
    sync_model: BacktestSyncModel,
    wallet: SimulatedWallet,
):
    """Open and reduce a short position w/interest.
    """


    portfolio = state.portfolio
    trades = []
    simulated_time = start_timestamp + datetime.timedelta(days=1)
    wallet = execution_model.wallet

    position_manager = PositionManager(
        simulated_time,
        strategy_universe,
        state,
        pricing_model,
    )

    # Open WETH/USDC short
    trades += position_manager.open_short(
        weth_usdc,
        500.0,
    )

    execution_model.execute_trades(
        simulated_time,
        state,
        trades,
        routing_model,
        BacktestRoutingState(strategy_universe.data_universe.pairs, wallet),
    )

    assert any(t.is_success() for t in trades)
    assert len(state.portfolio.open_positions) == 1

    # Simulate accrued interest
    # by having the on-chain balances to update "by itself"
    eth_shorting_pair = strategy_universe.get_shorting_pair(weth_usdc)
    eth_short_position = position_manager.get_current_position_for_pair(eth_shorting_pair)

    # Next day, gain interest, close shorts
    trades: List[TradeExecution] = []
    simulated_time += datetime.timedelta(days=1)

    # Read on-chain token balances back to the state
    balance_updates = sync_model.sync_interests(
        simulated_time,
        state,
        strategy_universe,
        pricing_model,
    )
    assert len(balance_updates) == 2
    assert eth_short_position.loan.get_collateral_interest() == pytest.approx(6.839041095890411)
    assert eth_short_position.loan.get_borrow_interest() == pytest.approx(13.657534246575342)
    assert eth_short_position.loan.get_leverage() == pytest.approx(1.0107840482192085)
    assert eth_short_position.loan.get_net_asset_value() == pytest.approx(493.18150684931504)

    assert eth_short_position.get_loan_based_nav(include_interest=False) == pytest.approx(500)
    assert eth_short_position.get_loan_based_nav(include_interest=True) == pytest.approx(493.18150684931504)
    assert portfolio.get_cash() == pytest.approx(9500.0)

    # assert eth_short_position.get_accrued_interest() == pytest.approx(0)

    # Recreate position manager with changed timestamp
    position_manager = PositionManager(
        simulated_time,
        strategy_universe,
        state,
        pricing_model,
    )

    # Reduce the short 50%
    trades += position_manager.adjust_short(
        eth_short_position,
        250,
    )

    # vWETH down
    t = trades[0]
    # assert t.planned_loan_update.last_token_amount =
    assert t.planned_loan_update.get_net_asset_value() == pytest.approx(243.9395948854795)
    assert t.planned_loan_update.get_leverage() == pytest.approx(1.0327881545302768)
    assert t.planned_loan_update.collateral.quantity == pytest.approx(Decimal(502.6960120548021326847276668))
    assert t.planned_loan_update.borrowed.quantity == pytest.approx(Decimal(0.1513149772783246303501363274))
    assert t.planned_loan_update.get_borrowed_quantity() == pytest.approx(Decimal(0.1595177498154421760805932217))  # Includes 0.1 ETH interest
    assert t.planned_loan_update.get_collateral_quantity() == pytest.approx(Decimal(509.5350531506925436434179064))

    execution_model.execute_trades(
        simulated_time,
        state,
        trades,
        routing_model,
        BacktestRoutingState(strategy_universe.data_universe.pairs, wallet),
    )

    assert any(t.is_success() for t in trades)
    assert len(state.portfolio.open_positions) == 1
    assert len(state.portfolio.closed_positions) == 0

    # Check that we have cleared the wallet, including dust
    assert wallet.get_balance(weth) == 0
    assert wallet.get_balance(eth_shorting_pair.base) == pytest.approx(Decimal(0.1595177498154421760805932217))
    assert wallet.get_balance(eth_shorting_pair.quote) == pytest.approx(Decimal(509.5350531506925436434179064))
    assert wallet.get_balance(usdc) == pytest.approx(Decimal(9743.181506849315042018133681))

    assert portfolio.get_cash() == pytest.approx(9743.181506849315042018133681)
    assert portfolio.get_net_asset_value() == pytest.approx(9987.121101734794)


def test_open_and_increase_one_short_with_interest(
    state: State,
    strategy_universe: TradingStrategyUniverse,
    pricing_model,
    weth_usdc: TradingPairIdentifier,
    aave_usdc: TradingPairIdentifier,
    weth: AssetIdentifier,
    usdc: AssetIdentifier,
    start_timestamp,
    execution_model: BacktestExecution,
    routing_model: BacktestRoutingModel,
    sync_model: BacktestSyncModel,
    wallet: SimulatedWallet,
):
    """Open and increase a short position w/interest.
    """

    portfolio = state.portfolio
    trades = []
    simulated_time = start_timestamp + datetime.timedelta(days=1)
    wallet = execution_model.wallet

    position_manager = PositionManager(
        simulated_time,
        strategy_universe,
        state,
        pricing_model,
    )

    # Open WETH/USDC short
    trades += position_manager.open_short(
        weth_usdc,
        500.0,
    )

    execution_model.execute_trades(
        simulated_time,
        state,
        trades,
        routing_model,
        BacktestRoutingState(strategy_universe.data_universe.pairs, wallet),
    )

    assert any(t.is_success() for t in trades)
    assert len(state.portfolio.open_positions) == 1

    # Simulate accrued interest
    # by having the on-chain balances to update "by itself"
    eth_shorting_pair = strategy_universe.get_shorting_pair(weth_usdc)
    eth_short_position = position_manager.get_current_position_for_pair(eth_shorting_pair)

    # Next day, gain interest, close shorts
    trades: List[TradeExecution] = []
    simulated_time += datetime.timedelta(days=1)

    # Read on-chain token balances back to the state
    balance_updates = sync_model.sync_interests(
        simulated_time,
        state,
        strategy_universe,
        pricing_model,
    )
    assert len(balance_updates) == 2
    assert portfolio.get_cash() == pytest.approx(9500.0)
    assert eth_short_position.loan.get_leverage() == pytest.approx(1.0107840482192085)
    assert eth_short_position.loan.get_net_asset_value() == pytest.approx(493.18150684931504)
    assert eth_short_position.loan.get_collateral_quantity() == pytest.approx(Decimal(1005.339041095890410927465217))

    # assert eth_short_position.get_accrued_interest() == pytest.approx(0)

    # Recreate position manager with changed timestamp
    position_manager = PositionManager(
        simulated_time,
        strategy_universe,
        state,
        pricing_model,
    )

    # Increase the short $500 -> $750
    trades += position_manager.adjust_short(
        eth_short_position,
        750,
    )

    # vWETH down
    t = trades[0]
    # assert t.planned_loan_update.last_token_amount =
    assert t.planned_reserve == pytest.approx(Decimal(256.8184931506849579818663187325000762939453125))
    assert t.planned_loan_update.get_net_asset_value() == pytest.approx(743.1815068493153)
    assert t.planned_loan_update.get_leverage() == pytest.approx(1.0200577236889097)
    assert t.planned_loan_update.collateral.quantity == pytest.approx(Decimal(1508.088036164406398054183000))
    assert t.planned_loan_update.borrowed.quantity == pytest.approx(Decimal(0.4553108644282586795919334237))
    assert t.planned_loan_update.get_borrowed_quantity() == pytest.approx(Decimal(0.4635136369653762253223903180))  # Includes 0.01 ETH interest
    assert t.planned_loan_update.get_collateral_quantity() == pytest.approx(Decimal(1514.927077260296809012873240))  # Includes $5 interest
    assert t.planned_loan_update.get_leverage() == pytest.approx(1.0200577236889097)

    execution_model.execute_trades(
        simulated_time,
        state,
        trades,
        routing_model,
        BacktestRoutingState(strategy_universe.data_universe.pairs, wallet),
    )

    assert any(t.is_success() for t in trades)
    assert len(state.portfolio.open_positions) == 1
    assert len(state.portfolio.closed_positions) == 0

    # Check that we have cleared the wallet, including dust
    assert wallet.get_balance(weth) == 0
    assert wallet.get_balance(eth_shorting_pair.base) == pytest.approx(Decimal(0.4635136369653762253223903180))
    assert wallet.get_balance(eth_shorting_pair.quote) == pytest.approx(Decimal(1514.927077260296809012873240))
    assert wallet.get_balance(usdc) == pytest.approx(Decimal(9243.181506849315042018133681))

    assert portfolio.get_cash() == pytest.approx(9243.181506849314)
    assert portfolio.get_net_asset_value() == pytest.approx(9986.36301369863)