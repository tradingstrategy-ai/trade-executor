"""Test stop loss/position trigger on backtest data.

- Generate random hourly candles

- Upsample hourly candles to daily candles

- Hourly candles are our "real time" stop loss data

- Run test for stop loss triggers using a made up strategy
  to trigger stop losses

- Run test for take profit triggers using a made up strategy
  to trigger stop losses

"""
import datetime
import logging
import os
from pathlib import Path
from typing import List, Dict

import pytest

import pandas as pd

from tradeexecutor.analysis.stop_loss import analyse_stop_losses, analyse_trigger_updates
from tradeexecutor.analysis.trade_analyser import build_trade_analysis
from tradeexecutor.backtest.backtest_routing import BacktestRoutingModel
from tradeexecutor.backtest.backtest_runner import run_backtest, setup_backtest_for_universe, run_backtest_inline
from tradeexecutor.cli.log import setup_pytest_logging
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.execution_model import AutoClosingOrderUnsupported
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.reserve_currency import ReserveCurrency
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, create_pair_universe_from_code
from tradeexecutor.testing.synthetic_ethereum_data import generate_random_ethereum_address
from tradeexecutor.testing.synthetic_exchange_data import generate_exchange, generate_simple_routing_model
from tradeexecutor.testing.synthetic_price_data import generate_ohlcv_candles
from tradingstrategy.candle import GroupedCandleUniverse, is_candle_green
from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import Exchange, ExchangeUniverse
from tradingstrategy.pair import PandasPairUniverse, DEXPair
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe
from tradingstrategy.utils.groupeduniverse import resample_candles


def stop_loss_decide_trades_factory(stop_loss_pct=None):
    """Factory allows easily test the strategy with different stop loss parameters."""

    def stop_loss_decide_trades(
            timestamp: pd.Timestamp,
            universe: Universe,
            state: State,
            pricing_model: PricingModel,
            cycle_debug_data: Dict) -> List[TradeExecution]:
        """Keep triggering position stop losses.

        Trading logic

        - On 2 green candles open a position

        - Use 5% stop loss

        - Close the position if we get 4 green candles
        """

        candles: pd.DataFrame = universe.candles.get_single_pair_data(timestamp, sample_count=4, raise_on_not_enough_data=False)

        # The pair we are trading
        pair = universe.pairs.get_single()

        # How much cash we have in the hand
        cash = state.portfolio.get_cash()

        position_manager = PositionManager(timestamp, universe, state, pricing_model)

        trades = []

        if position_manager.is_any_open():
            # Close position on 4 green candles thru rebalance (take profit)
            tail = candles.tail(4)
            if len(tail) >= 4 and all(is_candle_green(c) for idx, c in tail.iterrows()):
                trades += position_manager.close_all()
        else:
            # Open new position if 2 green daily candles
            tail = candles.tail(2)
            if len(tail) >= 2:
                last_candle = tail.iloc[-1]
                second_last_candle = tail.iloc[-2]
                if is_candle_green(last_candle) and is_candle_green(second_last_candle):
                    if stop_loss_pct:
                        # Stop loss activated
                        trades += position_manager.open_spot(pair, cash * 0.1, stop_loss_pct=stop_loss_pct)
                    else:
                        # Stop loss inactive
                        trades += position_manager.open_spot(pair, cash * 0.1)

        return trades

    return stop_loss_decide_trades


def take_profit_decide_trades_factory(take_profit_pct=None):
    """Factory allows easily test the strategy with different tak profit parameters."""

    def stop_loss_decide_trades(
            timestamp: pd.Timestamp,
            universe: Universe,
            state: State,
            pricing_model: PricingModel,
            cycle_debug_data: Dict) -> List[TradeExecution]:
        """Keep triggering position stop losses.

        Trading logic

        - On 2 green candles open a position

        - Use 5% stop loss

        - Close the position if we get 4 green candles
        """

        candles: pd.DataFrame = universe.candles.get_single_pair_data(timestamp, sample_count=4, raise_on_not_enough_data=False)

        # The pair we are trading
        pair = universe.pairs.get_single()

        # How much cash we have in the hand
        cash = state.portfolio.get_cash()

        position_manager = PositionManager(timestamp, universe, state, pricing_model)

        trades = []

        if position_manager.is_any_open():
            # Close position on 4 green candles thru rebalance (take profit)
            tail = candles.tail(4)
            if len(tail) >= 4 and all(is_candle_green(c) for idx, c in tail.iterrows()):
                trades += position_manager.close_all()
        else:
            # Open new position if 2 green daily candles
            tail = candles.tail(2)
            if len(tail) >= 2:
                last_candle = tail.iloc[-1]
                second_last_candle = tail.iloc[-2]
                if is_candle_green(last_candle) and is_candle_green(second_last_candle):
                    if take_profit_pct:
                        # Stop loss activated
                        price = last_candle["close"]
                        trades += position_manager.open_spot(pair, cash * 0.1, take_profit_pct=take_profit_pct)
                    else:
                        # Stop loss inactive
                        trades += position_manager.open_spot(pair, cash * 0.1)

        return trades

    return stop_loss_decide_trades


def trailing_stop_loss_decide_trades_factory(trailing_stop_loss_pct=None):
    """Factory allows easily test the strategy with different traikign stop loss parameters."""

    def stop_loss_decide_trades(
            timestamp: pd.Timestamp,
            universe: Universe,
            state: State,
            pricing_model: PricingModel,
            cycle_debug_data: Dict) -> List[TradeExecution]:
        """Keep triggering position stop losses.

        Trading logic

        - On 2 green candles open a position

        - Use 5% stop loss

        - Close the position if we get 4 green candles
        """

        candles: pd.DataFrame = universe.candles.get_single_pair_data(timestamp, sample_count=4, raise_on_not_enough_data=False)

        # The pair we are trading
        pair = universe.pairs.get_single()

        # How much cash we have in the hand
        cash = state.portfolio.get_cash()

        position_manager = PositionManager(timestamp, universe, state, pricing_model)

        trades = []

        if position_manager.is_any_open():
            # Close position on 4 green candles thru rebalance (take profit)
            tail = candles.tail(4)
            if len(tail) >= 4 and all(is_candle_green(c) for idx, c in tail.iterrows()):
                trades += position_manager.close_all()
        else:
            # Open new position if 2 green daily candles
            tail = candles.tail(2)
            if len(tail) >= 2:
                last_candle = tail.iloc[-1]
                second_last_candle = tail.iloc[-2]
                if is_candle_green(last_candle) and is_candle_green(second_last_candle):
                    trades += position_manager.open_spot(pair, cash * 0.1, trailing_stop_loss_pct=trailing_stop_loss_pct)

        return trades

    return stop_loss_decide_trades


def stop_loss_usd_decide_trades_factory(stop_loss_pct=None):
    """Factory allows easily test the strategy with different stop loss parameters."""

    def stop_loss_decide_trades(
            timestamp: pd.Timestamp,
            universe: Universe,
            state: State,
            pricing_model: PricingModel,
            cycle_debug_data: Dict) -> List[TradeExecution]:
        """Keep triggering position stop losses.

        Trading logic

        - On 2 green candles open a position

        - Use 5% stop loss

        - Close the position if we get 4 green candles
        """

        candles: pd.DataFrame = universe.candles.get_single_pair_data(timestamp, sample_count=4, raise_on_not_enough_data=False)

        # The pair we are trading
        pair = universe.pairs.get_single()

        # How much cash we have in the hand
        cash = state.portfolio.get_cash()

        position_manager = PositionManager(timestamp, universe, state, pricing_model)

        trades = []

        if position_manager.is_any_open():
            # Close position on 4 green candles thru rebalance (take profit)
            tail = candles.tail(4)
            if len(tail) >= 4 and all(is_candle_green(c) for idx, c in tail.iterrows()):
                trades += position_manager.close_all()
        else:
            # Open new position if 2 green daily candles
            tail = candles.tail(2)
            if len(tail) >= 2:
                last_candle = tail.iloc[-1]
                second_last_candle = tail.iloc[-2]
                if is_candle_green(last_candle) and is_candle_green(second_last_candle):
                    if stop_loss_pct:
                        # Stop loss activated
                        trades += position_manager.open_spot(pair, cash * 0.1, stop_loss_usd=pricing_model.get_mid_price() * stop_loss_pct)
                    else:
                        # Stop loss inactive
                        trades += position_manager.open_spot(pair, cash * 0.1)

        return trades

    return stop_loss_decide_trades


@pytest.fixture(scope="module")
def logger(request):
    """Setup test logger."""
    return setup_pytest_logging(request, mute_requests=False)


@pytest.fixture
def mock_chain_id() -> ChainId:
    """Mock a chai id."""
    return ChainId.ethereum


@pytest.fixture
def mock_exchange(mock_chain_id) -> Exchange:
    """Mock an exchange."""
    return generate_exchange(exchange_id=1, chain_id=mock_chain_id, address=generate_random_ethereum_address())


@pytest.fixture
def usdc() -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(ChainId.ethereum.value, generate_random_ethereum_address(), "USDC", 6, 1)


@pytest.fixture
def weth() -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(ChainId.ethereum.value, generate_random_ethereum_address(), "WETH", 18, 2)


@pytest.fixture
def weth_usdc(mock_exchange, usdc, weth) -> TradingPairIdentifier:
    """Mock some assets"""
    return TradingPairIdentifier(
        weth,
        usdc,
        generate_random_ethereum_address(),
        mock_exchange.address,
        internal_id=555,
        internal_exchange_id=mock_exchange.exchange_id)


@pytest.fixture()
def strategy_path() -> Path:
    """Where do we load our strategy file."""
    return Path(os.path.join(os.path.dirname(__file__), "../..", "strategies", "test_only", "pandas_buy_every_second_day.py"))


@pytest.fixture()
def synthetic_universe(mock_chain_id, mock_exchange, weth_usdc) -> TradingStrategyUniverse:
    """Generate synthetic trading data universe for a single trading pair.

    - Single mock exchange

    - Single mock trading pair

    - Random hourly candles as stop loss signal

    - Random daily candles samples from the above

    - No liquidity data available
    """

    start_date = datetime.datetime(2021, 6, 1)
    end_date = datetime.datetime(2022, 1, 1)

    stop_loss_time_bucket = TimeBucket.h1
    time_bucket = TimeBucket.d1

    pair_universe = create_pair_universe_from_code(mock_chain_id, [weth_usdc])

    # Generate candles for pair_id = 1
    stop_loss_candles = generate_ohlcv_candles(
        stop_loss_time_bucket,
        start_date,
        end_date,
        pair_id=weth_usdc.internal_id)

    candles = resample_candles(stop_loss_candles, pd.Timedelta(days=1))

    assert len(candles) == len(stop_loss_candles) / 24

    candle_universe = GroupedCandleUniverse.create_from_single_pair_dataframe(candles)

    universe = Universe(
        time_bucket=time_bucket,
        chains={mock_chain_id},
        exchanges={mock_exchange},
        pairs=pair_universe,
        candles=candle_universe,
        liquidity=None
    )

    return TradingStrategyUniverse(
        data_universe=universe,
        reserve_assets=[weth_usdc.quote],
        backtest_stop_loss_candles=GroupedCandleUniverse(stop_loss_candles),
        backtest_stop_loss_time_bucket=stop_loss_time_bucket,
    )


@pytest.fixture()
def routing_model(synthetic_universe) -> BacktestRoutingModel:
    return generate_simple_routing_model(synthetic_universe)


def test_synthetic_data_backtest_stop_loss(
        logger: logging.Logger,
        strategy_path: Path,
        synthetic_universe: TradingStrategyUniverse,
        routing_model: BacktestRoutingModel,
    ):
    """Run the strategy that triggers stop losses."""

    assert synthetic_universe.has_stop_loss_data()

    start_at, end_at = synthetic_universe.data_universe.candles.get_timestamp_range()

    routing_model = generate_simple_routing_model(synthetic_universe)

    stop_loss_decide_trades = stop_loss_decide_trades_factory(stop_loss_pct=None)

    # Run the test
    state, universe, debug_dump = run_backtest_inline(
        name="No stop loss",
        start_at=start_at.to_pydatetime(),
        end_at=end_at.to_pydatetime(),
        client=None,  # None of downloads needed, because we are using synthetic data
        cycle_duration=CycleDuration.cycle_1d,  # Override to use 24h cycles despite what strategy file says
        decide_trades=stop_loss_decide_trades,
        create_trading_universe=None,
        universe=synthetic_universe,
        initial_deposit=10_000,
        reserve_currency=ReserveCurrency.busd,
        trade_routing=TradeRouting.user_supplied_routing_model,
        routing_model=routing_model,
        log_level=logging.WARNING,
        allow_missing_fees=True,
    )

    # Expect backtesting for 213 days
    assert len(debug_dump) == 215
    assert len(list(state.portfolio.get_all_positions())) == 9
    assert len(list(state.portfolio.get_all_trades())) == 17

    analysis = build_trade_analysis(state.portfolio)
    summary = analysis.calculate_summary_statistics()
    profitability = summary.realised_profit / summary.initial_cash
    assert profitability > 0

    #
    # Now run the same strategy with stop less set to 5%
    #

    stop_loss_decide_trades = stop_loss_decide_trades_factory(stop_loss_pct=0.95)
    state, universe, debug_dump = run_backtest_inline(
        name="With 95% stop loss",
        start_at=start_at.to_pydatetime(),
        end_at=end_at.to_pydatetime(),
        client=None,
        cycle_duration=CycleDuration.cycle_1d,
        decide_trades=stop_loss_decide_trades,
        create_trading_universe=None,
        universe=synthetic_universe,
        initial_deposit=10_000,
        reserve_currency=ReserveCurrency.busd,
        trade_routing=TradeRouting.user_supplied_routing_model,
        routing_model=routing_model,
        log_level=logging.WARNING,
        allow_missing_fees=True,
    )

    assert universe.backtest_stop_loss_candles
    assert universe.backtest_stop_loss_time_bucket

    assert len(universe.data_universe.candles.df) == 214
    assert len(universe.backtest_stop_loss_candles.df) == 5136

    # Expect backtesting for 213 days
    assert len(debug_dump) == 215

    # Check that all positions had stop loss set
    for p in state.portfolio.get_all_positions():
        assert p.stop_loss, f"Position did not have stop loss: {p}"

    assert len(list(state.portfolio.get_all_positions())) == 31
    assert len([p for p in state.portfolio.get_all_positions() if p.is_closed()]) == 31
    assert len(list(state.portfolio.get_all_trades())) == 62

    # We should have some stop loss trades and some trades closed for profit
    stop_loss_trades = [t for t in state.portfolio.get_all_trades() if t.is_stop_loss()]
    rebalance_trades = [t for t in state.portfolio.get_all_trades() if t.is_rebalance()]
    assert len(rebalance_trades) == 37
    assert len(stop_loss_trades) == 25

    # Check are stop loss positions unprofitable
    stop_loss_positions = [p for p in state.portfolio.get_all_positions() if p.is_stop_loss()]
    for p in stop_loss_positions:
        assert p.is_loss()


def test_synthetic_data_backtest_take_profit(
        logger: logging.Logger,
        strategy_path: Path,
        synthetic_universe: TradingStrategyUniverse,
        routing_model: BacktestRoutingModel,
    ):
    """Run the strategy that triggers take profits.

    Run the same strategy

    - Without take profit

    - With take profit 5%

    And see the results make sense.
    """
    start_at, end_at = synthetic_universe.data_universe.candles.get_timestamp_range()

    routing_model = generate_simple_routing_model(synthetic_universe)

    take_profit_trades = take_profit_decide_trades_factory(take_profit_pct=None)

    # Run the test
    state, universe, debug_dump = run_backtest_inline(
        name="No stop loss",
        start_at=start_at.to_pydatetime(),
        end_at=end_at.to_pydatetime(),
        client=None,
        cycle_duration=CycleDuration.cycle_1d,
        decide_trades=take_profit_trades,
        create_trading_universe=None,
        universe=synthetic_universe,
        initial_deposit=10_000,
        reserve_currency=ReserveCurrency.busd,
        trade_routing=TradeRouting.user_supplied_routing_model,
        routing_model=routing_model,
        log_level=logging.WARNING,
        allow_missing_fees=True,
    )

    # Expect backtesting for 213 days
    assert len(debug_dump) == 215
    assert len(list(state.portfolio.get_all_positions())) == 9
    assert len(list(state.portfolio.get_all_trades())) == 17

    analysis = build_trade_analysis(state.portfolio)
    summary = analysis.calculate_summary_statistics()
    profitability = summary.realised_profit / summary.initial_cash
    assert profitability > 0

    #
    # Now run the same strategy with stop less set to 5%
    #

    take_profit_trades = take_profit_decide_trades_factory(take_profit_pct=1.05)
    state, universe, debug_dump = run_backtest_inline(
        name="With 95% stop loss",
        start_at=start_at.to_pydatetime(),
        end_at=end_at.to_pydatetime(),
        client=None,
        cycle_duration=CycleDuration.cycle_1d,
        decide_trades=take_profit_trades,
        create_trading_universe=None,
        universe=synthetic_universe,
        initial_deposit=10_000,
        reserve_currency=ReserveCurrency.busd,
        trade_routing=TradeRouting.user_supplied_routing_model,
        routing_model=routing_model,
        log_level=logging.WARNING,
        allow_missing_fees=True,
    )

    assert universe.backtest_stop_loss_candles
    assert universe.backtest_stop_loss_time_bucket

    # Expect backtesting for 213 days
    assert len(debug_dump) == 215

    # Check that all positions had take profit set
    for p in state.portfolio.get_all_positions():
        assert p.take_profit, f"Position did not have take profit: {p}"

    assert len(list(state.portfolio.get_all_positions())) == 35
    assert len(list(state.portfolio.get_all_trades())) == 69

    # We should have some stop loss trades and some trades closed for profit
    take_profit_trades = [t for t in state.portfolio.get_all_trades() if t.is_take_profit()]
    rebalance_trades = [t for t in state.portfolio.get_all_trades() if t.is_rebalance()]
    assert len(rebalance_trades) == 37
    assert len(take_profit_trades) == 32

    # Check are stop loss positions unprofitable
    take_profit_positions = [p for p in state.portfolio.get_all_positions() if p.is_take_profit()]
    for p in take_profit_positions:
        assert p.is_profitable()


def test_synthetic_data_backtest_stop_loss_data_missing(
        logger: logging.Logger,
        strategy_path: Path,
        synthetic_universe: TradingStrategyUniverse,
        routing_model: BacktestRoutingModel,
    ):
    """Stop loss backtesting fails when no data is available."""

    synthetic_universe.backtest_stop_loss_candles = None

    assert not synthetic_universe.has_stop_loss_data()

    start_at, end_at = synthetic_universe.data_universe.candles.get_timestamp_range()

    routing_model = generate_simple_routing_model(synthetic_universe)

    stop_loss_decide_trades = stop_loss_decide_trades_factory(stop_loss_pct=0.95)

    # Run the test
    with pytest.raises(AutoClosingOrderUnsupported):
        state, universe, debug_dump = run_backtest_inline(
            name="No stop loss",
            start_at=start_at.to_pydatetime(),
            end_at=end_at.to_pydatetime(),
            client=None,
            cycle_duration=CycleDuration.cycle_1d,
            decide_trades=stop_loss_decide_trades,
            create_trading_universe=None,
            universe=synthetic_universe,
            initial_deposit=10_000,
            reserve_currency=ReserveCurrency.busd,
            trade_routing=TradeRouting.user_supplied_routing_model,
            routing_model=routing_model,
            allow_missing_fees=True,
        )


def test_synthetic_data_backtest_trailing_stop_loss(
        logger: logging.Logger,
        strategy_path: Path,
        synthetic_universe: TradingStrategyUniverse,
        routing_model: BacktestRoutingModel,
    ):
    """Run the strategy with trailing stop losses."""

    assert synthetic_universe.has_stop_loss_data()

    start_at, end_at = synthetic_universe.data_universe.candles.get_timestamp_range()

    routing_model = generate_simple_routing_model(synthetic_universe)

    stop_loss_decide_trades = trailing_stop_loss_decide_trades_factory(trailing_stop_loss_pct=0.98)

    # Run the test
    state, universe, debug_dump = run_backtest_inline(
        name="No stop loss",
        start_at=start_at.to_pydatetime(),
        end_at=end_at.to_pydatetime(),
        client=None,
        cycle_duration=CycleDuration.cycle_1d,
        decide_trades=stop_loss_decide_trades,
        create_trading_universe=None,
        universe=synthetic_universe,
        initial_deposit=10_000,
        reserve_currency=ReserveCurrency.busd,
        trade_routing=TradeRouting.user_supplied_routing_model,
        routing_model=routing_model,
        allow_missing_fees=True,
    )

    # Expect backtesting for 213 days
    assert len(debug_dump) == 215
    assert len(list(state.portfolio.get_all_positions())) == 46
    assert len([p for p in state.portfolio.get_all_positions() if p.is_closed()]) == 46
    assert len(list(state.portfolio.get_all_trades())) == 92

    # Check we got trailing stop losses triggered
    stop_loss_positions = [p for p in state.portfolio.get_all_positions() if p.is_stop_loss()]
    trailing_stop_loss_positions = [p for p in state.portfolio.get_all_positions() if p.is_trailing_stop_loss()]
    assert len(stop_loss_positions) == 46
    assert len(trailing_stop_loss_positions) == 46

    # Check one trigger update for contents
    p: TradingPosition
    p = trailing_stop_loss_positions[0]
    assert len(p.trigger_updates) > 0
    
    # p.trigger_updates[0] has stop_loss_before = None
    tu = p.trigger_updates[1]
    assert tu.timestamp > datetime.datetime(1970, 1, 1)
    assert tu.mid_price > 1000.0  # ETH/USD
    assert tu.stop_loss_before < tu.mid_price
    assert tu.stop_loss_after < tu.mid_price
    assert tu.stop_loss_after > tu.stop_loss_before

    # Check some of the trailing stop losses ended up for profit
    profitable_trailing_stop_losses = [p for p in trailing_stop_loss_positions if p.is_profitable()]
    assert len(profitable_trailing_stop_losses) == 18

    # Check we do not inject bad state variables
    dump = state.to_json()
    state2: State = State.from_json(dump)
    assert len(state2.portfolio.closed_positions) > 0


def test_synthetic_data_backtest_stop_loss_usd(
        logger: logging.Logger,
        strategy_path: Path,
        synthetic_universe: TradingStrategyUniverse,
        routing_model: BacktestRoutingModel,
    ):
    """Run the strategy that triggers stop losses."""

    assert synthetic_universe.has_stop_loss_data()

    start_at, end_at = synthetic_universe.data_universe.candles.get_timestamp_range()

    routing_model = generate_simple_routing_model(synthetic_universe)

    stop_loss_decide_trades = stop_loss_decide_trades_factory(stop_loss_pct=None)

    # Run the test
    state, universe, debug_dump = run_backtest_inline(
        name="No stop loss",
        start_at=start_at.to_pydatetime(),
        end_at=end_at.to_pydatetime(),
        client=None,  # None of downloads needed, because we are using synthetic data
        cycle_duration=CycleDuration.cycle_1d,  # Override to use 24h cycles despite what strategy file says
        decide_trades=stop_loss_decide_trades,
        create_trading_universe=None,
        universe=synthetic_universe,
        initial_deposit=10_000,
        reserve_currency=ReserveCurrency.busd,
        trade_routing=TradeRouting.user_supplied_routing_model,
        routing_model=routing_model,
        log_level=logging.WARNING,
        allow_missing_fees=True,
    )

    # Expect backtesting for 213 days
    assert len(debug_dump) == 215
    assert len(list(state.portfolio.get_all_positions())) == 9
    assert len([p for p in state.portfolio.get_all_positions() if p.is_open()]) == 1
    assert len(list(state.portfolio.get_all_trades())) == 17

    analysis = build_trade_analysis(state.portfolio)
    summary = analysis.calculate_summary_statistics()
    profitability = summary.realised_profit / summary.initial_cash
    assert profitability > 0

    #
    # Now run the same strategy with stop less set to 5%
    #

    stop_loss_decide_trades = stop_loss_decide_trades_factory(stop_loss_pct=0.95)
    state, universe, debug_dump = run_backtest_inline(
        name="With 95% stop loss",
        start_at=start_at.to_pydatetime(),
        end_at=end_at.to_pydatetime(),
        client=None,
        cycle_duration=CycleDuration.cycle_1d,
        decide_trades=stop_loss_decide_trades,
        create_trading_universe=None,
        universe=synthetic_universe,
        initial_deposit=10_000,
        reserve_currency=ReserveCurrency.busd,
        trade_routing=TradeRouting.user_supplied_routing_model,
        routing_model=routing_model,
        log_level=logging.WARNING,
        allow_missing_fees=True,
    )

    assert universe.backtest_stop_loss_candles
    assert universe.backtest_stop_loss_time_bucket

    assert len(universe.data_universe.candles.df) == 214
    assert len(universe.backtest_stop_loss_candles.df) == 5136

    # Expect backtesting for 213 days
    assert len(debug_dump) == 215

    # Check that all positions had stop loss set
    for p in state.portfolio.get_all_positions():
        assert p.stop_loss, f"Position did not have stop loss: {p}"

    assert len(list(state.portfolio.get_all_positions())) == 31
    assert len([p for p in state.portfolio.get_all_positions() if p.is_closed()]) == 31
    assert len(list(state.portfolio.get_all_trades())) == 62

    # We should have some stop loss trades and some trades closed for profit
    stop_loss_trades = [t for t in state.portfolio.get_all_trades() if t.is_stop_loss()]
    rebalance_trades = [t for t in state.portfolio.get_all_trades() if t.is_rebalance()]
    assert len(rebalance_trades) == 37
    assert len(stop_loss_trades) == 25

    # Check are stop loss positions unprofitable
    stop_loss_positions = [p for p in state.portfolio.get_all_positions() if p.is_stop_loss()]
    for p in stop_loss_positions:
        assert p.is_loss()


def test_synthetic_data_backtest_stop_loss_data_export(
        logger: logging.Logger,
        strategy_path: Path,
        synthetic_universe: TradingStrategyUniverse,
        routing_model: BacktestRoutingModel,
    ):
    """Export the trailing stop loss analytics data frame."""

    assert synthetic_universe.has_stop_loss_data()

    start_at, end_at = synthetic_universe.data_universe.candles.get_timestamp_range()

    routing_model = generate_simple_routing_model(synthetic_universe)

    stop_loss_decide_trades = trailing_stop_loss_decide_trades_factory(trailing_stop_loss_pct=0.98)

    # Run the test
    state, universe, debug_dump = run_backtest_inline(
        name="No stop loss",
        start_at=start_at.to_pydatetime(),
        end_at=end_at.to_pydatetime(),
        client=None,
        cycle_duration=CycleDuration.cycle_1d,
        decide_trades=stop_loss_decide_trades,
        create_trading_universe=None,
        universe=synthetic_universe,
        initial_deposit=10_000,
        reserve_currency=ReserveCurrency.busd,
        trade_routing=TradeRouting.user_supplied_routing_model,
        routing_model=routing_model,
        allow_missing_fees=True,
    )

    df = analyse_stop_losses(state)
    assert len(df) == 46

    p = state.portfolio.get_position_by_id(1)
    df = analyse_trigger_updates(p)
    assert len(df) == 2