"""Run a backtest where we open a credit supply position, using real oracle data.

"""
import os
import logging
import datetime
import random
from _decimal import Decimal

import pytest
from typing import List, Dict

import pandas as pd

from tradingstrategy.client import Client
from tradingstrategy.chain import ChainId
from tradingstrategy.lending import LendingProtocolType
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.universe import Universe

from tradeexecutor.backtest.backtest_pricing import BacktestSimplePricingModel
from tradeexecutor.backtest.backtest_routing import BacktestRoutingModel
from tradeexecutor.backtest.backtest_runner import run_backtest_inline
from tradeexecutor.backtest.simulated_wallet import SimulatedWallet
from tradeexecutor.ethereum.routing_data import get_backtest_routing_model
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.state import State
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier, TradingPairKind
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, load_partial_data
from tradeexecutor.strategy.execution_context import ExecutionContext, unit_test_execution_context
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.reserve_currency import ReserveCurrency
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.universe_model import UniverseOptions, default_universe_options
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, create_pair_universe_from_code
from tradeexecutor.testing.synthetic_ethereum_data import generate_random_ethereum_address
from tradeexecutor.testing.synthetic_exchange_data import generate_exchange
from tradeexecutor.testing.synthetic_price_data import generate_ohlcv_candles
from tradeexecutor.testing.synthetic_universe_data import create_synthetic_single_pair_universe
from tradeexecutor.testing.synthetic_lending_data import generate_lending_reserve, generate_lending_universe


# https://docs.pytest.org/en/latest/how-to/skipping.html#skip-all-test-functions-of-a-class-or-module
pytestmark = pytest.mark.skipif(os.environ.get("TRADING_STRATEGY_API_KEY") is None, reason="Set TRADING_STRATEGY_API_KEY environment variable to run this test")

start_at = datetime.datetime(2023, 1, 1)
end_at = datetime.datetime(2023, 1, 5)

@pytest.fixture(scope="module")
def universe() -> TradingStrategyUniverse:
    """Set up a mock universe."""

    time_bucket = TimeBucket.d1

    # Set up fake assets
    chain_id = ChainId.ethereum
    mock_exchange = generate_exchange(
        exchange_id=random.randint(1, 1000),
        chain_id=chain_id,
        address=generate_random_ethereum_address(),
    )
    usdc = AssetIdentifier(chain_id.value, generate_random_ethereum_address(), "USDC", 6, 1)
    weth = AssetIdentifier(chain_id.value, generate_random_ethereum_address(), "WETH", 18, 2)
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

    candles = generate_ohlcv_candles(
        time_bucket,
        start_at,
        end_at,
        start_price=1800,
        pair_id=weth_usdc.internal_id,
        exchange_id=mock_exchange.exchange_id,
    )

    return create_synthetic_single_pair_universe(
        candles=candles,
        chain_id=chain_id,
        exchange=mock_exchange,
        time_bucket=time_bucket,
        pair=weth_usdc,
        lending_candles=lending_candle_universe,
    )


def test_backtest_open_only_short_synthetic_data(
    persistent_test_client: Client,
    universe,
):
    """Run the strategy backtest using inline decide_trades function.

    - Open short position
    - Check unrealised PnL after 4 days
    - ETH price goes 1794 -> 1712
    - Short goes to profit
    """

    capital = 10000
    leverage = 2

    def decide_trades(
        timestamp: pd.Timestamp,
        strategy_universe: TradingStrategyUniverse,
        state: State,
        pricing_model: PricingModel,
        cycle_debug_data: Dict
    ) -> List[TradeExecution]:
        """A simple strategy that opens a single 2x short position."""
        trade_pair = strategy_universe.universe.pairs.get_single()

        cash = state.portfolio.get_cash()
        
        position_manager = PositionManager(timestamp, strategy_universe, state, pricing_model)

        trades = []

        if not position_manager.is_any_open():
            trades += position_manager.open_short(trade_pair, cash, leverage=leverage)
            t: TradeExecution
            t = trades[0]
            assert t.planned_price == pytest.approx(1794.6)  # ETH opening value

        return trades

    # Run the test
    state, universe, debug_dump = run_backtest_inline(
        start_at=start_at,
        end_at=end_at,
        client=persistent_test_client,
        cycle_duration=CycleDuration.cycle_1d,
        decide_trades=decide_trades,
        universe=universe,
        initial_deposit=capital,
        reserve_currency=ReserveCurrency.usdc,
        trade_routing=TradeRouting.uniswap_v3_usdc_poly,
        engine_version="0.3",
    )

    portfolio = state.portfolio
    assert len(portfolio.open_positions) == 1

    # Check that the unrealised position looks good
    position = portfolio.open_positions[1]
    assert position.is_short()
    assert position.is_open()
    assert position.pair.kind.is_shorting()
    assert position.get_value_at_open() == capital
    assert position.get_collateral() == pytest.approx(19970)
    assert position.get_borrowed() == pytest.approx(9540.86179207702)
    assert position.opened_at == datetime.datetime(2023, 1, 1)
    assert position.get_accrued_interest() == pytest.approx(1.71447282377892)
    assert position.get_claimed_interest() == pytest.approx(0)
    assert position.get_realised_profit_usd() is None
    assert position.get_unrealised_profit_usd() == pytest.approx(460.92815965634367)
    assert position.get_value() == Decimal(10430.85268074676)

    # Check 1st trade looks good
    trade = position.get_first_trade()
    assert trade.opened_at == datetime.datetime(2023, 1, 1)
    assert trade.planned_price == pytest.approx(1794.6)  # ETH opening value
    assert trade.get_planned_value() == 10000
    assert float(trade.planned_quantity) == pytest.approx(-5.572272372673576)

    # Check that the loan object looks good
    loan = position.loan
    assert loan.get_net_asset_value() == pytest.approx(10430.85268074676)
    assert loan.collateral.get_usd_value() == pytest.approx(19970)
    assert loan.borrowed.get_usd_value() == pytest.approx(9540.86179207702)
    assert loan.borrowed.last_usd_price == pytest.approx(1712.203057206142)  # ETH current value
    assert loan.get_collateral_interest() == pytest.approx(3.282919605462178)
    assert loan.get_collateral_quantity() == pytest.approx(Decimal(19973.28291960546217718918569))
    assert loan.get_borrowed_quantity() == pytest.approx(Decimal(5.573188412844794660521435197))
    assert loan.get_borrow_interest() == pytest.approx(1.568446781683258)
    assert loan.get_net_interest() == pytest.approx(1.71447282377892)

    # Check that the portfolio looks good
    assert portfolio.get_cash() == 0
    assert portfolio.get_net_asset_value(include_interest=True) == pytest.approx(10430.85268074676)
    assert portfolio.get_net_asset_value(include_interest=False) == pytest.approx(10429.13820792298)
    # difference should come from interest
    assert portfolio.get_net_asset_value(include_interest=True) - portfolio.get_net_asset_value(include_interest=False) == pytest.approx(loan.get_net_interest())

    # Check token balances in the wallet
    wallet = debug_dump["wallet"]
    balances = wallet.balances
    pair = position.pair
    usdc = pair.quote.underlying
    ausdc = pair.quote
    vweth = pair.base
    weth = pair.base.underlying

    assert balances[usdc.address] == 0
    assert balances[ausdc.address] == pytest.approx(Decimal(19973.28291960546217718918569))
    assert balances[vweth.address] == pytest.approx(Decimal(5.573188412844794660521435197))
    assert balances.get(weth.address, Decimal(0)) == pytest.approx(Decimal(0))


def test_backtest_open_and_close_short_synthetic_data(
    persistent_test_client: Client,
    universe,
):
    """Run the strategy backtest using inline decide_trades function.

    - Open short position
    - ETH price goes 1794 -> 1712
    - Short goes to profit
    - Close short position after 4 days
    """

    capital = 10000
    leverage = 2

    def decide_trades(
        timestamp: pd.Timestamp,
        strategy_universe: TradingStrategyUniverse,
        state: State,
        pricing_model: PricingModel,
        cycle_debug_data: Dict
    ) -> List[TradeExecution]:
        """A simple strategy that opens and closes a single 2x short position."""
        trade_pair = strategy_universe.universe.pairs.get_single()

        cash = state.portfolio.get_cash()
        
        position_manager = PositionManager(timestamp, strategy_universe, state, pricing_model)

        trades = []

        if not position_manager.is_any_open():
            trades += position_manager.open_short(trade_pair, cash, leverage=leverage)
        else:
            if timestamp == datetime.datetime(2023, 1, 4):

                # Check that how much total collateral we should receive
                position = position_manager.get_current_position()
                loan = position.loan
                assert loan.get_net_asset_value() == pytest.approx(10430.85268074676)
                assert loan.get_collateral_interest() == pytest.approx(3.282919605462178)
                assert loan.get_borrow_interest() == pytest.approx(1.568446781683258)
                assert loan.get_net_interest() == pytest.approx(1.71447282377892)

                received_cash = loan.get_collateral_value(include_interest=False) + loan.get_collateral_interest() - loan.get_borrow_value(include_interest=False) - loan.get_borrow_interest()
                # Interest double counted: 10462
                assert received_cash == pytest.approx(10430.85268074676)

                trades += position_manager.close_all()

        return trades

    # Run the test
    state, universe, debug_dump = run_backtest_inline(
        start_at=start_at,
        end_at=end_at,
        client=persistent_test_client,
        cycle_duration=CycleDuration.cycle_1d,
        decide_trades=decide_trades,
        universe=universe,
        initial_deposit=capital,
        reserve_currency=ReserveCurrency.usdc,
        trade_routing=TradeRouting.uniswap_v3_usdc_poly,
        engine_version="0.3",
    )

    portfolio = state.portfolio
    assert len(portfolio.open_positions) == 0
    assert len(portfolio.closed_positions) == 1

    # Check that the unrealised position looks good
    position = portfolio.closed_positions[1]
    assert position.is_short()
    assert position.is_closed()
    assert position.pair.kind.is_shorting()
    assert position.get_value_at_open() == capital
    assert position.get_collateral() == 0
    assert position.get_borrowed() == 0
    assert position.get_accrued_interest() == pytest.approx(0)
    assert position.get_claimed_interest() == pytest.approx(3.282919605462178)
    assert position.get_repaid_interest() == pytest.approx(1.568446781683258)
    assert position.get_realised_profit_usd() == pytest.approx(460.92815965634367)
    assert position.get_unrealised_profit_usd() == 0
    assert position.get_value() == pytest.approx(0)

    # Check opening trade looks good
    assert len(position.trades) == 2
    open_trade = position.get_first_trade()
    assert open_trade.opened_at == datetime.datetime(2023, 1, 1)
    assert open_trade.planned_price == pytest.approx(1794.6)  # ETH opening value
    assert open_trade.planned_mid_price == pytest.approx(1800.0)  # ETH opening mid price
    assert open_trade.get_planned_value() == 10000
    assert float(open_trade.planned_quantity) == pytest.approx(-5.572272372673576)

    # Check closing trade looks good
    close_trade = position.get_last_trade()
    assert close_trade.opened_at == datetime.datetime(2023, 1, 4)
    assert close_trade.planned_price == pytest.approx(1712.203057206142)  # ETH current value
    assert close_trade.get_planned_value() == pytest.approx(9542.430238858704)
    assert float(close_trade.planned_quantity) == pytest.approx(5.573188412844795)

    # # Check that the portfolio looks good
    assert portfolio.get_cash() == pytest.approx(10430.852680746759)
    assert portfolio.get_net_asset_value(include_interest=True) == pytest.approx(10430.852680746759)

    # Check token balances in the wallet
    wallet = debug_dump["wallet"]
    balances = wallet.balances
    pair = position.pair
    usdc = pair.quote.underlying
    ausdc = pair.quote
    vweth = pair.base
    weth = pair.base.underlying
    assert balances[ausdc.address] == pytest.approx(Decimal(0))
    assert balances[vweth.address] == pytest.approx(Decimal(0))
    assert balances.get(weth.address, Decimal(0)) == pytest.approx(Decimal(0))
    assert balances[usdc.address] == pytest.approx(Decimal(10430.852680746759))


def test_backtest_short_underlying_price_feed(
    persistent_test_client: Client,
    universe: TradingStrategyUniverse,
):
    """Query the price for a short pair.

    - We need to resolve the underlying trading pair and ask its price
    """

    routing_model = get_backtest_routing_model(
        TradeRouting.ignore,
        ReserveCurrency.usdc,
    )

    pricing_model = BacktestSimplePricingModel(
        universe.universe.candles,
        routing_model,
    )

    spot_pair = universe.get_single_pair()
    shorting_pair = universe.get_shorting_pair(spot_pair)

    spot_pair_two = shorting_pair.get_pricing_pair()
    assert spot_pair == spot_pair_two

    price_structure = pricing_model.get_buy_price(
        start_at,
        spot_pair_two,
        Decimal(10_000)
    )

    assert price_structure.mid_price == pytest.approx(1800.0)


def test_backtest_open_short_failure_too_high_leverage(persistent_test_client: Client, universe):
    """Backtest should raise exception if leverage is too high."""

    def decide_trades(
        timestamp: pd.Timestamp,
        strategy_universe: TradingStrategyUniverse,
        state: State,
        pricing_model: PricingModel,
        cycle_debug_data: Dict
    ) -> List[TradeExecution]:
        """A simple strategy that opens a single 10x short position."""
        trade_pair = strategy_universe.universe.pairs.get_single()

        cash = state.portfolio.get_cash()
        
        position_manager = PositionManager(timestamp, strategy_universe, state, pricing_model)

        trades = []

        if not position_manager.is_any_open():
            trades += position_manager.open_short(trade_pair, cash, leverage=10)

        return trades

    # backtest should raise exception when trying to open short position
    with pytest.raises(AssertionError) as e:
        _, universe, _ = run_backtest_inline(
            start_at=start_at,
            end_at=end_at,
            client=persistent_test_client,
            cycle_duration=CycleDuration.cycle_1d,
            decide_trades=decide_trades,
            universe=universe,
            initial_deposit=10000,
            reserve_currency=ReserveCurrency.usdc,
            trade_routing=TradeRouting.uniswap_v3_usdc_poly,
            engine_version="0.3",
        )

    assert str(e.value) == "Max short leverage for USDC is 5.666666666666666, got 10"


def test_backtest_open_short_failure_too_far_stoploss(persistent_test_client: Client, universe):
    """Backtest should raise exception if stoploss is higher than liquidation price."""

    def decide_trades(
        timestamp: pd.Timestamp,
        strategy_universe: TradingStrategyUniverse,
        state: State,
        pricing_model: PricingModel,
        cycle_debug_data: Dict
    ) -> List[TradeExecution]:
        """A simple strategy that opens a single 10x short position."""
        trade_pair = strategy_universe.universe.pairs.get_single()

        cash = state.portfolio.get_cash()
        
        position_manager = PositionManager(timestamp, strategy_universe, state, pricing_model)

        trades = []

        if not position_manager.is_any_open():
            trades += position_manager.open_short(trade_pair, cash, leverage=4, stop_loss_pct=0.6)

        return trades

    # backtest should raise exception when trying to open short position
    with pytest.raises(AssertionError) as e:
        _, universe, _ = run_backtest_inline(
            start_at=start_at,
            end_at=end_at,
            client=persistent_test_client,
            cycle_duration=CycleDuration.cycle_1d,
            decide_trades=decide_trades,
            universe=universe,
            initial_deposit=10000,
            reserve_currency=ReserveCurrency.usdc,
            trade_routing=TradeRouting.uniswap_v3_usdc_poly,
            engine_version="0.3",
        )

    assert str(e.value) == "stop_loss_pct must be bigger than liquidation distance 0.8701, got 0.6"


def test_backtest_short_stop_loss_triggered(persistent_test_client: Client, universe):
    """Run the strategy backtest using inline decide_trades function.

    - Open short position, set a 1% stoploss
    - ETH price goes 1686 -> 1712
    - Position should be closed automatically with a loss
    """

    def decide_trades(
        timestamp: pd.Timestamp,
        strategy_universe: TradingStrategyUniverse,
        state: State,
        pricing_model: PricingModel,
        cycle_debug_data: Dict
    ) -> List[TradeExecution]:
        """A simple strategy that opens a single 4x short position."""
        trade_pair = strategy_universe.universe.pairs.get_single()

        cash = state.portfolio.get_cash()
        position_size = cash * 0.8

        position_manager = PositionManager(timestamp, strategy_universe, state, pricing_model)

        trades = []

        if not position_manager.is_any_open() and timestamp == datetime.datetime(2023, 1, 3):
            trades += position_manager.open_short(trade_pair, position_size, leverage=4, stop_loss_pct=0.99)

        return trades

    state, universe, debug_dump = run_backtest_inline(
        start_at=start_at,
        end_at=end_at,
        client=persistent_test_client,
        cycle_duration=CycleDuration.cycle_1d,
        decide_trades=decide_trades,
        universe=universe,
        initial_deposit=10000,
        reserve_currency=ReserveCurrency.usdc,
        trade_routing=TradeRouting.uniswap_v3_usdc_poly,
        engine_version="0.3",
    )

    portfolio = state.portfolio
    assert len(portfolio.open_positions) == 0
    assert len(portfolio.closed_positions) == 1

    # Check that the unrealised position looks good
    position = portfolio.closed_positions[1]
    assert position.is_short()
    assert position.is_closed()
    assert position.pair.kind.is_shorting()
    assert position.is_stop_loss()

    assert position.liquidation_price == pytest.approx(Decimal(1911.5195057377280))
    assert position.stop_loss == pytest.approx(Decimal(1708.6270878474588))

    # assert position.get_value_at_open() == 8000
    assert position.get_collateral() == 0
    assert position.get_borrowed() == 0
    assert position.get_accrued_interest() == pytest.approx(0)
    assert position.get_unrealised_profit_usd() == 0
    assert position.get_realised_profit_usd() == pytest.approx(-363.82313453462916)
    assert position.get_claimed_interest() == pytest.approx(0)
    assert position.get_repaid_interest() == pytest.approx(0)
    assert position.get_value() == pytest.approx(0)

    # Check opening trade looks good
    assert len(position.trades) == 2
    open_trade = position.get_first_trade()
    assert open_trade.opened_at == datetime.datetime(2023, 1, 3)
    assert open_trade.planned_price == pytest.approx(1686.634)  # ETH opening value
    assert open_trade.get_planned_value() == 24000
    assert open_trade.planned_quantity == pytest.approx(Decimal(-14.22951736477441))

    # Check closing trade looks good
    close_trade = position.get_last_trade()
    assert close_trade.opened_at == datetime.datetime(2023, 1, 4)
    assert close_trade.planned_price == pytest.approx(1712.203057206142)  # ETH current value
    assert close_trade.get_planned_value() == pytest.approx(24363.82313453463)
    assert close_trade.planned_quantity == pytest.approx(Decimal(14.22951736477441))

    # Check that the portfolio looks good
    assert portfolio.get_cash() == pytest.approx(9564.176865465372)
    assert portfolio.get_net_asset_value(include_interest=True) == pytest.approx(9564.176865465372)

    # Check token balances in the wallet
    wallet = debug_dump["wallet"]
    balances = wallet.balances
    pair = position.pair
    usdc = pair.quote.underlying
    ausdc = pair.quote
    vweth = pair.base
    weth = pair.base.underlying
    assert balances[ausdc.address] == pytest.approx(Decimal(0))
    assert balances[vweth.address] == pytest.approx(Decimal(0))
    assert balances.get(weth.address, Decimal(0)) == pytest.approx(Decimal(0))
    assert balances[usdc.address] == pytest.approx(Decimal(9564.176865465372))


def test_backtest_short_take_profit_triggered(persistent_test_client: Client, universe):
    """Run the strategy backtest using inline decide_trades function.

    - Open short position, set a 2% take profit
    - ETH price goes 1794 -> 1712
    - Position should be closed automatically with profit
    """

    def decide_trades(
        timestamp: pd.Timestamp,
        strategy_universe: TradingStrategyUniverse,
        state: State,
        pricing_model: PricingModel,
        cycle_debug_data: Dict
    ) -> List[TradeExecution]:
        """A simple strategy that opens a single 4x short position."""
        trade_pair = strategy_universe.universe.pairs.get_single()

        cash = state.portfolio.get_cash()
        position_size = cash * 0.8

        position_manager = PositionManager(timestamp, strategy_universe, state, pricing_model)

        trades = []

        if not position_manager.is_any_open() and timestamp == datetime.datetime(2023, 1, 1):
            trades += position_manager.open_short(trade_pair, position_size, leverage=4, take_profit_pct=1.02)

        return trades

    state, universe, debug_dump = run_backtest_inline(
        start_at=start_at,
        end_at=end_at,
        client=persistent_test_client,
        cycle_duration=CycleDuration.cycle_1d,
        decide_trades=decide_trades,
        universe=universe,
        initial_deposit=10000,
        reserve_currency=ReserveCurrency.usdc,
        trade_routing=TradeRouting.uniswap_v3_usdc_poly,
        engine_version="0.3",
    )

    portfolio = state.portfolio
    assert len(portfolio.open_positions) == 0
    assert len(portfolio.closed_positions) == 1

    # Check that the unrealised position looks good
    position = portfolio.closed_positions[1]
    assert position.is_short()
    assert position.is_closed()
    assert position.pair.kind.is_shorting()
    assert position.is_take_profit()
    assert position.take_profit == pytest.approx(1764.0)
    assert position.liquidation_price == pytest.approx(Decimal(2033.879999999999843))

    # assert position.get_value_at_open() == capital
    assert position.get_collateral() == 0
    assert position.get_borrowed() == 0
    assert position.get_accrued_interest() == pytest.approx(0)
    assert position.get_unrealised_profit_usd() == 0
    assert position.get_realised_profit_usd() == pytest.approx(877.5258141302367)
    assert position.get_claimed_interest() == pytest.approx(0)
    assert position.get_repaid_interest() == pytest.approx(0)
    assert position.get_value() == pytest.approx(0)

    # Check opening trade looks good
    assert len(position.trades) == 2
    open_trade = position.get_first_trade()
    assert open_trade.opened_at == datetime.datetime(2023, 1, 1)
    assert open_trade.planned_price == pytest.approx(1794.6)  # ETH opening value
    assert open_trade.get_planned_value() == 24000
    assert open_trade.planned_quantity == pytest.approx(Decimal(-13.373453694416584))

    # Check closing trade looks good
    close_trade = position.get_last_trade()
    assert close_trade.opened_at == datetime.datetime(2023, 1, 2)
    assert close_trade.planned_price == pytest.approx(1728.9830072484115)  # ETH current value
    assert close_trade.get_planned_value() == pytest.approx(23122.474185869763)
    assert close_trade.planned_quantity == pytest.approx(Decimal(13.373453694416584))

    # Check that the portfolio looks good
    assert portfolio.get_cash() == pytest.approx(10805.525814130237)
    assert portfolio.get_net_asset_value(include_interest=True) == pytest.approx(10805.525814130237)

    # Check token balances in the wallet
    wallet = debug_dump["wallet"]
    balances = wallet.balances
    pair = position.pair
    usdc = pair.quote.underlying
    ausdc = pair.quote
    vweth = pair.base
    weth = pair.base.underlying
    assert balances[ausdc.address] == pytest.approx(Decimal(0))
    assert balances[vweth.address] == pytest.approx(Decimal(0))
    assert balances.get(weth.address, Decimal(0)) == pytest.approx(Decimal(0))
    assert balances[usdc.address] == pytest.approx(Decimal(10805.525814130237))
