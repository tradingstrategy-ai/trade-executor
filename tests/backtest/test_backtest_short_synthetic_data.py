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

from tradeexecutor.analysis.trade_analyser import build_trade_analysis
from tradeexecutor.backtest.backtest_pricing import BacktestPricing
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
candle_end_at = datetime.datetime(2023, 1, 30)


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
        candle_end_at,
        reserves=[usdc_reserve, weth_reserve],
        aprs={
            "supply": 2,
            "variable": 5,
        }
    )

    candles = generate_ohlcv_candles(
        time_bucket,
        start_at,
        candle_end_at,
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


@pytest.fixture(scope="module")
def strategy_universe(universe) -> TradingStrategyUniverse:
    return universe


def test_backtest_open_only_short_synthetic_data(
    persistent_test_client: Client,
    strategy_universe,
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
        trade_pair = strategy_universe.data_universe.pairs.get_single()

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
    state, strategy_universe, debug_dump = run_backtest_inline(
        start_at=start_at,
        end_at=end_at,
        client=persistent_test_client,
        cycle_duration=CycleDuration.cycle_1d,
        decide_trades=decide_trades,
        universe=strategy_universe,
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
    assert position.get_value_at_open() == pytest.approx(19940)
    assert position.get_collateral() == pytest.approx(29940)
    assert position.get_borrowed() == pytest.approx(19024.478413401575)
    assert position.opened_at == datetime.datetime(2023, 1, 1)
    assert position.get_accrued_interest() == pytest.approx(-2.897436326371551)
    assert position.get_claimed_interest() == pytest.approx(0)
    assert position.get_realised_profit_usd() is None
    assert position.get_unrealised_profit_usd() == pytest.approx(913.0004435622891)
    assert position.get_value() == Decimal(10912.624150272055)

    # Check 1st trade looks good
    trade = position.get_first_trade()
    assert trade.opened_at == datetime.datetime(2023, 1, 1)
    assert trade.planned_price == pytest.approx(1794.6)  # ETH opening value
    assert trade.get_planned_value() == pytest.approx(19940)
    assert float(trade.planned_quantity) == pytest.approx(-11.11111111111111)

    # Check that the loan object looks good
    loan = position.loan
    assert loan.get_net_asset_value() == pytest.approx(10912.624150272055)
    assert loan.collateral.get_usd_value() == pytest.approx(29940)
    assert loan.borrowed.get_usd_value() == pytest.approx(19024.478413401575)
    assert loan.borrowed.last_usd_price == pytest.approx(1712.203057206142)  # ETH current value
    assert loan.get_collateral_interest() == pytest.approx(4.921913519656365)
    assert loan.get_collateral_quantity() == pytest.approx(Decimal(29944.92191351965636348513714))
    assert loan.get_borrowed_quantity() == pytest.approx(Decimal(11.11567794669356066441189513))
    assert loan.get_borrow_interest() == pytest.approx(7.819349846027916)
    assert loan.get_net_interest() == pytest.approx(-2.897436326371551)

    # Check that the portfolio looks good
    assert portfolio.get_cash() == 0
    assert portfolio.get_net_asset_value(include_interest=True) == pytest.approx(10912.624150272055)
    assert portfolio.get_net_asset_value(include_interest=False) == pytest.approx(10915.521586598425)
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
    assert balances[ausdc.address] == pytest.approx(Decimal(29944.92191351965636348513714))
    assert balances[vweth.address] == pytest.approx(Decimal(11.11567794669356066441189513))
    assert balances.get(weth.address, Decimal(0)) == pytest.approx(Decimal(0))


def test_backtest_open_and_close_short_synthetic_data(
    persistent_test_client: Client,
        strategy_universe,
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
        trade_pair = strategy_universe.data_universe.pairs.get_single()

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
                assert loan.get_net_asset_value() == pytest.approx(10912.624150272055)
                assert loan.get_collateral_interest() == pytest.approx(4.921913519656365)
                assert loan.get_borrow_interest() == pytest.approx(7.819349846027916)
                assert loan.get_net_interest() == pytest.approx(-2.897436326371551)

                received_cash = loan.get_collateral_value(include_interest=False) + loan.get_collateral_interest() - loan.get_borrow_value(include_interest=False) - loan.get_borrow_interest()
                # Interest double counted: 10462
                assert received_cash == pytest.approx(10912.624150272055)

                trades += position_manager.close_all()

        return trades

    # Run the test
    state, strategy_universe, debug_dump = run_backtest_inline(
        start_at=start_at,
        end_at=end_at,
        client=persistent_test_client,
        cycle_duration=CycleDuration.cycle_1d,
        decide_trades=decide_trades,
        universe=strategy_universe,
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
    assert position.get_value_at_open() == pytest.approx(19940)
    assert position.get_collateral() == 0
    assert position.get_borrowed() == 0
    assert position.get_accrued_interest() == pytest.approx(0)
    assert position.get_claimed_interest() == pytest.approx(4.921913519656365)
    assert position.get_repaid_interest() == pytest.approx(7.86640711691675)
    assert position.get_realised_profit_usd() == pytest.approx(798.4159875155882)
    assert position.get_unrealised_profit_usd() == 0
    assert position.get_value() == pytest.approx(0)

    # Check opening trade looks good
    assert len(position.trades) == 2
    open_trade = position.get_first_trade()
    assert open_trade.opened_at == datetime.datetime(2023, 1, 1)
    assert open_trade.planned_price == pytest.approx(1794.6)  # ETH opening value
    assert open_trade.planned_mid_price == pytest.approx(1800.0)  # ETH opening mid price
    assert open_trade.get_planned_value() == pytest.approx(19940)
    assert float(open_trade.planned_quantity) == pytest.approx(-11.11111111111111)

    # Check closing trade looks good
    close_trade = position.get_last_trade()
    assert close_trade.opened_at == datetime.datetime(2023, 1, 4)
    assert close_trade.planned_price == pytest.approx(1722.507187941585)  # ETH current value
    assert close_trade.get_planned_value() == pytest.approx(19146.835162023417)
    assert float(close_trade.planned_quantity) == pytest.approx(11.11567794669356)

    # # Check that the portfolio looks good
    assert portfolio.get_cash() == pytest.approx(10797.914428428872)
    assert portfolio.get_net_asset_value(include_interest=True) == pytest.approx(10797.914428428872)

    #
    # Check token balances in the wallet.
    # Here we have around 4.92191351965636473413804 USD interest on aUSD which we
    # also need to get back to the wallet when we close the position.
    #

    wallet: SimulatedWallet = debug_dump["wallet"]
    balances = wallet.balances
    pair = position.pair
    usdc = pair.quote.underlying
    ausdc = pair.quote
    vweth = pair.base
    weth = pair.base.underlying
    # print(wallet.get_all_balances())
    # import ipdb ; ipdb.set_trace()
    assert balances[ausdc.address] == pytest.approx(Decimal(0))
    assert balances[vweth.address] == pytest.approx(Decimal(0))
    assert balances.get(weth.address, Decimal(0)) == pytest.approx(Decimal(0))
    assert balances[usdc.address] == pytest.approx(Decimal(10797.914428428872))


def test_backtest_short_underlying_price_feed(
    persistent_test_client: Client,
        strategy_universe,
):
    """Query the price for a short pair.

    - We need to resolve the underlying trading pair and ask its price
    """

    routing_model = get_backtest_routing_model(
        TradeRouting.ignore,
        ReserveCurrency.usdc,
    )

    pricing_model = BacktestPricing(
        strategy_universe.data_universe.candles,
        routing_model,
    )

    spot_pair = strategy_universe.get_single_pair()
    shorting_pair = strategy_universe.get_shorting_pair(spot_pair)

    spot_pair_two = shorting_pair.get_pricing_pair()
    assert spot_pair == spot_pair_two

    price_structure = pricing_model.get_buy_price(
        start_at,
        spot_pair_two,
        Decimal(10_000)
    )

    assert price_structure.mid_price == pytest.approx(1800.0)


def test_backtest_open_short_failure_too_high_leverage(persistent_test_client: Client, strategy_universe):
    """Backtest should raise exception if leverage is too high."""

    def decide_trades(
        timestamp: pd.Timestamp,
        strategy_universe: TradingStrategyUniverse,
        state: State,
        pricing_model: PricingModel,
        cycle_debug_data: Dict
    ) -> List[TradeExecution]:
        """A simple strategy that opens a single 10x short position."""
        trade_pair = strategy_universe.data_universe.pairs.get_single()

        cash = state.portfolio.get_cash()
        
        position_manager = PositionManager(timestamp, strategy_universe, state, pricing_model)

        trades = []

        if not position_manager.is_any_open():
            trades += position_manager.open_short(trade_pair, cash, leverage=10)

        return trades

    # backtest should raise exception when trying to open short position
    with pytest.raises(AssertionError) as e:
        _, strategy_universe, _ = run_backtest_inline(
            start_at=start_at,
            end_at=end_at,
            client=persistent_test_client,
            cycle_duration=CycleDuration.cycle_1d,
            decide_trades=decide_trades,
            universe=strategy_universe,
            initial_deposit=10000,
            reserve_currency=ReserveCurrency.usdc,
            trade_routing=TradeRouting.uniswap_v3_usdc_poly,
            engine_version="0.3",
        )

    assert str(e.value) == "Max short leverage for USDC is 5.666666666666666, got 10"


def test_backtest_open_short_failure_too_far_stoploss(persistent_test_client: Client, strategy_universe):
    """Backtest should raise exception if stoploss is higher than liquidation price."""

    def decide_trades(
        timestamp: pd.Timestamp,
        strategy_universe: TradingStrategyUniverse,
        state: State,
        pricing_model: PricingModel,
        cycle_debug_data: Dict
    ) -> List[TradeExecution]:
        """A simple strategy that opens a single 10x short position."""
        trade_pair = strategy_universe.data_universe.pairs.get_single()

        cash = state.portfolio.get_cash()
        
        position_manager = PositionManager(timestamp, strategy_universe, state, pricing_model)

        trades = []

        if not position_manager.is_any_open():
            trades += position_manager.open_short(trade_pair, cash, leverage=4, stop_loss_pct=0.6)

        return trades

    # backtest should raise exception when trying to open short position
    with pytest.raises(AssertionError) as e:
        _, strategy_universe, _ = run_backtest_inline(
            start_at=start_at,
            end_at=end_at,
            client=persistent_test_client,
            cycle_duration=CycleDuration.cycle_1d,
            decide_trades=decide_trades,
            universe=strategy_universe,
            initial_deposit=10000,
            reserve_currency=ReserveCurrency.usdc,
            trade_routing=TradeRouting.uniswap_v3_usdc_poly,
            engine_version="0.3",
        )

    assert str(e.value) == "stop_loss_pct must be bigger than liquidation distance 0.9375, got 0.6"


def test_backtest_short_stop_loss_triggered(persistent_test_client: Client, strategy_universe):
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
        trade_pair = strategy_universe.data_universe.pairs.get_single()

        cash = state.portfolio.get_cash()
        position_size = cash * 0.8

        position_manager = PositionManager(timestamp, strategy_universe, state, pricing_model)

        trades = []

        if not position_manager.is_any_open() and timestamp == datetime.datetime(2023, 1, 3):
            trades += position_manager.open_short(trade_pair, position_size, leverage=4, stop_loss_pct=0.99)

        return trades

    state, strategy_universe, debug_dump = run_backtest_inline(
        start_at=start_at,
        end_at=end_at,
        client=persistent_test_client,
        cycle_duration=CycleDuration.cycle_1d,
        decide_trades=decide_trades,
        universe=strategy_universe,
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

    assert position.liquidation_price == pytest.approx(Decimal(1797.441862215767254381815544))
    assert position.stop_loss == pytest.approx(Decimal(1708.6270878474588))

    # assert position.get_value_at_open() == 8000
    assert position.get_collateral() == 0
    assert position.get_borrowed() == 0
    assert position.get_accrued_interest() == pytest.approx(0)
    assert position.get_unrealised_profit_usd() == 0
    assert position.get_realised_profit_usd() == pytest.approx(-678.5528052509759)
    assert position.get_claimed_interest() == pytest.approx(0)
    assert position.get_repaid_interest() == pytest.approx(0)
    assert position.get_value() == pytest.approx(0)

    # Check opening trade looks good
    assert len(position.trades) == 2
    open_trade = position.get_first_trade()
    assert open_trade.opened_at == datetime.datetime(2023, 1, 3)
    assert open_trade.planned_price == pytest.approx(1686.634)  # ETH opening value
    assert open_trade.get_planned_value() == pytest.approx(31904)
    assert open_trade.planned_quantity == pytest.approx(Decimal(-18.91577175024011707186953272))

    # Check closing trade looks good
    close_trade = position.get_last_trade()
    assert close_trade.opened_at == datetime.datetime(2023, 1, 4)
    assert close_trade.planned_price == pytest.approx(1722.507187941585)  # ETH current value
    assert close_trade.get_planned_value() == pytest.approx(32582.55280525098)
    assert close_trade.planned_quantity == pytest.approx(Decimal(18.91577175024011707186953272))

    # Check that the portfolio looks good
    assert portfolio.get_cash() == pytest.approx(9321.153949134561)
    # loss should be equal to the difference between the opening and closing trade minus trading fees
    # assert portfolio.get_cash() == pytest.approx(10000 - (close_trade.get_planned_value() - open_trade.get_planned_value()) - position.get_total_lp_fees_paid())
    assert portfolio.get_net_asset_value(include_interest=True) == pytest.approx(9321.153949134565)

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
    assert balances[usdc.address] == pytest.approx(Decimal(9321.153949134565))


def test_backtest_short_take_profit_triggered(persistent_test_client: Client, strategy_universe):
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
        trade_pair = strategy_universe.data_universe.pairs.get_single()

        cash = state.portfolio.get_cash()
        position_size = cash * 0.8

        position_manager = PositionManager(timestamp, strategy_universe, state, pricing_model)

        trades = []

        if not position_manager.is_any_open() and timestamp == datetime.datetime(2023, 1, 1):
            trades += position_manager.open_short(trade_pair, position_size, leverage=4, take_profit_pct=1.02)

        return trades

    state, strategy_universe, debug_dump = run_backtest_inline(
        start_at=start_at,
        end_at=end_at,
        client=persistent_test_client,
        cycle_duration=CycleDuration.cycle_1d,
        decide_trades=decide_trades,
        universe=strategy_universe,
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
    assert position.liquidation_price == pytest.approx(Decimal(1912.499999999999950039963892))

    # assert position.get_value_at_open() == capital
    assert position.get_collateral() == 0
    assert position.get_borrowed() == 0
    assert position.get_accrued_interest() == pytest.approx(0)
    assert position.get_unrealised_profit_usd() == 0
    assert position.get_realised_profit_usd() == pytest.approx(981.5445220968387)
    assert position.get_claimed_interest() == pytest.approx(0)
    assert position.get_repaid_interest() == pytest.approx(0)
    assert position.get_value() == pytest.approx(0)

    # Check opening trade looks good
    assert len(position.trades) == 2
    open_trade = position.get_first_trade()
    assert open_trade.opened_at == datetime.datetime(2023, 1, 1)
    assert open_trade.planned_price == pytest.approx(1794.6)  # ETH opening value
    assert open_trade.get_planned_value() == pytest.approx(31904)
    assert open_trade.planned_quantity == pytest.approx(Decimal(-17.77777777777777777777777778))

    # Check closing trade looks good
    close_trade = position.get_last_trade()
    assert close_trade.opened_at == datetime.datetime(2023, 1, 2)
    assert close_trade.planned_price == pytest.approx(1739.3881206320527)  # ETH current value
    assert close_trade.get_planned_value() == pytest.approx(30922.45547790316)
    assert close_trade.planned_quantity == pytest.approx(Decimal(17.77777777777777777777777778))

    # Check that the portfolio looks good
    assert portfolio.get_cash() == pytest.approx(10981.266217492795)
    # loss should be equal to the difference between the opening and closing trade minus trading fees
    # assert portfolio.get_cash() == pytest.approx(10000 + (open_trade.get_planned_value() - close_trade.get_planned_value()) - position.get_total_lp_fees_paid())
    assert portfolio.get_net_asset_value(include_interest=True) == pytest.approx(10981.266217492794)

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
    assert balances[usdc.address] == pytest.approx(Decimal(10981.266217492794))


def test_backtest_short_trailing_stop_loss_triggered(persistent_test_client: Client, strategy_universe):
    """Run the strategy backtest using inline decide_trades function.

    - Open short position, set a 2% trailing stoploss
    - ETH price goes 1800 -> 1636 -> 1679
    - Position should be closed automatically with a profit since new stoploss is lowered than original opening price
    """

    def decide_trades(
        timestamp: pd.Timestamp,
        strategy_universe: TradingStrategyUniverse,
        state: State,
        pricing_model: PricingModel,
        cycle_debug_data: Dict
    ) -> List[TradeExecution]:
        """A simple strategy that opens a single 4x short position."""
        trade_pair = strategy_universe.data_universe.pairs.get_single()

        cash = state.portfolio.get_cash()
        position_size = cash * 0.8

        position_manager = PositionManager(timestamp, strategy_universe, state, pricing_model)

        trades = []

        if not position_manager.is_any_open() and timestamp == datetime.datetime(2023, 1, 1):
            trades += position_manager.open_short(
                trade_pair,
                position_size,
                leverage=4,
                trailing_stop_loss_pct=0.98,
            )
        else:
            if timestamp == datetime.datetime(2023, 1, 2):
                position = position_manager.get_current_position()
                assert position.stop_loss == pytest.approx(1768.8692752190368)

        return trades

    state, strategy_universe, debug_dump = run_backtest_inline(
        start_at=start_at,
        end_at=datetime.datetime(2023, 1, 10),
        client=persistent_test_client,
        cycle_duration=CycleDuration.cycle_1d,
        decide_trades=decide_trades,
        universe=strategy_universe,
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

    assert position.liquidation_price == pytest.approx(Decimal(1912.499999999999950039963892))
    assert position.stop_loss == pytest.approx(1669.082747543819)
    assert position.get_realised_profit_percent() == pytest.approx(0.06124905034476604)
    assert position.get_realised_profit_percent() == position.get_total_profit_percent()

    assert portfolio.get_cash() == pytest.approx(11952.745496750647)
    
    analysis = build_trade_analysis(state.portfolio)
    summary = analysis.calculate_summary_statistics(state=state, time_bucket=TimeBucket.d1)
    
    assert summary.return_percent == pytest.approx(0.1952745496750649)
    assert summary.compounding_returns.iloc[-1] == pytest.approx(summary.return_percent, abs=1e-3)  # TODO make match more precisely
