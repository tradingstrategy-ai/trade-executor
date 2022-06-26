"""Visualisation tests."""
import datetime
from decimal import Decimal

import pandas as pd
import pytest

from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.reserve import ReservePosition
from tradeexecutor.state.state import State
from tradeexecutor.state.visualisation import PlotKind
from tradeexecutor.testing.synthetic_price_data import generate_ohlcv_candles
from tradeexecutor.testing.dummy_trader import DummyTestTrader
from tradeexecutor.visual.single_pair import visualise_single_pair, export_plot_as_dataframe
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket


@pytest.fixture
def mock_exchange_address() -> str:
    """Mock some assets"""
    return "0x1"


@pytest.fixture
def usdc() -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(ChainId.ethereum.value, "0x0", "USDC", 6)


@pytest.fixture
def weth() -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(ChainId.ethereum.value, "0x1", "WETH", 18)


@pytest.fixture
def weth_usdc(mock_exchange_address, usdc, weth) -> TradingPairIdentifier:
    """Mock some assets"""
    return TradingPairIdentifier(weth, usdc, "0x4", mock_exchange_address, internal_id=555)


def test_synthetic_candles_timezone(usdc, weth, weth_usdc):
    """Check synthetic candle data for timezone issues."""
    start_date = datetime.datetime(2021, 1, 1)
    end_date = datetime.datetime(2021, 3, 1)
    candles = generate_ohlcv_candles(TimeBucket.d1, start_date, end_date, pair_id=weth_usdc.internal_id)
    assert candles.iloc[0]["timestamp"] == pd.Timestamp("2021-01-01 00:00:00")


def test_visualise_trades_with_indicator(usdc, weth, weth_usdc):
    """Do a single token purchase."""
    state = State()

    start_date = datetime.datetime(2021, 1, 1)
    end_date = datetime.datetime(2021, 3, 1)

    # Start with 100k USD
    state.update_reserves([ReservePosition(usdc, Decimal(100_000), start_date, 1.0, 0)])

    # Generate candles for pair_id = 1
    candles = generate_ohlcv_candles(TimeBucket.d1, start_date, end_date, pair_id=weth_usdc.internal_id)
    candle_universe = GroupedCandleUniverse.create_from_single_pair_dataframe(candles)

    trader = DummyTestTrader(state)

    # Day 1
    # Buy 10 ETH at 1700 USD/ETH
    trader.time_travel(start_date)
    pos, trade = trader.buy_with_price_data(weth_usdc, 10, candle_universe)
    start_q = Decimal('9.899999999999999911182158030')
    assert trade.is_buy()
    assert pos.get_quantity() == pytest.approx(start_q)
    assert pos.get_opening_price() == pytest.approx(1716.8437083008298)
    state.visualisation.plot_indicator(trader.ts, "Test indicator", PlotKind.technical_indicator_on_price, 1700)

    sell_q_1 = start_q / 2
    sell_q_2 = start_q - sell_q_1

    # Day 2
    # Sell 5 ETH at 1800 USD/ETH
    trader.time_travel(datetime.datetime(2021, 2, 1))
    pos, trade = trader.sell_with_price_data(weth_usdc, sell_q_1, candle_universe)
    assert trade.is_sell()
    assert pos.get_quantity() == pytest.approx(Decimal('4.949999999999999955591079015'))
    state.visualisation.plot_indicator(trader.ts, "Test indicator", PlotKind.technical_indicator_on_price, 1700)

    # Day 2
    # Sell 5 ETH at 1800 USD/ETH
    trader.time_travel(end_date)
    pos, trade = trader.sell_with_price_data(weth_usdc, sell_q_2, candle_universe)
    assert pos.get_quantity() == 0
    state.visualisation.plot_indicator(trader.ts, "Test indicator", PlotKind.technical_indicator_on_price, 1700)

    assert len(list(state.portfolio.get_all_trades())) == 3
    assert len(state.portfolio.open_positions) == 0
    assert len(state.portfolio.closed_positions) == 1

    #
    # Now visualise the events
    #
    fig = visualise_single_pair(state, candle_universe)

    # List of candles, markers 1, markers
    data = fig.to_dict()["data"]
    assert len(data) == 4
    assert data[1]["name"] == "Test indicator"
    assert data[2]["name"] == "Buys"
    assert data[3]["name"] == "Sells"

    # Check test indicator data
    # that we have proper timestamps
    plot = state.visualisation.plots["Test indicator"]
    df = export_plot_as_dataframe(plot)
    ts = df.iloc[0]["timestamp"]
    ts = ts.replace(minute=0, second=0)
    assert ts == pd.Timestamp("2021-1-1 00:00")

    return fig



