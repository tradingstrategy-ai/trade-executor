"""Visualisation tests."""
import datetime
from decimal import Decimal

import pandas as pd
import pytest
import random
import os

from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.reserve import ReservePosition
from tradeexecutor.state.state import State
from tradeexecutor.state.validator import validate_state_serialisation
from tradeexecutor.state.visualisation import PlotKind
from tradeexecutor.statistics.core import calculate_statistics, update_statistics
from tradeexecutor.strategy.execution_context import ExecutionMode, unit_test_execution_context
from tradeexecutor.testing.synthetic_price_data import generate_ohlcv_candles
from tradeexecutor.testing.unit_test_trader import UnitTestTrader
from tradeexecutor.visual.single_pair import visualise_single_pair, visualise_single_pair_positions_with_duration_and_slippage
from tradeexecutor.visual.technical_indicator import export_plot_as_dataframe
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.charting.candle_chart import VolumeBarMode

from tradeexecutor.visual.utils import export_trades_as_dataframe


@pytest.fixture(scope = "module")
def mock_exchange_address() -> str:
    """Mock some assets"""
    return "0x1"


@pytest.fixture(scope = "module")
def usdc() -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(ChainId.ethereum.value, "0x0", "USDC", 6)


@pytest.fixture(scope = "module")
def weth() -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(ChainId.ethereum.value, "0x1", "WETH", 18)


@pytest.fixture(scope = "module")
def weth_usdc(mock_exchange_address, usdc, weth) -> TradingPairIdentifier:
    """Mock some assets"""
    return TradingPairIdentifier(weth, usdc, "0x4", mock_exchange_address, internal_id=555)


@pytest.fixture(scope = "module")
def state_and_candles(usdc, weth, weth_usdc) -> tuple[State, pd.DataFrame]:
    state = State(name="Visualisation test")

    start_date = datetime.datetime(2021, 1, 1)
    end_date = datetime.datetime(2021, 3, 1)

    # Start with 100k USD
    state.update_reserves([ReservePosition(usdc, Decimal(100_000), start_date, 1.0, 0)])

    # Generate candles for pair_id = 1
    time_bucket = TimeBucket.d1
    candles = generate_ohlcv_candles(time_bucket, start_date, end_date, pair_id=weth_usdc.internal_id)
    candle_universe = GroupedCandleUniverse.create_from_single_pair_dataframe(candles, time_bucket)

    trader = UnitTestTrader(state)

    # Day 1
    # Buy 10 ETH at 1700 USD/ETH
    trader.time_travel(start_date)
    pos, trade = trader.buy_with_price_data(weth_usdc, 10, candle_universe)
    start_q = Decimal('9.899999999999999911182158030')
    assert trade.is_buy()
    assert pos.get_quantity() == pytest.approx(start_q)
    assert pos.get_opening_price() == pytest.approx(1716.8437083008298)
    
    state.visualisation.plot_indicator(trader.ts, "Test indicator", PlotKind.technical_indicator_on_price, 1700)
    
    state.visualisation.plot_indicator(trader.ts, "random 1", PlotKind.technical_indicator_detached, 1000, colour="green")
    
    state.visualisation.plot_indicator(trader.ts, "random 2", PlotKind.technical_indicator_detached, 1100, colour="yellow")
    
    state.visualisation.plot_indicator(trader.ts, "random 3", PlotKind.technical_indicator_overlay_on_detached, 1200, colour="green", detached_overlay_name="random 2")
    
    state.visualisation.plot_indicator(trader.ts, "random 4", PlotKind.technical_indicator_overlay_on_detached, 1300, colour="blue", detached_overlay_name="random 2")

    sell_q_1 = start_q / 2
    sell_q_2 = start_q - sell_q_1

    # Day 2
    # Sell 5 ETH at 1800 USD/ETH
    trader.time_travel(datetime.datetime(2021, 2, 1))
    pos, trade = trader.sell_with_price_data(weth_usdc, sell_q_1, candle_universe)
    assert trade.is_sell()
    assert pos.get_quantity() == pytest.approx(Decimal('4.949999999999999955591079015'))
    
    state.visualisation.plot_indicator(trader.ts, "Test indicator", PlotKind.technical_indicator_on_price, 1700, colour="aqua")
    
    state.visualisation.plot_indicator(trader.ts, "random 1", PlotKind.technical_indicator_detached, 1000, colour="green")
    
    state.visualisation.plot_indicator(trader.ts, "random 2", PlotKind.technical_indicator_detached, 1100, colour="yellow")
    
    state.visualisation.plot_indicator(trader.ts, "random 3", PlotKind.technical_indicator_overlay_on_detached, 1200, colour="green", detached_overlay_name="random 2")
    
    state.visualisation.plot_indicator(trader.ts, "random 4", PlotKind.technical_indicator_overlay_on_detached, 1300, colour="blue", detached_overlay_name="random 2")

    # Day 2
    # Sell 5 ETH at 1800 USD/ETH
    trader.time_travel(end_date)
    pos, trade = trader.sell_with_price_data(weth_usdc, sell_q_2, candle_universe)
    assert pos.get_quantity() == 0
    
    state.visualisation.plot_indicator(trader.ts, "Test indicator", PlotKind.technical_indicator_on_price, 1700, colour="azure")
    
    state.visualisation.plot_indicator(trader.ts, "random 1", PlotKind.technical_indicator_detached, 1200, colour="green")
    
    state.visualisation.plot_indicator(trader.ts, "random 2", PlotKind.technical_indicator_detached, 1100, colour="yellow")
    
    state.visualisation.plot_indicator(trader.ts, "random 3", PlotKind.technical_indicator_overlay_on_detached, 1400, colour="green", detached_overlay_name="random 2")
    
    state.visualisation.plot_indicator(trader.ts, "random 4", PlotKind.technical_indicator_overlay_on_detached, 1500, colour="blue", detached_overlay_name="random 2")

    return state, candles


def test_synthetic_candles_timezone(usdc, weth, weth_usdc):
    """Check synthetic candle data for timezone issues."""
    start_date = datetime.datetime(2021, 1, 1)
    end_date = datetime.datetime(2021, 3, 1)
    candles = generate_ohlcv_candles(TimeBucket.d1, start_date, end_date, pair_id=weth_usdc.internal_id)
    assert candles.iloc[0]["timestamp"] == pd.Timestamp("2021-01-01 00:00:00")
    

def test_visualise_trades_with_indicator(state_and_candles: tuple[State, pd.DataFrame]):
    """Do a single token purchase.
    
    Uses default VolumeBarMode.overlay"""

    state, candles = state_and_candles

    # TODO: Test does not correctly handle volume data
    # manually fix later
    candles = candles.drop(labels=["volume"], axis="columns")

    candle_universe = GroupedCandleUniverse.create_from_single_pair_dataframe(candles)
    
    validate_state_serialisation(state)

    assert len(list(state.portfolio.get_all_trades())) == 3
    assert len(state.portfolio.open_positions) == 0
    assert len(state.portfolio.closed_positions) == 1

    # Some legacy date range logic checks
    pair_id = candles["pair_id"].unique()[0]
    start_at, end_at = state.get_strategy_start_and_end()
    #assert start_at is None
    #assert end_at is None
    #start_at = candle_universe.get_timestamp_range()[0]
    #end_at = candle_universe.get_timestamp_range()[1]
    assert start_at == pd.Timestamp('2021-01-01 00:00:00')
    assert end_at == pd.Timestamp('2021-03-01 00:00:03')
    trades = export_trades_as_dataframe(state.portfolio, pair_id, start_at, end_at)
    assert len(trades) == 3


    #
    # Now visualise the events
    #
    fig = visualise_single_pair(
        state,
        unit_test_execution_context,
        candle_universe,
    )

    # 3 distinct plot grids
    assert len(fig._grid_ref) == 3
    
    # check the main title
    assert fig.layout.title.text == "Visualisation test"
    
    # check subplot titles
    subplot_titles = [annotation['text'] for annotation in fig['layout']['annotations']]
    assert subplot_titles[0] == "random 1"
    assert subplot_titles[1] == "random 2<br> + random 3<br> + random 4"
    
    # List of candles, indicators, and markers
    data = fig.to_dict()["data"]
    assert len(data) == 8
    assert data[1]["name"] == "Test indicator"
    assert data[2]["name"] == "random 1"
    assert data[3]["name"] == "random 2"
    assert data[4]["name"] == "random 3"
    assert data[5]["name"] == "random 4"
    assert data[6]["name"] == "Buy"
    assert data[7]["name"] == "Sell"

    # check dates
    assert data[0]['x'][0] == datetime.datetime(2021, 1, 1, 0, 0)
    assert data[0]['x'][-1] == datetime.datetime(2021, 2, 28, 0, 0)

    # Check test indicator data
    # that we have proper timestamps
    plot = state.visualisation.plots["Test indicator"]
    df = export_plot_as_dataframe(plot)
    ts = df.iloc[0]["timestamp"]
    ts = ts.replace(minute=0, second=0)
    assert ts == pd.Timestamp("2021-1-1 00:00")

    fig2 = visualise_single_pair(
        state,
        unit_test_execution_context,
        candle_universe,
        detached_indicators=False
    )

    assert len(fig2.data) == 4
    assert len(fig2._grid_ref) == 1

    fig3 = visualise_single_pair(
        state,
        unit_test_execution_context,
        candle_universe,
        detached_indicators=False,
        volume_bar_mode=VolumeBarMode.separate,
    )

    assert len(fig3.data) == 4
    assert len(fig3._grid_ref) == 2

    fig4 = visualise_single_pair(
        state,
        unit_test_execution_context,
        candle_universe,
        detached_indicators=True,
        technical_indicators=False,
    )

    assert len(fig4.data) == 3
    assert len(fig4._grid_ref) == 1

    fig5 = visualise_single_pair(
        state,
        unit_test_execution_context,
        candle_universe,
        detached_indicators=True,
        technical_indicators=False,
        volume_bar_mode=VolumeBarMode.separate,
    )

    assert len(fig5.data) == 3
    assert len(fig5._grid_ref) == 2


def test_visualise_trades_separate_volume(
    state_and_candles: tuple[State, pd.DataFrame]
):
    """Do a single token purchase.
    
    Uses VolumeBarMode.separate"""

    state, candles = state_and_candles

    # TODO: Test does not correctly handle volume data
    # manually fix later
    candles = candles.drop(labels=["volume"], axis="columns")

    candle_universe = GroupedCandleUniverse.create_from_single_pair_dataframe(candles)
    
    validate_state_serialisation(state)

    assert len(list(state.portfolio.get_all_trades())) == 3
    assert len(state.portfolio.open_positions) == 0
    assert len(state.portfolio.closed_positions) == 1

    fig = visualise_single_pair(
        state,
        unit_test_execution_context,
        candle_universe, 
        volume_bar_mode=VolumeBarMode.separate,
        relative_sizing=[1, 0.2, 0.2, 0.5],
    )

    # 4 distinct plot grids
    assert len(fig._grid_ref) == 4
    
    # check the main title
    assert fig.layout.title.text == "Visualisation test"
    
    # check subplot titles
    subplot_titles = [annotation['text'] for annotation in fig['layout']['annotations']]
    assert subplot_titles[0] == "Volume USD"
    assert subplot_titles[1] == "random 1"
    assert subplot_titles[2] == "random 2<br> + random 3<br> + random 4"
    
    # List of candles, indicators, and markers
    data = fig.to_dict()["data"]

    assert data[1]["name"] == "Test indicator"
    assert data[2]["name"] == "random 1"
    assert data[3]["name"] == "random 2"
    assert data[4]["name"] == "random 3"
    assert data[5]["name"] == "random 4"
    assert data[6]["name"] == "Buy"
    assert data[7]["name"] == "Sell"

    # check dates
    assert data[0]['x'][0] == datetime.datetime(2021, 1, 1, 0, 0)
    assert data[0]['x'][-1] == datetime.datetime(2021, 2, 28, 0, 0)

    # Check test indicator data
    # that we have proper timestamps
    plot = state.visualisation.plots["Test indicator"]
    df = export_plot_as_dataframe(plot)
    ts = df.iloc[0]["timestamp"]
    ts = ts.replace(minute=0, second=0)
    assert ts == pd.Timestamp("2021-1-1 00:00")


def test_visualise_trades_with_duration_and_slippage(
    weth_usdc, state_and_candles: tuple[State, pd.DataFrame]
):
    """Do a single token purchase.
    
    Uses VolumeBarMode.hidden"""
    
    state, candles = state_and_candles

    validate_state_serialisation(state)

    assert len(list(state.portfolio.get_all_trades())) == 3
    assert len(state.portfolio.open_positions) == 0
    assert len(state.portfolio.closed_positions) == 1
    
    #
    # Now visualise the events
    #
    fig = visualise_single_pair_positions_with_duration_and_slippage(
        state,
        unit_test_execution_context,
        candles,
    )

    # 3 distinct plot grids
    assert len(fig._grid_ref) == 3
    
    # check the main title
    assert fig.layout.title.text == "Visualisation test"
    
    # check subplot titles
    subplot_titles = [annotation['text'] for annotation in fig['layout']['annotations']]
    assert subplot_titles[0] == "random 1"
    assert subplot_titles[1] == "random 2<br> + random 3<br> + random 4"
    
    # List of candles, indicators, and markers
    data = fig.to_dict()["data"]
    assert len(data) == 9
    assert data[1]["name"] == "Test indicator"
    assert data[2]["name"] == "random 1"
    assert data[3]["name"] == "random 2"
    assert data[4]["name"] == "random 3"
    assert data[5]["name"] == "random 4"

    # check dates
    assert data[0]['x'][0] == datetime.datetime(2021, 1, 1, 0, 0)
    assert data[0]['x'][-1] == datetime.datetime(2021, 2, 28, 0, 0)

    # Check test indicator data
    # that we have proper timestamps
    plot = state.visualisation.plots["Test indicator"]
    df = export_plot_as_dataframe(plot)
    ts = df.iloc[0]["timestamp"]
    ts = ts.replace(minute=0, second=0)
    assert ts == pd.Timestamp("2021-1-1 00:00")


@pytest.mark.parametrize("relative_sizing, vertical_spacing, subplot_font_size, title, volume_axis_name, volume_bar_mode", [
    ([1, 3, 0.2], 0.2, 20, None, None, VolumeBarMode.hidden),
    ([1, 0.2, 0.2, 0.5], 0.1, 10, "Test title", "Volume USD", VolumeBarMode.separate),
    ([100, 0.2, 0.2], 0.3, 250, "Test title 2", None, VolumeBarMode.overlay),
    (None, None, None, None, None, VolumeBarMode.hidden),
])
def test_visualise_single_pair_no_error(
    state_and_candles: tuple[State, pd.DataFrame],
    relative_sizing,
    vertical_spacing,
    subplot_font_size,
    title,
    volume_axis_name,
    volume_bar_mode,
):
    """Test various arguments for visualise_single_pair and visualise_single_pair_positions_with_duration_and_slippage.
    
    These arguments should not raise error"""

    state, candles = state_and_candles
    candle_universe = GroupedCandleUniverse.create_from_single_pair_dataframe(candles)
    
    validate_state_serialisation(state)

    assert len(list(state.portfolio.get_all_trades())) == 3
    assert len(state.portfolio.open_positions) == 0
    assert len(state.portfolio.closed_positions) == 1
    
    # check no error with different arguments
    fig = visualise_single_pair(
        state,
        unit_test_execution_context,
        candle_universe,
        relative_sizing=relative_sizing,
        vertical_spacing=vertical_spacing, 
        subplot_font_size=subplot_font_size,
        title=title,
        volume_axis_name=volume_axis_name,
        volume_bar_mode=volume_bar_mode,
    )


@pytest.mark.parametrize("relative_sizing, vertical_spacing, subplot_font_size, title, volume_axis_name, volume_bar_mode", [
    ([1, 3, 0.2], 0.2, 20, None, None, VolumeBarMode.separate),
    ([1, 0.2, 0.2, 0.5], 0.1, 10, "Test title 0", "Volume USD", VolumeBarMode.overlay),
    ([100, 0.2, 0.2, 0.5], 0.3, 250, "Test title 2", "Volume USD 2", VolumeBarMode.hidden),
])
def test_visualise_single_pair_with_error(
    state_and_candles: tuple[State, pd.DataFrame],
    relative_sizing,
    vertical_spacing,
    subplot_font_size,
    title,
    volume_axis_name,
    volume_bar_mode,
):
    """Test various arguments for visualise_single_pair and visualise_single_pair_positions_with_duration_and_slippage.
    
    These arguments should not raise error"""

    state, candles = state_and_candles
    candle_universe = GroupedCandleUniverse.create_from_single_pair_dataframe(candles)
    
    # check no error with different arguments
    with pytest.raises(AssertionError):
        fig = visualise_single_pair(
            state,
            unit_test_execution_context,
            candle_universe,
            relative_sizing=relative_sizing,
            vertical_spacing=vertical_spacing, 
            subplot_font_size=subplot_font_size,
            title=title,
            volume_axis_name=volume_axis_name,
            volume_bar_mode=volume_bar_mode,
        )


@pytest.mark.parametrize("relative_sizing, vertical_spacing, subplot_font_size, title", [
    ([1, 3, 0.2], 0.2, 20, None),
    ([1, 0.2, 0.2,], 0.1, 10, "Test title"),
    ([100, 0.2, 0.2], 0.3, 250, "Test title 2"),
    (None, None, None, None),
])
def test_visualise_single_pair_with_duration_and_slippage_no_error(
    state_and_candles: tuple[State, pd.DataFrame],
    relative_sizing,
    vertical_spacing,
    subplot_font_size,
    title,
):
    """Test various arguments for visualise_single_pair and visualise_single_pair_positions_with_duration_and_slippage.
    
    These arguments should not raise error"""

    state, candles = state_and_candles
    
    assert len(list(state.portfolio.get_all_trades())) == 3
    assert len(state.portfolio.open_positions) == 0
    assert len(state.portfolio.closed_positions) == 1
    
    fig = visualise_single_pair_positions_with_duration_and_slippage(
        state=state,
        execution_context=unit_test_execution_context,
        candles=candles,
        relative_sizing=relative_sizing,
        vertical_spacing=vertical_spacing, 
        subplot_font_size=subplot_font_size,
        title=title,
    )
        

@pytest.mark.parametrize("relative_sizing, vertical_spacing, subplot_font_size, title", [
    ([1, 3], None, None, None),
    ([1, 0.2, 0.2, 0.5], 0.1, 10, "Test title 0"),
    ([100, 0.2, 0.2, 0.5], 0.3, 250, "Test title 2"),
])  
def test_visualise_single_pair_with_duration_and_slippage_with_error(
    state_and_candles: tuple[State, pd.DataFrame],
    relative_sizing,
    vertical_spacing,
    subplot_font_size,
    title
):
    state, candles = state_and_candles
    
    with pytest.raises(AssertionError):
        fig = visualise_single_pair_positions_with_duration_and_slippage(
            state=state,
            execution_context=unit_test_execution_context,
            candles=candles,
            relative_sizing=relative_sizing,
            vertical_spacing=vertical_spacing, 
            subplot_font_size=subplot_font_size,
            title=title,
        ) 
    