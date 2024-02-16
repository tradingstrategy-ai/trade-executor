"""Compare portfolio performance against other strategies."""
import datetime
import warnings
from typing import Optional, List, Union, Collection, Dict

import plotly.graph_objects as go
import pandas as pd

from tradeexecutor.analysis.trade_analyser import build_trade_analysis
from tradeexecutor.state.statistics import PortfolioStatistics
from tradeexecutor.state.visualisation import Plot
from tradeexecutor.state.state import State
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.visual.technical_indicator import visualise_technical_indicator
from tradeexecutor.visual.equity_curve import calculate_long_compounding_realised_trading_profitability, calculate_short_compounding_realised_trading_profitability, calculate_compounding_realised_trading_profitability
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.pair import HumanReadableTradingPairDescription
from tradingstrategy.types import USDollarAmount


def visualise_portfolio_equity_curve(
        name: str,
        portfolio_statistics: List[PortfolioStatistics],
        colour="#008800") -> go.Scatter:
    """Draw portfolio performance."""

    plot = []
    for s in portfolio_statistics:
        plot.append({
            "timestamp": pd.Timestamp(s.calculated_at),
            "value": s.total_equity,
        })

    df = pd.DataFrame(plot)
    df.set_index("timestamp", inplace=True)

    return go.Scatter(
        x=df.index,
        y=df["value"],
        mode="lines",
        name=name,
        line=dict(color=colour),
    )


def visualise_all_cash(
        start_at: pd.Timestamp,
        end_at: pd.Timestamp,
        all_cash: float,
        colour="#000088") -> go.Scatter:
    """Draw portfolio performance."""

    plot = []
    plot.append({
        "timestamp": start_at,
        "value": all_cash,
    })

    plot.append({
        "timestamp": end_at,
        "value": all_cash,
    })

    df = pd.DataFrame(plot)
    df.set_index("timestamp", inplace=True)

    return go.Scatter(
        x=df.index,
        y=df["value"],
        mode="lines",
        name="Hold cash",
        line=dict(color=colour),
    )


def visualise_buy_and_hold(
        buy_and_hold_asset_name: str,
        price_series: pd.Series,
        all_cash: float,
        colour="#880000") -> go.Scatter:
    """Draw portfolio performance."""

    # Whatever we bought at the start
    initial_inventory = all_cash / float(price_series.iloc[0])
    series = price_series * initial_inventory

    return go.Scatter(
        x=series.index,
        y=series,
        mode="lines",
        name=f"Buy and hold {buy_and_hold_asset_name}",
        line=dict(color=colour),
    )


def visualise_equity_curve_benchmark(
    name: Optional[str] = None,
    portfolio_statistics: Optional[List[PortfolioStatistics]] = None,
    all_cash: Optional[float] = None,
    buy_and_hold_asset_name: Optional[str] = None,
    buy_and_hold_price_series: Optional[pd.Series] = None,
    benchmark_indexes: pd.DataFrame = None,
    additional_indicators: Collection[Plot] = None,
    height=1200,
    start_at: Optional[Union[pd.Timestamp, datetime.datetime]] = None,
    end_at: Optional[Union[pd.Timestamp, datetime.datetime]] = None,
) -> go.Figure:
    """Visualise strategy performance against benchmarks.

    - Live or backtested strategies

    - Benchmark against buy and hold of various assets

    - Benchmark against hold all cash

    Example for a single trading pair strategy:

    .. code-block:: python

        from tradeexecutor.visual.benchmark import visualise_benchmark

        traded_pair = universe.universe.pairs.get_single()

        fig = visualise_benchmark(
            state.name,
            portfolio_statistics=state.stats.portfolio,
            all_cash=state.portfolio.get_initial_deposit(),
            buy_and_hold_asset_name=traded_pair.base_token_symbol,
            buy_and_hold_price_series=universe.universe.candles.get_single_pair_data()["close"],
            height=800
        )

        fig.show()

    Example how to benchmark a strategy against buy-and-hold BTC and ETH:

    .. code-block:: python

        from tradeexecutor.visual.benchmark import visualise_benchmark

        # List of pair descriptions we used to look up pair metadata
        our_pairs = [
            (ChainId.centralised_exchange, "binance", "BTC", "USDT"),
            (ChainId.centralised_exchange, "binance", "ETH", "USDT"),
        ]

        btc_pair = strategy_universe.data_universe.pairs.get_pair_by_human_description(our_pairs[0])
        eth_pair = strategy_universe.data_universe.pairs.get_pair_by_human_description(our_pairs[1])

        benchmark_indexes = pd.DataFrame({
            "BTC": strategy_universe.data_universe.candles.get_candles_by_pair(btc_pair)["close"],
            "ETH": strategy_universe.data_universe.candles.get_candles_by_pair(eth_pair)["close"],
        })
        benchmark_indexes["BTC"].attrs = {"colour": "orange"}
        benchmark_indexes["ETH"].attrs = {"colour": "blue"}

        fig = visualise_benchmark(
            name=state.name,
            portfolio_statistics=state.stats.portfolio,
            all_cash=state.portfolio.get_initial_deposit(),
            benchmark_indexes=benchmark_indexes,
        )

        fig.show()

    Another example:

    .. code-block:: python

        from tradeexecutor.visual.benchmark import visualise_benchmark

        TRADING_PAIRS = [
            (ChainId.avalanche, "trader-joe", "WAVAX", "USDC"), # Avax
            (ChainId.polygon, "quickswap", "WMATIC", "USDC"),  # Matic
            (ChainId.ethereum, "uniswap-v2", "WETH", "USDC"),  # Eth
            (ChainId.ethereum, "uniswap-v2", "WBTC", "USDC"),  # Btc
        ]

        # Benchmark against all of our assets
        benchmarks = pd.DataFrame()
        for pair_description in TRADING_PAIRS:
            token_symbol = pair_description[2]
            pair = universe.get_pair_by_human_description(pair_description)
            benchmarks[token_symbol] = universe.universe.candles.get_candles_by_pair(pair.internal_id)["close"]

        fig = visualise_benchmark(
            "Bollinger bands example strategy",
            portfolio_statistics=state.stats.portfolio,
            all_cash=state.portfolio.get_initial_deposit(),
            benchmark_indexes=benchmarks,
            start_at=START_AT,
            end_at=END_AT,
            height=800
        )

        fig.show()

    :param portfolio_statistics:
        Portfolio performance record.

    :param all_cash:
        Set a linear line of just holding X amount

    :param buy_and_hold_asset_name:

        Visualise holding all_cash amount in the asset,
        bought at the start.
        This is basically price * all_cash.

        .. note ::

            This is a legacy argument. Use `benchmark_indexes` instead.

    :param buy_and_hold_price_series:

        Visualise holding all_cash amount in the asset,
        bought at the start.
        This is basically price * all_cash.

        .. note ::

            This is a legacy argument. Use `benchmark_indexes` instead.

    :param benchmark_indexes:
        List of other asset price series displayed on the timeline besides equity curve.

        DataFrame containing multiple series.

        - Asset name is the series name.
        - Setting `colour` for `pd.Series.attrs` allows you to override the colour of the index

    :param height:
        Chart height in pixels

    :param start_at:
        When the backtest started

    :param end_at:
        When the backtest ended

    :param additional_indicators:
        Additional technical indicators drawn on this chart.

        List of indicator names.

        The indicators must be plotted earlier using `state.visualisation.plot_indicator()`.

        **Note**: Currently not very useful due to Y axis scale

    """

    fig = go.Figure()

    assert portfolio_statistics

    if isinstance(start_at, datetime.datetime):
        start_at = pd.Timestamp(start_at)

    if isinstance(end_at, datetime.datetime):
        end_at = pd.Timestamp(end_at)

    if start_at is not None:
        assert isinstance(start_at, pd.Timestamp)

    if end_at is not None:
        assert isinstance(end_at, pd.Timestamp)

    first_portfolio_timestamp = pd.Timestamp(portfolio_statistics[0].calculated_at)
    last_portfolio_timestamp = pd.Timestamp(portfolio_statistics[-1].calculated_at)

    if not start_at or start_at < first_portfolio_timestamp:
        start_at = first_portfolio_timestamp

    if not end_at or end_at > last_portfolio_timestamp:
        end_at = last_portfolio_timestamp

    scatter = visualise_portfolio_equity_curve(name, portfolio_statistics)
    fig.add_trace(scatter)

    if all_cash:
        scatter = visualise_all_cash(start_at, end_at, all_cash)
        fig.add_trace(scatter)

    if benchmark_indexes is None:
        benchmark_indexes = pd.DataFrame()

    # Backwards compatible arguments
    if buy_and_hold_price_series is not None:
        benchmark_indexes[buy_and_hold_asset_name] = buy_and_hold_price_series

    # Plot all benchmark series
    for benchmark_name in benchmark_indexes:
        buy_and_hold_price_series = benchmark_indexes[benchmark_name]
        # Clip to the backtest time frame
        buy_and_hold_price_series = buy_and_hold_price_series[start_at:end_at]
        colour = buy_and_hold_price_series.attrs.get("colour")
        scatter = visualise_buy_and_hold(
            benchmark_name,
            buy_and_hold_price_series,
            all_cash,
            colour=colour,
        )
        fig.add_trace(scatter)

    if additional_indicators:
        for plot in additional_indicators:
            scatter = visualise_technical_indicator(plot, start_at, end_at)
            fig.add_trace(scatter)

    if name:
        fig.update_layout(title=f"{name}", height=height)
    else:
        fig.update_layout(title=f"Portfolio value", height=height)

    fig.update_yaxes(title="Value $", showgrid=False)

    fig.update_xaxes(rangeslider={"visible": False})

    # Move legend to the bottom so we have more space for
    # time axis in narrow notebook views
    # https://plotly.com/python/legend/
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))

    return fig


def visualise_benchmark(*args, **kwargs) -> go.Figure:
    warnings.warn('This function is deprecated. Use visualise_equity_curve_benchmark instead', DeprecationWarning, stacklevel=2)
    return visualise_equity_curve_benchmark(*args, **kwargs)



def create_benchmark_equity_curves(
    strategy_universe: TradingStrategyUniverse,
    pairs: Dict[str, HumanReadableTradingPairDescription],
    initial_cash: USDollarAmount,
    custom_colours={"BTC": "orange", "ETH": "blue", "All cash": "black"}
) -> pd.DataFrame:
    """Create data series of different buy-and-hold benchmarks.

    - Create different benchmark indexes to compare y our backtest results against

    - Has default colours set for `BTC` and `ETH` pair labels

    See also

    - Output be given e.g. to :py:func:`tradeexecutor.analysis.grid_search.visualise_grid_search_equity_curves`

    Example:

    .. code-block:: python

        from tradeexecutor.analysis.grid_search import visualise_grid_search_equity_curves
        from tradeexecutor.visual.benchmark import create_benchmark_equity_curves

        # List of pair descriptions we used to look up pair metadata
        our_pairs = [
            (ChainId.centralised_exchange, "binance", "BTC", "USDT"),
            (ChainId.centralised_exchange, "binance", "ETH", "USDT"),
        ]

        benchmark_indexes = create_benchmark_equity_curves(
            strategy_universe,
            {"BTC": our_pairs[0], "ETH": our_pairs[1]},
            initial_cash=StrategyParameters.initial_cash,
        )

        fig = visualise_grid_search_equity_curves(
            grid_search_results,
            benchmark_indexes=benchmark_indexes,
        )
        fig.show()

    :param strategy_universe:
        Strategy universe from where we

    :param pairs:
        Trading pairs benchmarked.

        In a format `short label` : `pair description`.

    :param initial_cash:
        The value for all cash benchmark and the initial backtest deposit.

        All cash is that you would just sit on the top of the cash pile
        since start of the backtest.

    :param custom_colours:
        Apply these colours on the benchmark series.

        Label -> color mappings.

    :return:
        Pandas DataFrame.

        DataFrame has series labelled "BTC", "ETH", "All cash", etc.

        DataFrame and its series' `attrs` contains colour information for well-known pairs.

    """

    returns = {}

    # Get close prices for all pairs
    for key, value in pairs.items():
        pair = strategy_universe.data_universe.pairs.get_pair_by_human_description(value)
        close_data = strategy_universe.data_universe.candles.get_candles_by_pair(pair)["close"]
        initial_inventory = initial_cash / float(close_data.iloc[0])
        series = close_data * initial_inventory
        returns[key] = series

    # Form all cash line
    if initial_cash is not None:
        assert len(returns) > 0
        assert type(initial_cash) in (int, float)
        first_close = next(iter(returns.values()))
        start_at = first_close.index[0]
        end_at = first_close.index[-1]
        idx = [start_at, end_at]
        values = [initial_cash, initial_cash]
        all_cash_series = pd.Series(values, idx)
        returns["All cash"] = all_cash_series

    # Wrap it up in a DataFrame
    benchmark_indexes = pd.DataFrame(returns)

    # Apply custom colors
    for symbol, colour in custom_colours.items():
        if symbol in benchmark_indexes.columns:
            benchmark_indexes[symbol].attrs = {"colour": colour}

    return benchmark_indexes



def visualise_benchmark(*args, **kwargs) -> go.Figure:
    warnings.warn('This function is deprecated. Use visualise_equity_curve_benchmark instead', DeprecationWarning, stacklevel=2)
    return visualise_equity_curve_benchmark(*args, **kwargs)



def create_benchmark_equity_curves(
    strategy_universe: TradingStrategyUniverse,
    pairs: Dict[str, HumanReadableTradingPairDescription],
    initial_cash: USDollarAmount,
    custom_colours={"BTC": "orange", "ETH": "blue", "All cash": "black"}
) -> pd.DataFrame:
    """Create data series of different buy-and-hold benchmarks.

    - Create different benchmark indexes to compare y our backtest results against

    - Has default colours set for `BTC` and `ETH` pair labels

    See also

    - Output be given e.g. to :py:func:`tradeexecutor.analysis.grid_search.visualise_grid_search_equity_curves`

    Example:

    .. code-block:: python

        from tradeexecutor.analysis.grid_search import visualise_grid_search_equity_curves
        from tradeexecutor.visual.benchmark import create_benchmark_equity_curves

        # List of pair descriptions we used to look up pair metadata
        our_pairs = [
            (ChainId.centralised_exchange, "binance", "BTC", "USDT"),
            (ChainId.centralised_exchange, "binance", "ETH", "USDT"),
        ]

        benchmark_indexes = create_benchmark_equity_curves(
            strategy_universe,
            {"BTC": our_pairs[0], "ETH": our_pairs[1]},
            initial_cash=StrategyParameters.initial_cash,
        )

        fig = visualise_grid_search_equity_curves(
            grid_search_results,
            benchmark_indexes=benchmark_indexes,
        )
        fig.show()

    :param strategy_universe:
        Strategy universe from where we

    :param pairs:
        Trading pairs benchmarked.

        In a format `short label` : `pair description`.

    :param initial_cash:
        The value for all cash benchmark and the initial backtest deposit.

        All cash is that you would just sit on the top of the cash pile
        since start of the backtest.

    :param custom_colours:
        Apply these colours on the benchmark series

    :return:
        Pandas DataFrame.

        DataFrame has series labelled "BTC", "ETH", "All cash", etc.

        DataFrame and its series' `attrs` contains colour information for well-known pairs.

    """

    returns = {}

    # Get close prices for all pairs
    for key, value in pairs.items():
        pair = strategy_universe.data_universe.pairs.get_pair_by_human_description(value)
        close_data = strategy_universe.data_universe.candles.get_candles_by_pair(pair)["close"]
        initial_inventory = initial_cash / float(close_data.iloc[0])
        series = close_data * initial_inventory
        returns[key] = series

    # Form all cash line
    if initial_cash is not None:
        assert len(returns) > 0
        assert type(initial_cash) in (int, float)
        first_close = next(iter(returns.values()))
        start_at = first_close.index[0]
        end_at = first_close.index[-1]
        idx = [start_at, end_at]
        values = [initial_cash, initial_cash]
        all_cash_series = pd.Series(values, idx)
        returns["All cash"] = all_cash_series

    # Wrap it up in a DataFrame
    benchmark_indexes = pd.DataFrame(returns)

    # Apply custom colors
    for symbol, colour in custom_colours.items():
        if symbol in benchmark_indexes.columns:
            benchmark_indexes[symbol].attrs = {"colour": colour}

    return benchmark_indexes


def visualise_long_short_benchmark(
    state: State,
    name: str | None = None,
    height: int | None = None,
):
    """Visualise separate benchmarks for both longing and shorting"""
    
    long_compounding_returns = calculate_long_compounding_realised_trading_profitability(state)
    short_compounding_returns = calculate_short_compounding_realised_trading_profitability(state)
    overall_compounding_returns = calculate_compounding_realised_trading_profitability(state)

    # visualise long equity curve
    long_curve = get_plot_from_series("long", "#006400", long_compounding_returns)
    short_curve = get_plot_from_series("short", "#8B0000", short_compounding_returns)
    overall_curve = get_plot_from_series("overall", "rgba(0, 0, 255, 1)", overall_compounding_returns)

    fig = go.Figure()
    fig.add_trace(long_curve)
    fig.add_trace(short_curve)
    fig.add_trace(overall_curve)

    fig.update_yaxes(title="compounding return %")

    if name:
        fig.update_layout(title=f"{name}")
    else:
        fig.update_layout(title="Equity curve for longs and shorts")
        
    if height:
        fig.update_layout(height=height)
        
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig

def get_plot_from_series(name, colour, series) -> go.Scatter:
    """Draw portfolio performance.
    
    :param name: name of the plot
    :param colour: colour of the plot
    :param series: series of daily returns
    :return: plotly scatter plot
    """
    plot = []
    for index, daily_return in series.items():
        plot.append({
            "timestamp": index,
            "value": daily_return,
        })

    df = pd.DataFrame(plot, columns=["timestamp", "value"])
    df.set_index("timestamp", inplace=True)

    fig = go.Scatter(
        x=df.index,
        y=df["value"],
        mode="lines",
        name=name,
        line=dict(color=colour),
    )
    
    return fig