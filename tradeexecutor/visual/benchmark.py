"""Compare portfolio performance against other strategies."""
import datetime
import warnings
from typing import Optional, List, Union, Collection, Dict

import plotly.graph_objects as go
import pandas as pd
from pandas._libs.tslibs.offsets import MonthBegin

from tradeexecutor.analysis.curve import DEFAULT_BENCHMARK_COLOURS, CurveType
from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.statistics import PortfolioStatistics
from tradeexecutor.state.visualisation import Plot
from tradeexecutor.state.state import State
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.visual.technical_indicator import visualise_technical_indicator
from tradeexecutor.visual.equity_curve import calculate_long_compounding_realised_trading_profitability, \
    calculate_short_compounding_realised_trading_profitability, calculate_compounding_realised_trading_profitability, resample_returns, calculate_returns
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


def visualise_equity_curve_comparison(
    benchmark_name: str,
    price_series: pd.Series,
    all_cash: float,
    colour="#880000"
) -> go.Scatter:
    """Draw portfolio performance."""

    # Whatever we bought at the start
    initial_inventory = all_cash / float(price_series.iloc[0])
    series = price_series * initial_inventory

    return go.Scatter(
        x=series.index,
        y=series,
        mode="lines",
        name=benchmark_name,
        line=dict(color=colour),
    )


def visualise_equity_curve_benchmark(
    name: Optional[str] = None,
    title: Optional[str] = None,
    portfolio_statistics: Optional[List[PortfolioStatistics]] = None,
    all_cash: Optional[float] = None,
    buy_and_hold_asset_name: Optional[str] = None,
    buy_and_hold_price_series: Optional[pd.Series] = None,
    benchmark_indexes: pd.DataFrame = None,
    additional_indicators: Collection[Plot] = None,
    height=1200,
    start_at: Optional[Union[pd.Timestamp, datetime.datetime]] = None,
    end_at: Optional[Union[pd.Timestamp, datetime.datetime]] = None,
    log_y=False,
) -> go.Figure:
    """Visualise strategy performance against benchmarks.

    - Live or backtested strategies

    - Benchmark against buy and hold of various assets

    - Benchmark against hold all cash

    .. note::

        This will be deprecated. Use :py:func:`visualise_equity_curves` instead.

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

    :param name:
        The name of the primary asset we benchark

    :param title:
        The title of the chart if separate from primary asset

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

    :param log_y:
        Use logarithmic Y-axis.

        Because we accumulate larger treasury over time,
        the swings in the value will be higher later.
        We need to use a logarithmic Y axis so that we can compare the performance
        early in the strateg and late in the strategy.

    :return:
        Plotly figure
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
    for benchmark_name, buy_and_hold_price_series in benchmark_indexes.items():

        colour = buy_and_hold_price_series.attrs.get("colour")

        # Clip to the backtest time frame
        buy_and_hold_price_series = buy_and_hold_price_series[start_at:end_at]

        if buy_and_hold_price_series.attrs.get("returns_series_type") == "cumulative_returns":
            # See get_benchmark_data()
            scatter = go.Scatter(
                x=buy_and_hold_price_series.index,
                y=buy_and_hold_price_series,
                mode="lines",
                name=benchmark_name,
                line=dict(color=buy_and_hold_price_series.attrs.get("colour")),
            )

            fig.add_trace(scatter)
        else:
            # Legacy path without get_benchmark_data()
            scatter = visualise_equity_curve_comparison(
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
        fig.update_layout(title=f"{title or name}", height=height)
    else:
        fig.update_layout(title=f"Portfolio value", height=height)

    if log_y:
        fig.update_yaxes(title="Value $ (logarithmic)", showgrid=False, type="log")
    else:
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


def visualise_benchmark(*args, **kwargs) -> go.Figure:
    warnings.warn('This function is deprecated. Use visualise_equity_curve_benchmark instead', DeprecationWarning, stacklevel=2)
    return visualise_equity_curve_benchmark(*args, **kwargs)


def create_benchmark_equity_curves(
    strategy_universe: TradingStrategyUniverse,
    pairs: Dict[str, HumanReadableTradingPairDescription | TradingPairIdentifier],
    initial_cash: USDollarAmount,
    custom_colours=DEFAULT_BENCHMARK_COLOURS,
    convert_to_daily=False,
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
        if isinstance(value, TradingPairIdentifier):
            close_data = strategy_universe.data_universe.candles.get_candles_by_pair(value.internal_id)["close"]
        else:
            # Assume HumanReadableTradingPairDescription
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
            benchmark_indexes[symbol].attrs = {"colour": colour, "name": symbol}

    return benchmark_indexes


def visualise_long_short_benchmark(
    state: State,
    name: str | None = None,
    height: int | None = None,
) -> go.Figure:
    """Visualise separate benchmarks for both longing and shorting

    .. note ::
        This chart is inaccurate for strategies that can have multiple positions open at the same time.
    
    :param state: state of the strategy
    :param name: name of the plot
    :param height: height of the plot
    :return: plotly figure
    """
    
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

    scatter = go.Scatter(
        x=df.index,
        y=df["value"],
        mode="lines",
        name=name,
        line=dict(color=colour),
    )
    
    return scatter


def visualise_vs_returns(
    returns: pd.Series,
    benchmark_indexes: pd.DataFrame,
    name="Strategy returns multiplier vs. benchmark indices",
    height=800,
    skipped_benchmarks=("All cash",),
    freq: pd.DateOffset = MonthBegin()
)-> go.Figure:
    """Create a chart that shows the strategy returns direction vs. benchmark.

    - This will tell if the strategy is performing better or worse over time

    :param returns:
        Strategy returns.

        `Series.attrs` can contain `name` and `colour`.

    :param benchmark_indexes:
        Benchmark buy and hold indexes.

        Assume starts with `initial_cash` like $10,000 and then moves with the price action.

        Each `Series.attrs` can have keys `name` and `colour`.
    
    :param name:
        Figure title.

    :param height:
        Chart height in pixels.

    :param skipped_benchmarks:
        Benchmark indices we do not need to render.

    :param freq:
        Binning frequency for more readable charts.

        Choose between weekly, monthly, quaterly, yearly binning.
    """

    assert isinstance(returns, pd.Series)
    assert isinstance(benchmark_indexes, pd.DataFrame)
    assert len(benchmark_indexes.columns) > 0, f"benchmark_indexes is empty"

    fig = go.Figure()

    resampled_returns = resample_returns(returns, freq)

    fig.update_layout(title=f"name", height=height)
    for benchmark in benchmark_indexes.columns:

        if any(s for s in skipped_benchmarks if s in benchmark):
            # Simple string match filter
            continue
        benchmark_series = benchmark_indexes[benchmark]
        benchmark_returns = calculate_returns(benchmark_series)
        resampled_benchmark = resample_returns(benchmark_returns, freq)
        ratio_series = (1+resampled_returns) / (1+resampled_benchmark)
        colour = benchmark_series.attrs.get("colour")
        scatter = go.Scatter(
            x=ratio_series.index,
            y=ratio_series,
            mode="lines",
            name=f"Strategy returns vs. {benchmark} returns",
            line=dict(color=colour),
        )
        fig.add_trace(scatter)

    # Add line at 1
    scatter = go.Scatter(
        x=resampled_returns.index,
        y=[1] * len(resampled_returns.index),
        mode="lines",
        name=f"Strategy and index perform equal",
        line=dict(color="black"),
    )
    fig.add_trace(scatter)

    fig.update_yaxes(title="Strategy x benchmark", showgrid=False, tickformat=".2fx")
    fig.update_xaxes(rangeslider={"visible": False})
    fig.update_layout(title=name)

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



def transform_to_equity_curve(curve: pd.Series, initial_cash: USDollarAmount) -> pd.Series:
    """Transform returns series to equity series.

    :param curve:
        Returns series with `pd.Series.attrs["curve"]` set

    :param initial_cash:
        The amount of strategy initial cash.

        Needed to transform benchmark return series to comparable equity curve.

    :return:
        Equity curve ready to plot
    """

    name = curve.attrs.get("name", "<unnamed>")
    curve_type = curve.attrs.get("curve")
    assert isinstance(curve_type, CurveType)

    match curve_type:
        case CurveType.equity:
            return curve
        case CurveType.returns:
            assert initial_cash, "initial_cash must be given to transform different return series to equity curve"
            raise NotImplementedError("Please add implementation")
        case _:
            raise NotImplementedError(f"Unsupported curve type {curve_type} on {name}")


def visualise_equity_curves(
    curves: list[pd.Series],
    name="Equity curve comparison",
    height=800,
    log_y=False,
) -> go.Figure:
    """Compare equity curves.

    - Draw a plot of different equity curve / return series in the same diagram

    - To benchmark grid search results, see :py:func:`tradeeexecutor.visual.grid_search.visualise_grid_search_result_benchmark`

    See also

    - :py:class:`tradeexecutor.analysis.curve.CurveType`

    - :py:func:`visualise_equity_curve_benchmark` (legacy)

    :param curves:
        pd.Series with their "curve" attribute set.

        Each Pandas series can have attributes

        - name

        - colour: Plotly colour name

        - curve: Equity curve type

    :param height:
        Height in pixels

    :param initial_cash:
        The amount of strategy initial cash.

        Needed to transform benchmark return series to comparable equity curve.

    :return:
        Plotly figure
    """

    fig = go.Figure()

    for curve in curves:
        # Do sanity checks on incoming data
        # See curve.py for potential attributes
        assert isinstance(curve, pd.Series), f"Expected pd.Series, got {type(curve)}"
        assert isinstance(curve.index, pd.DatetimeIndex), f"Expected DateTimeIndex, got {type(curve.index)}"
        curve_name = curve.attrs.get("name")
        assert curve_name, "Series lacks attrs['name']"
        curve_type = curve.attrs.get("curve")
        assert curve_type, f"Series lacks attrs['curve']: {curve_name}"
        colour = curve.attrs.get("colour")
        assert colour, f"Series lacks attrs['colour']: {curve_name}"
        assert isinstance(curve_type, CurveType), f"{name}: Expected curve to be CurveType, got {type(curve_type)}"
        assert curve_type == CurveType.equity, f"Only CurveType.equity is supported in this point, got {curve_type} for {curve_name}"

        # not implemented yet
        # series = transform_to_equity_curve(curve, initial_cash)
        series = curve

        trace = go.Scatter(
            x=series.index,
            y=series,
            mode="lines",
            name=curve_name,
            line=dict(color=colour),
        )

        fig.add_trace(trace)

    fig.update_layout(title=name, height=height)

    if log_y:
        fig.update_yaxes(title="Value $ (logarithmic)", showgrid=False, type="log")
    else:
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

def visualise_portfolio_interest_curve(
    name: str,
    state: State,
    colour="#008800",
) -> go.Figure:
    """Draw portfolio's interest performance."""

    plot = []
    for p in state.portfolio.closed_positions.values():
        if p.is_closed() and p.is_credit_supply():
            plot.append(
                {
                    "opened_at": p.opened_at, 
                    "closed_at": p.closed_at, 
                    "realised_profit_usd": p.get_realised_profit_usd(),
                }
            )

    df = pd.DataFrame(plot)
    df.set_index("opened_at", inplace=True)
    df["total_interest"] = df["realised_profit_usd"].cumsum()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["total_interest"],
            mode="lines",
            name=name,
            line=dict(color=colour),
        )
    )

    if name:
        fig.update_layout(title=name, height=1200)

    return fig
