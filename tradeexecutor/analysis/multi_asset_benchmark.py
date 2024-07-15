"""Calculate portfolio benchmark tables for multiple assets side-by-side.

"""

import pandas as pd

from tradeexecutor.analysis.advanced_metrics import AdvancedMetricsMode, calculate_advanced_metrics
from tradeexecutor.state.state import State
from tradeexecutor.state.types import USDollarAmount
from tradeexecutor.analysis.curve import DEFAULT_BENCHMARK_COLOURS, CurveType
from tradeexecutor.visual.equity_curve import calculate_equity_curve, calculate_returns, resample_returns

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, translate_trading_pair
from tradingstrategy.timebucket import TimeBucket

from tradingstrategy.types import TokenSymbol


#: What is the priority of buy-and-hold assets to shown in the benchmarks
DEFAULT_BENCHMARK_ASSETS = [
    "BTC",
    "WBTC",
    "ETH",
    "WETH",
    "WMATIC",
    "MATIC",
    "ARB",
    "WARB"
]


def _find_benchmark_pair(strategy_universe: TradingStrategyUniverse, token_symbol: TokenSymbol) -> TradingPairIdentifier | None:
    """Try to find the price series for our comparison asset in the trading universe."""
    for dex_pair in strategy_universe.data_universe.pairs.iterate_pairs():
        pair = translate_trading_pair(dex_pair)
        if pair.base.token_symbol == token_symbol:
            return pair

    return None


def get_benchmark_data(
    strategy_universe: TradingStrategyUniverse,
    max_count=2,
    interesting_assets=DEFAULT_BENCHMARK_ASSETS,
    cumulative_with_initial_cash: USDollarAmount =0.0,
    asset_colours=DEFAULT_BENCHMARK_COLOURS,
    start_at: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Get returns series of different benchmark index assets from the universe.

    - Assets are: BTC, ETH, MATIC

    To be used with :py:func:`compare_multiple_portfolios` and :py:func:`tradeexecutor.visual.benchmark.visualise_equity_curve_benchmark`.

    Example:

    .. code-block:: python

        from tradeexecutor.analysis.multi_asset_benchmark import get_benchmark_data
        from tradeexecutor.visual.benchmark import visualise_equity_curve_benchmark

        benchmark_indexes = get_benchmark_data(
            strategy_universe,
            cumulative_with_initial_cash=state.portfolio.get_initial_cash()
        )

        fig = visualise_equity_curve_benchmark(
            name=state.name,
            portfolio_statistics=state.stats.portfolio,
            all_cash=state.portfolio.get_initial_cash(),
            benchmark_indexes=benchmark_indexes,
            height=800,
            log_y=False,
        )

        fig.show()

    :param max_count:
        Return this many benchmark series

    :param interesting_assets:
        Choose to benchmark from these.

        We also check for wrapped token symbol varients.

    :param cumulative_with_initial_cash:
        Get cumulative returns instead of daily returns.

        Set to the cumulative initial cash value.

    :return:
        DataFrame with returns series for each asset.

        Each series has colour and name metadata added to the `series.attr`.
    """

    benchmark_assets = {}

    # Get the trading pair ids for the assets we want to compare
    for asset in interesting_assets:
        pair = _find_benchmark_pair(strategy_universe, asset)
        if not pair:
            continue
        unwrapped_name = asset[1:] if asset.startswith("W") else asset
        benchmark_assets[unwrapped_name] = pair
        if len(benchmark_assets) >= max_count:
            break

    df = pd.DataFrame()

    # Check that we have a good source for daily returns
    # If we have only weekly candles we cannot really calculate daily returns
    # and will get funny results. In this case, check the availability
    # of resampling price series from stop loss backtesting data.
    candle_source = None
    candle_source_freq = None
    if strategy_universe.data_universe.time_bucket > TimeBucket.d1:
        # Check if we can use stop loss data
        if strategy_universe.backtest_stop_loss_time_bucket:
            if strategy_universe.backtest_stop_loss_time_bucket <= TimeBucket.d1:
                candle_source = strategy_universe.backtest_stop_loss_candles
                candle_source_freq = strategy_universe.backtest_stop_loss_time_bucket

    else:
        candle_source = strategy_universe.data_universe.candles
        candle_source_freq = strategy_universe.data_universe.time_bucket

    assert candle_source, f"Could not find daily or shorter candles to calculate daily returns for benchmark data indexes.\n" \
                          f"Candle time bucket: {strategy_universe.data_universe.time_bucket}\n" \
                          f"Stop-loss candle time bucket: {strategy_universe.backtest_stop_loss_time_bucket}\n"

    for name, pair in benchmark_assets.items():

        price_series = candle_source.get_candles_by_pair(pair.internal_id)["close"]

        if start_at:
            price_series = price_series.loc[start_at:]

        assert len(price_series.dropna()) != 0, f"Failed to read benchmark price series for {name}: {pair}"
        if isinstance(price_series.index, pd.MultiIndex):
            index_fixed_series = pd.Series(data=price_series.values, index=price_series.index.get_level_values(1))
        else:
            index_fixed_series = price_series

        if candle_source_freq == TimeBucket.d1:
            daily_returns = index_fixed_series.pct_change()
        else:
            daily_returns = resample_returns(index_fixed_series.pct_change(), freq="D")

        if cumulative_with_initial_cash:
            cumulative_returns = (1 + daily_returns).cumprod()
            equity_curve = cumulative_returns * cumulative_with_initial_cash

            # Merge df and equity_curve_df with an outer join
            # since indexes can differ
            equity_curve_df = equity_curve.to_frame(name=name)
            df = df.merge(equity_curve_df, left_index=True, right_index=True, how='outer')

            df[name] = equity_curve
        else:
            df[name] = daily_returns
            
    # reassign attrs after merge since they are lost
    for column in df.columns:
        name = df[column].name
        if cumulative_with_initial_cash:
            df[name].attrs["name"] = name
            df[name].attrs["colour"] = asset_colours.get(name)
            df[name].attrs["returns_series_type"] = "cumulative_returns"
            df[name].attrs["curve"] = CurveType.equity
        else:
            df[name].attrs["returns_series_type"] = "daily_returns"
            df[name].attrs["period"] = "D"
            df[name].attrs["curve"] = CurveType.returns

    return df


def compare_multiple_portfolios(
    portfolios: pd.DataFrame,
    indexes: pd.DataFrame | None = None,
    mode: AdvancedMetricsMode=AdvancedMetricsMode.basic,
    periods_per_year=365,
    display=False,
) -> pd.DataFrame:
    """Compare multiple portfolios.

    - Assets against each other: BTC vs. ETH

    - Strategy against index: Strategy vs. BTC

    - Multiple assets in the same table: Strategt vs. BTC vs. ETH

    :param portfolios:
        A DataFrame of different daily series of actively trading portfolios.

        See :py:func:`tradeexecutor.visual.equity_curve.calculate_returns`.

        Each portfolio must have the returns series for the same time period,
        time periods are not matched.

    :param indexes:
        A DataFrame of different daily returns passive buy and hold indexes.

    :return:
        QuantStats comparison of all different returns.
    """

    result_table = pd.DataFrame()

    assert len(portfolios.columns) + len(indexes.columns) >= 1, f"Need at least one portfolios to benchmark: {portfolios.columns}, {indexes.columns}"

    for name, portfolio_series in portfolios.items():
        metrics = calculate_advanced_metrics(
            portfolio_series,
            mode=mode,
            periods_per_year=periods_per_year,
            convert_to_daily=False,
            display=display,
        )
        assert "Strategy" in metrics.columns, f"We got {metrics.columns}"
        result_table[name] = metrics["Strategy"]
        last_series = portfolio_series

    for name, index_series in indexes.items():
        metrics = calculate_advanced_metrics(
            last_series,
            benchmark=index_series,
            mode=mode,
            periods_per_year=periods_per_year,
            convert_to_daily=False,
            display=display,
        )
        result_table[name] = metrics["Benchmark"]

    return result_table


def compare_strategy_backtest_to_multiple_assets(
    state: State | None,
    strategy_universe: TradingStrategyUniverse,
    returns: pd.Series | None = None,
    display=False,
    asset_count=3,
) -> pd.DataFrame:
    """Backtest comparison of strategy against buy and hold assets.

    :return:
        DataFrame with QuantStats results.

        One column for strategy and for each benchmark asset we have loaded in the strategy universe.
    """

    # Get daily returns
    if returns is None:
        assert state, "State must be given if no returns are given"
        equity = calculate_equity_curve(state)
        returns = calculate_returns(equity)

    import ipdb ; ipdb.set_trace()
    daily_returns = resample_returns(returns, "D")

    benchmarks = get_benchmark_data(
        strategy_universe,
        max_count=asset_count,
    )

    portfolios = pd.DataFrame(
        {"Strategy": daily_returns}
    )

    return compare_multiple_portfolios(
        portfolios=portfolios,
        indexes=benchmarks,
        display=display,
    )

