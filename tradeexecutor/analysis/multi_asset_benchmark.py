"""Calculate portfolio benchmarks for multiple assets simulatenously. """
import pandas as pd

from tradeexecutor.analysis.advanced_metrics import AdvancedMetricsMode, calculate_advanced_metrics
from tradeexecutor.state.state import State
from tradeexecutor.visual.equity_curve import calculate_equity_curve, calculate_returns

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.types import USDollarAmount
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, translate_trading_pair
from tradeexecutor.visual.benchmark import create_benchmark_equity_curves
from tradingstrategy.types import TokenSymbol


# Default branch colours of well-known assets
#
_colours = {
    "MATIC": "purple",
    "BTC": "orange",
    "ETH": "blue",
}


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
    interesting_assets=("BTC", "WBTC", "ETH", "WETH", "WMATIC", "MATIC"),
    initial_cash: USDollarAmount = None,
    all_cash=False,
) -> pd.DataFrame:
    """Get returns series of different benchmark index assets from the universe.

    - Assets are: BTC, ETH, MATIC

    To be used with :py:func:`compare_multiple_portfolios`.

    :param max_count:
        Return this many benchmark series

    :param interesting_assets:
        Choose to benchmark from these.

        We also check for wrapped token symbol varients.

    :param initial_cash:
        Also benchmark hold cash.

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

    # Get returns comparison data
    benchmark_indexes = create_benchmark_equity_curves(
        strategy_universe,
        benchmark_assets,
        initial_cash=initial_cash,
    )

    if not all_cash:
        del benchmark_indexes["All cash"]

    # Add series colour and name metadata
    for index_name in benchmark_indexes.columns:
        benchmark_returns = benchmark_indexes[index_name]
        benchmark_returns.attrs["name"] = index_name
        colour = _colours.get(index_name)
        if colour:
            benchmark_returns.attrs["colour"] = colour

    return benchmark_indexes


def compare_multiple_portfolios(
    portfolios: pd.DataFrame,
    mode: AdvancedMetricsMode=AdvancedMetricsMode.basic,
    periods_per_year=365,
    convert_to_daily=True,
) -> pd.DataFrame:
    """Compare multiple portfolios.

    - Assets against each other: BTC vs. ETH

    - Strategy against index: Strategy vs. BTC

    - Multiple assets in the same table: Strategt vs. BTC vs. ETH

    :param portfolios:
        A DataFrame of different return series.

        See :py:func:`tradeexecutor.visual.equity_curve.calculate_returns`.

        Each portfolio must have the returns series for the same time period,
        time periods are not matched.

    :return:
        QuantStats comparison of all different returns.
    """

    result_table = pd.DataFrame()

    for series_name in portfolios.columns:
        series = portfolios[series_name]

        metrics = calculate_advanced_metrics(
            series,
            mode=mode,
            convert_to_daily=convert_to_daily,
            periods_per_year=periods_per_year,
        )

        result_table[series_name] = metrics["Strategy"]

    return result_table


def compare_strategy_backtest_to_multiple_assets(
    state: State,
    strategy_universe: TradingStrategyUniverse,
    all_cash=False,
) -> pd.DataFrame:
    """Backtest comparison of strategy against buy and hold assets."""
    equity = calculate_equity_curve(state)
    returns = calculate_returns(equity)
    benchmarks = get_benchmark_data(
        strategy_universe,
        initial_cash=state.portfolio.get_initial_cash(),
        all_cash=all_cash,
    )

    comparison_returns = pd.DataFrame(
        {"Strategy": returns}
    )

    for series_name, series in benchmarks.items():
        comparison_returns[series_name] = series

    return compare_multiple_portfolios(comparison_returns)

