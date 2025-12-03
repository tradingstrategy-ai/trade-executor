"""Analysis of trading universe."""

import pandas as pd
from plotly.graph_objects import Figure
import plotly.express as px

from tradeexecutor.strategy.chart.definition import ChartInput
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorNotFound
from tradingstrategy.liquidity import LiquidityDataUnavailable


def available_trading_pairs(
    input: ChartInput,
    all_criteria_included_pair_count="all_criteria_included_pair_count",
    volume_included_pair_count="volume_included_pair_count",
    tvl_included_pair_count="tvl_included_pair_count",
    trading_pair_count="trading_pair_count",
    with_dataframe: bool = False,
) -> Figure | tuple[Figure, pd.DataFrame]:
    """Render a chart showing the number of trading pairs available for the strategy to trade over history.

    :param input: ChartInput containing strategy input indicators.
    :param all_criteria_included_pair_count: Indicator name for pairs meeting all criteria.
    :param volume_included_pair_count: Indicator name for pairs meeting volume criteria.
    :param tvl_included_pair_count: Indicator name for pairs meeting TVL criteria.
    :param trading_pair_count: Indicator name for total trading pairs available.
    :param with_dataframe: If True, return both DataFrame and Figure.
    """

    indicator_data = input.strategy_input_indicators

    try:
        # Strategy does not define volume-inclusion criteria (e.g. vaults)
        volume_criteria = indicator_data.resolve_indicator_data(
            volume_included_pair_count,
            unlimited=True,
        )
    except IndicatorNotFound:
        volume_criteria = None

    data = {}

    data["Inclusion criteria met (all)"] = indicator_data.get_indicator_series(all_criteria_included_pair_count)
    if volume_criteria is not None:
        data["Inclusion criteria met (volume)"] = indicator_data.get_indicator_series(volume_included_pair_count)

    data["Inclusion criteria met (TVL)"] = indicator_data.get_indicator_series(tvl_included_pair_count)
    data["Visible pairs"] = indicator_data.get_indicator_series(trading_pair_count)

    # df = pd.DataFrame({
    #    : indicator_data.get_indicator_series(all_criteria_included_pair_count),
    #    "Volume criteria met": indicator_data.get_indicator_series(volume_included_pair_count) if volume_criteria is not None else None,
    #    "TVL criteria met": indicator_data.get_indicator_series(tvl_included_pair_count),
    #    "Visible pairs": indicator_data.get_indicator_series(trading_pair_count),
    #})

    df = pd.DataFrame(data)

    fig = px.line(df, title='Trading pairs available for strategy to trade')
    fig.update_yaxes(title="Number of assets")
    fig.update_xaxes(title="Time")

    if with_dataframe:
        return fig, df
    else:
        return fig


def inclusion_criteria_check(
    input: ChartInput,
    inclusion_criteria="inclusion_criteria",
    rolling_cumulative_volume="rolling_cumulative_volume",
) -> pd.DataFrame:
    """Create a table showing when certain pairs were included.

    :param inclusion_criteria:
        inclusion_criteria indicator name

    :param rolling_cumulative_volume:
        rolling_cumulative_volume indicator name
    """

    indicator_data = input.strategy_input_indicators
    strategy_universe = input.strategy_universe
    timestamp = input.end_at

    assert timestamp is not None, "End timestamp must be provided inclusion_criteria_check(). Set in ChartBacktestRenderingSetup()."

    series = indicator_data.get_indicator_series(inclusion_criteria)

    exploded = series.explode()
    first_appearance_series = exploded.groupby(exploded.values).apply(lambda x: x.index[0])

    df = pd.DataFrame({
        "Included at": first_appearance_series
    })

    def _get_ticker(pair_id):
        try:
            return strategy_universe.get_pair_by_id(pair_id).get_ticker()
        except Exception:
            return "<pair metadata missing>"

    def _get_dex(pair_id):
        try:
            return strategy_universe.get_pair_by_id(pair_id).exchange_name
        except Exception:
            return "<DEX metadata missing>"

    df["Ticker"] = first_appearance_series.index.map(_get_ticker)
    df["DEX"] = first_appearance_series.index.map(_get_dex)
    df = df.sort_values("Included at")

    def _map_tvl(row):
        pair_id = row.name  # Indxe
        timestamp = row["Included at"]
        try:
            tvl, delay = strategy_universe.data_universe.liquidity.get_liquidity_with_tolerance(
                pair_id,
                timestamp,
                tolerance=pd.Timedelta(days=2),
            )
            return tvl
        except LiquidityDataUnavailable:
            # TODO Data dates mismatch?
            return 0

    def _map_tvl_end(row):
        pair_id = row.name  # Indxe
        try:
            tvl, delay = strategy_universe.data_universe.liquidity.get_liquidity_with_tolerance(
                pair_id,
                timestamp,
                tolerance=pd.Timedelta(days=2),
            )
            return tvl
        except LiquidityDataUnavailable:
            # TODO Data dates mismatch?
            return 0

        # Get the first entry and value of rolling cum volume of each pair

    if indicator_data.has_indicator(rolling_cumulative_volume):
        # Strategy does not define volume-inclusion criteria (e.g. vaults)
        volume_series = indicator_data.get_indicator_data_pairs_combined(rolling_cumulative_volume)
        first_volume_df = volume_series.reset_index().groupby("pair_id").first()
    else:
        volume_series = None
        first_volume_df = None

    def _map_volume_timestamp(row):
        nonlocal first_volume_df
        pair_id = row.name  # Index
        try:
            return first_volume_df.loc[pair_id]["timestamp"]
        except KeyError:
            return None

    def _map_volume_value(row):
        nonlocal first_volume_df
        pair_id = row.name  # Index
        try:
            return first_volume_df.loc[pair_id]["value"]
        except KeyError:
            return None

    df["TVL at inclusion"] = df.apply(_map_tvl, axis=1)
    df["TVL at end"] = df.apply(_map_tvl_end, axis=1)

    if volume_series is not None:
        df["Rolling volume first entry at"] = df.apply(_map_volume_timestamp, axis=1)
        df["Rolling volume initial"] = df.apply(_map_volume_value, axis=1)
    return df
