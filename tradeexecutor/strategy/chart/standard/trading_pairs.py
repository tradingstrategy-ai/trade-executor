import pandas as pd
from plotly.graph_objects import Figure
import plotly.express as px

from tradeexecutor.strategy.chart.chart_definition import ChartInput


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

    df = pd.DataFrame({
        "Inclusion criteria met (all)": indicator_data.get_indicator_series(all_criteria_included_pair_count),
        "Volume criteria met": indicator_data.get_indicator_series(volume_included_pair_count),
        "TVL criteria met": indicator_data.get_indicator_series(tvl_included_pair_count),
        "Visible pairs": indicator_data.get_indicator_series(trading_pair_count),
    })

    fig = px.line(df, title='Trading pairs available for strategy to trade')
    fig.update_yaxes(title="Number of assets")
    fig.update_xaxes(title="Time")

    if with_dataframe:
        return fig, df
    else:
        return fig
