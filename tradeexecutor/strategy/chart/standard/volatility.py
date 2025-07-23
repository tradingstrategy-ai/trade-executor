"""Volatility across pairs in the universe."""

import pandas as pd
from plotly.graph_objects import Figure
import plotly.express as px


from tradeexecutor.strategy.chart.definition import ChartInput


def volatility_benchmark(
    input: ChartInput,
    avg_volatility: str = "avg_volatility",
) -> Figure:
    """Render historical volatility for some pairs based on the provided volatility indicator name."""

    assert input.pairs and len(input.pairs) > 0, "No pairs provided in the input."

    indicator_data= input.strategy_input_indicators

    avg_volatility = indicator_data.get_indicator_series(avg_volatility)

    volatilities = {
        "avg_volatility": avg_volatility,
    }

    max_displayed_vol = avg_volatility.max() * 3

    # TODO: Plotly refuses correctly to plot the third y-axis
    for pair in input.pairs:
        symbol = pair.base.symbol
        volatility = indicator_data.get_indicator_series("volatility", pair=pair)
        volatility = volatility.clip(upper=max_displayed_vol)
        volatilities[symbol] = volatility

    volatility_df = pd.DataFrame(volatilities)
    fig = px.line(volatility_df)
    return fig