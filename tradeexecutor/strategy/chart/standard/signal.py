"""Compare signal for portfolio construction across pairs"""

import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.graph_objects import Figure

from tradeexecutor.strategy.chart.definition import ChartInput


def signal_comparison(
    input: ChartInput,
    avg_signal = "avg_signal",
) -> Figure:
    """Render share price and TVL for all vaults.

    - Get vault pairs from the strategy universe

    :param avg_signal:
        Indicator name for the average signal across pairs.

    :return:
        List of figures
    """

    assert input.pairs and len(input.pairs) > 0, "No pairs provided in the input."

    indicator_data = input.strategy_input_indicators

    # max_displayed_vol = avg_signal.max() * 1.1
    upper_displayed = 0.1

    avg_signal = indicator_data.get_indicator_series(avg_signal)

    data = {
        "avg_signal": avg_signal.clip(upper=upper_displayed),
    }

    # TODO: Plotly refuses correctly to plot the third y-axis
    for pair in input.pairs:
        signal = indicator_data.get_indicator_series("signal", pair=pair)
        signal = signal.clip(upper=upper_displayed)
        data[pair.base.symbol] = signal

    df = pd.DataFrame(data)
    fig = px.line(df)
    return fig


def price_vs_signal(
    input: ChartInput,
    indicator_name: str = "signal",
) -> list[Figure]:
    """Price vs. signal comparison for selected pairs."""

    assert input.pairs and len(input.pairs) > 0, "No pairs provided in the input."

    indicator_data = input.strategy_input_indicators
    strategy_universe = input.strategy_universe

    figs = []

    # TODO: Plotly refuses correctly to plot the third y-axis
    for pair in input.pairs:
        symbol = pair.base.token_symbol
        price = strategy_universe.data_universe.candles.get_candles_by_pair(pair.internal_id)["close"]
        signal = indicator_data.get_indicator_series(indicator_name, pair=pair)
        # volatility = indicator_data.get_indicator_series("volatility", pair=pair)

        df = pd.DataFrame({
            "price": price,
            "signal": signal,
            # "volatility": volatility,
        })

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(x=df.index, y=df["signal"], name="Signal"),
            secondary_y=False,
        )

        # fig.add_trace(
        #    go.Scatter(x=df.index, y=df["volatility"], name="Volatility", yaxis="y2"),
        #    secondary_y=True,
        #)

        fig.add_trace(
            go.Scatter(x=df.index, y=df["price"], name="Price", yaxis="y3"),
            secondary_y=True,
        )

        fig.update_layout(title=f"Price vs. signal {symbol}")
        fig.update_layout(showlegend=True)
        fig.update_xaxes(title="Time")
        # fig.update_layout(
        #    yaxis_type="log"
        #)

        figs.add[fig]

    return figs



