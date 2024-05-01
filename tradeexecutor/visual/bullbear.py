"""Bull/bear market regime filter visualisation.

- Visualise discreet bull/bear market filter by showing coloured regions over price chart

"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tradeexecutor.analysis.regime import Regime, get_regime_signal_regions

#: Different regime colours
DEFAULT_COLOUR_MAP = {
    Regime.bull: "green",
    Regime.bear: "red",
}


def visualise_market_regime_filter(
    price: pd.Series,
    signal: pd.Series,
    title="Regime filter",
    height=800,
    colour_map=DEFAULT_COLOUR_MAP,
    opacity=0.1,
) -> go.Figure:
    """Visualise a bull/bear market regime filter on the top of the price action chart.

    - Draw price
    - Under price colour regime filter regions. Bull regime green, bear regime read.
    - Crab market does not have colour

    :param price_series:
        Price on an asset

    :param signal:
        The market regime filter signal.

        See :py:class:`Regime`

        +1 for bull
        0 = crab
        -1 for bear

    :param title:
        Chart title

    :return:
        A chart with price action.

        Sections that are in bull market are coloured green.

        Sections that are in bear market are coloured red.

        Crab market sections are blue.
    """

    assert isinstance(price, pd.Series), f"Price expected to be a continous series: {price}"
    assert isinstance(signal, pd.Series), f"Signal expected to be a continous series: {signal}"

    assert len(price) > 0
    assert len(signal) > 0

    # TODO: Complications
    # assert len(price) == len(signal), f"Price and signal are different series. Price length: {len(price)}, signal length: {len(signal)}"

    signal_unique = signal.unique()
    for uniq_val in signal_unique:
        assert uniq_val in (-1, 0, 1), f"Got unknown market regime value: {uniq_val}"

    # Fill the area between close price and SMA indicator
    # See https://plotly.com/python/filled-area-plots/#interior-filling-for-area-chart
    # See also https://stackoverflow.com/a/64743166/315168
    fig = go.Figure(
        layout={
            "title": title,
            "height": height,
        }
    )

    # We need to use an invisible trace so we can reset "next y"
    for region in get_regime_signal_regions(signal):
        # https://stackoverflow.com/questions/55062965/python-how-to-make-shaded-areas-or-alternating-background-color-using-plotly
        colour = colour_map.get(region.regime)
        if colour:
            fig.add_vrect(
                x0=region.start,
                x1=region.end,
                fillcolor=colour_map.get(region.regime),
                opacity=opacity,
                line_width=0,
            )

    # for the red area indicator
    fig.add_trace(
        go.Scatter(
            x=price.index,
            y=price,
            line_color="black",
            showlegend=False,
        )
    )

    return fig


def visualise_raw_market_regime_indicator(
    price: pd.Series,
    indicator: pd.DataFrame,
    title="Raw market regime indicator data",
    height=800,
    indicator_height=200,
):
    """Draw a raw underlying market regime filter indicator data.

    - Visualise ADX indicator or similar under a price chart

    :param price:
        Close price series

    :param indicator:
        ADX indicator data or similar

    :return:
        Plotly figure
    """

    assert isinstance(price, pd.Series)
    assert isinstance(indicator, pd.DataFrame)

    indicator_count = len(indicator.columns)
    rows = 1 + indicator_count

    row_heights = [height] + [indicator_height] * indicator_count
    total_height = sum(row_heights)

    fig = make_subplots(
        rows=rows,
        shared_xaxes=True,
        row_heights=row_heights
    )

    fig.add_trace(
        go.Scatter(
            x=price.index,
            y=price,
            line_color="black",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text='Price',
        row=1,
        col=1,
    )
    row = 2
    for name, indicator in indicator.items():
        fig.add_scatter(
            x=indicator.index,
            y=indicator,
            row=row,
            col=1,
        )
        fig.update_yaxes(
            title_text=name,
            row=row,
            col=1,
        )
        row += 1

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        title=title,
        showlegend=False,
        hovermode="x unified",
        height=total_height,
    )

    # Unified cross hair
    # https://community.plotly.com/t/unified-hovermode-with-sublots/37606/2
    fig.update_traces(xaxis='x1')

    return fig

