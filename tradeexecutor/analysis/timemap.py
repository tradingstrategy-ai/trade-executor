"""Break strategy performance by time

- Hour of the day

- Day of the week

"""
from typing import Iterable

import plotly.express as px
import plotly.graph_objects as go

from tradeexecutor.state.position import TradingPosition

DEFAULT_COLOUR_SCHEME = {}

class ScoringMethod:
    realised_profitability = "realised_profitability"
    success_rate = "success_rate"
    failure_rate = "failure_rate"


CHART_TITLES = {
    ScoringMethod.realised_profitability: "Realised profitability per opening hour of a position",
    ScoringMethod.success_rate: "Number of profitable positions per opening hour",
    ScoringMethod.failure_rate: "Number of unprofitable positions per opening hour",
}


COLOUR_SCHEHEM_NAMING = {
    ScoringMethod.success_rate: "Positions",
    ScoringMethod.failure_rate: "Positions (neg)",
    ScoringMethod.realised_profitability: "Profitability %",
}


def visualise_weekly_time_heatmap(
    positions: Iterable[TradingPosition],
    method: ScoringMethod = ScoringMethod.success_rate,
    color_continuous_scale: str | None = None,  # Reversed, blue = best
    height=600,
) -> go.Figure:
    """Create a heatmap of which hours/days are best for trading.

    - Figyre out best trading hours

    - Mostly useful for strategies that trade 1h or more frequently

    :param positions:
        Any trading positions to consider.

        We will filter based on method.

    :param method:
        Which kind of heatmap to draw.

    :param color_continuous_scale:
        The color scheme of the heatmap.

    :return:
        Plotly heatmap figure
    """

    # Initialise the grid to 0 values
    weekday_names = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    if method == ScoringMethod.realised_profitability:
        initial_value = 1
    else:
        initial_value = 0

    data = {}
    for day in range(7):
        data[day] = [initial_value] * 24

    # Go through positions and add them a score on the timemap based on the position opening
    for p in positions:
        opened_at = p.opened_at
        weekday = opened_at.weekday()
        hour = opened_at.hour
        profit = p.get_unrealised_and_realised_profit_percent()

        match method:
            case ScoringMethod.success_rate:
                if profit > 0:
                    data[weekday][hour] += 1
            case ScoringMethod.failure_rate:
                if profit < 0:
                    data[weekday][hour] -= 1
            case ScoringMethod.realised_profitability:
                # Cumulative profifability for this day
                data[weekday][hour] *= (1+profit)

    matrix = [v for v in data.values()]

    # Normalise -100% to 100%
    if method == ScoringMethod.realised_profitability:
        for day in range(7):
            for hour in range(24):
                matrix[day][hour] = matrix[day][hour] * 100 - 100
        print(matrix)

    # https://plotly.com/python/heatmaps/
    fig = px.imshow(
        matrix,
        labels=dict(x="Hour (UTC)", y="Day of week", color=COLOUR_SCHEHEM_NAMING.get(method)),
        y=weekday_names,
        x=[f"{h:02d}:00" for h in range(24)],
        height=height,
        color_continuous_scale=color_continuous_scale,
    )
    fig.update_xaxes(side="top")
    fig.update_layout(
        title=CHART_TITLES.get(method)
    )
    return fig
