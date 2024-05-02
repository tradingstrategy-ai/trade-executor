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


def visualise_weekly_time_heatmap(
    positions: Iterable[TradingPosition],
    method: ScoringMethod = ScoringMethod.success_rate,
    colour_scheme=DEFAULT_COLOUR_SCHEME,
) -> go.Figure:
    """Create a heatmap of which hours/days are best for trading."""

    # Initialise the grid to 0 values

    # Each weekday is a series of entries 0-24
    hours = range(0, 24)

    weekday_names = [
        "Monday",
        "Tuesday"
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    data = {}
    for day in weekday_names:
        data[day] = [0] * hours

    # Go through positions and add them a score on the timemap based on the position opening

    for p in positions:

        assert p.is_closed(), "Only closed positions supported"

        opened_at = p.opened_at
        weekday = opened_at.weekday()
        named_weekday = weekday_names[weekday]
        hour = opened_at.hour

        match method:
            case ScoringMethod.success_rate:
                if p.get_realised_profit_usd() > 0:
                    data[named_weekday][hour] += 1
            case ScoringMethod.failure_rate:
                if p.get_realised_profit_usd() < 0:
                    data[named_weekday][hour] -= 1
            case ScoringMethod.realised_profitability:
                # Cumulative profifability for this day
                data[named_weekday][hour] *= (1+data[named_weekday][hour]) * p.get_realised_profit_percent() - 1

    match method:
        case ScoringMethod.success_rate:
            color = "Number of successful trades"
        case ScoringMethod.failure_rate:
            color = "Number of failed trades"
        case ScoringMethod.realised_profitability:
            color = "Realised profatibility"

    # https://plotly.com/python/heatmaps/
    fig = px.imshow(
        [v for v in data.values()],
        labels=dict(x="Day of Week", y="Time of Day", color=color),
        x=weekday_names,
        y=[str(h) for h in hours]
    )
    fig.update_xaxes(side="top")
    return fig







