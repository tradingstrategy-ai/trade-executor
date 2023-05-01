"""Grid search result analysis.

- Breaddown of performance of different grid search combinations

- Heatmap and other comparison methods

"""

from typing import List

import numpy as np
import pandas as pd
from IPython.core.display_functions import display

import plotly.express as px
from plotly.graph_objs import Figure

from tradeexecutor.backtest.grid_search import GridSearchResult


VALUE_COLS = ["Annualised return", "Max drawdown", "Sharpe", "Sortino", "Average position", "Median position"]

PERCENT_COLS = ["Annualised return", "Max drawdown", "Average position", "Median position"]


def analyse_combination(r: GridSearchResult) -> dict:
    """Create a grid search result table row.

    - Create columns we can use to compare different grid search combinations
    """

    row = {}

    for param in r.combination.parameters:
        row[param.name] = param.value

    def clean(x):
        if x == "-":
            return np.NaN
        elif x == "":
            return np.NaN
        return x

    # import ipdb ; ipdb.set_trace()

    row.update({
        # "Combination": r.combination.get_label(),
        "Positions": r.summary.total_positions,
        # "Return": r.summary.return_percent,
        # "Return2": r.summary.annualised_return_percent,
        #"Annualised profit": clean(r.metrics.loc["Expected Yearly"][0]),
        "Annualised return": clean(r.metrics.loc["All-time (ann.)"][0]),
        "Max drawdown": clean(r.metrics.loc["Max Drawdown"][0]),
        "Sharpe": clean(r.metrics.loc["Sharpe"][0]),
        "Sortino": clean(r.metrics.loc["Sortino"][0]),
        "Average position": r.summary.average_trade,
        "Median position": r.summary.median_trade,
    })
    return row


def analyse_grid_search_result(results: List[GridSearchResult], drop_index=True) -> pd.DataFrame:
    """Create aa table showing grid search result of each combination.

    - Each row have labeled parameters of its combination

    - Each row has some metrics extracted from the results by :py:func:`analyse_combination`

    See also :py:func:`analyse_combination`.

    :param results:
        Output from :py:meth:`tradeexecutor.backtest.grid_search.perform_grid_search`.

    :return:
        Table of grid search combinations
    """
    assert len(results) > 0, "No results"
    rows = [analyse_combination(r) for r in results]
    df = pd.DataFrame(rows)
    r = results[0]
    param_names = [p.name for p in r.combination.parameters]
    df = df.set_index(param_names, drop=drop_index)
    df = df.sort_index()
    return df


def visualise_table(df: pd.DataFrame):
    """Render a grid search combination table to notebook output.

    - Highlight winners and losers

    """

    # https://stackoverflow.com/a/57152529/315168

    # TODO:
    # Diverge color gradient around zero
    # https://stackoverflow.com/a/60654669/315168

    formatted = df.style.background_gradient(
        axis = 0,
        subset = VALUE_COLS,
    ).highlight_min(
        color = 'pink',
        axis = 0,
        subset = VALUE_COLS,
    ).highlight_max(
        color = 'darkgreen',
        axis = 0,
        subset = VALUE_COLS,
    ).format(
        formatter="{:.2%}",
        subset = PERCENT_COLS,
    )

    # formatted = df.style.highlight_max(
    #     color = 'lightgreen',
    #     axis = 0,
    #     subset = VALUE_COLS,
    # ).highlight_min(
    #     color = 'pink',
    #     axis = 0,
    #     subset = VALUE_COLS,
    # ).format(
    #     formatter="{:.2%}",
    #     subset = PERCENT_COLS,
    # )

    display(formatted)


def visualise_heatmap_2d(
        result: pd.DataFrame,
        parameter_1: str,
        parameter_2: str,
        metric: str,
        color_continuous_scale='Bluered_r'
) -> Figure:
    """Draw a heatmap square comparing two different parameters.

    Directly shows the resulting matplotlib figure.

    :param parameter_1:
        Y axis

    :param parameter_2:
        X axis

    :param metric:
        Value to examine

    :param result:
        Grid search results as a DataFrame.

        Created by :py:func:`analyse_grid_search_result`.

    :return:
        Plotly Figure object
    """

    df = result.reset_index().pivot(index=parameter_1, columns=parameter_2, values=metric)

    if metric in PERCENT_COLS:
        text = df.applymap(lambda x: f"{x * 100:,.2f}%")
    else:
        text = df.applymap(lambda x: f"{x:,.2f}")

    fig = px.imshow(
        df,
        labels=dict(x=parameter_2, y=parameter_1, color=metric),
        aspect="auto",
        title=metric,
        color_continuous_scale=color_continuous_scale,
    )

    fig.update_traces(text=text, texttemplate="%{text}")

    fig.update_layout(
        title={"text": metric},
        height=600,
    )
    return fig


