"""Curve visualisation helpers.

- Helpers for equity curves, returns, etc. series

- We create various curves and transform them, so we define some helpful attributes in `pd.Series.attrs`
  so we do not need to manually copy and pass this data around, shorteting the amount of code

"""
from enum import Enum


#: List of attributes we support in pd.Series
#:
#: This are dragged along in pd.Series.attrib dict to make
#: data logistics easier.
#:
#: - Name: series name e.g. "Strategy", "BTC"
#: - Colour: e.g. "black"
#: - Curve: e.g. `CurveType.equity`
#: - Period e.g. "D" - Pandas frequency moniker
#:
#: Aggregated returns should be `CurveType.returns` with its period set.
#:
SPECIAL_SERIES_ATTRIBUTES = {"name", "colour", "curve", "period"}

#: Default branc colours for benchmark asset lines
#:
#: This is set as `pd.Series.attrs["colour"]
#:
#: See :py:func:`tradeexecutor.visual.bnechmark.visualise_equity_curves`
#:
#:
#: See https://community.plotly.com/t/plotly-colours-list/11730/3
#:
DEFAULT_BENCHMARK_COLOURS = {
    "BTC": "orange",
    "BTC.e": "orange",
    "cbBTC": "orange",
    "ETH": "blue",
    "ETH.e": "blue",
    "MATIC": "purple",
    "SOL": "lightblue",
    "ARB": "red",
    "AVAX": "red",
    "DOGE": "darkorange",
    "PEPE": "darkmagenta",
    "AAVE": "#F289DA",
    "MKR": "#1AAB9B",
    "All cash": "black",
    "Strategy": "green",
}


class CurveType(Enum):
    """Different curve types.

    - This is set as `pd.Series.attrs["curve"]

    """

    #: Cumulate returns on initial cash.
    #:
    #: E.g. for $10,000 invested
    #:
    equity = "equity"

    #: Returns for 1.0 as the initial investment
    #:
    cumulative_returns = "cumulative_returns"

    #: Returns per event
    #:
    #: E.g. daily returns.
    #:
    #: If it is aggregated returns, then `pd.Series.attrs["period"]` should be set
    #:
    returns = "returns"
