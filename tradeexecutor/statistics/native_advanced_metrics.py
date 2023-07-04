"""Aims to be our own native implementation of the advanced statistics for which we currently use quantstats. Code is based on quantstats."""

import pandas as pd
import numpy as np


def calculate_sharpe_ratio(
    returns: pd.Series, risk_free_rate: float = 0, periods=365, annualize: bool = True
) -> float:
    """Calculate the Sharpe ratio for the given returns.

    See https://www.investopedia.com/terms/s/sharperatio.asp

    :param returns:
        Returns of the strategy. Use py:func:`calculate_daily_returns` (tradeexecutor.visual.equity_curve) to calculate them.

    :param risk_free_rate:
        Risk free rate of return. Currently using a default of zero.

    :return:
        Sharpe ratio.
    """

    _validate(risk_free_rate, periods)

    prepared_returns = _prepare_returns(returns, risk_free_rate)

    result = prepared_returns.mean() / prepared_returns.std(ddof=1)

    if annualize:
        return _annualize_result(result, periods)
    else:
        return result


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0,
    periods: int = 365,
    annualize: bool = True,
) -> float:
    """Calculate the Sortino ratio for the given returns.

    See https://www.investopedia.com/terms/s/sortinoratio.asp

    :param Returns:
        Returns of the strategy. Use py:func:`calculate_daily_returns` (tradeexecutor.visual.equity_curve) to calculate them.

    :param risk_free_rate:
        Risk free rate of return. Currently using a default of zero.

    :return:
        Sortino ratio.
    """

    _validate(risk_free_rate, periods)

    prepared_returns = _prepare_returns(returns, risk_free_rate)

    downside = np.sqrt(
        (prepared_returns[prepared_returns < 0] ** 2).sum() / len(prepared_returns)
    )

    result = returns.mean() / downside

    if annualize:
        return _annualize_result(result, periods)
    else:
        return result


def calculate_profit_factor(returns: pd.Series) -> float:
    """Measures the profit ratio (wins/loss). Does not have to be daily returns.

    :param returns:
        Returns of the strategy. Does not have to be daily returns.

    :return:
        Profit factor."""
    return abs(returns[returns >= 0].sum() / returns[returns < 0].sum())


def _validate(risk_free_rate: float, periods: int) -> None:
    """
    Validate the given parameters.

    :param risk_free_rate:
        Risk free rate of return. Currently using a default of zero.

    :param periods:
        Periods to annualize.

    :raises:
        AssertionError if the parameters are not valid.
    """
    assert risk_free_rate >= 0, "Risk free rate must be positive."
    assert periods > 0 and type(periods) == int, "Periods must be a positive integer."


def _prepare_returns(
    returns: pd.Series, risk_free_rate: float = 0, periods: int = 365
) -> pd.Series:
    """Get the returns corrected for the risk free rate.

    :param returns:
        Returns to correct.

    :param risk_free_rate:
        Risk free rate of return. Currently using a default of zero.

    :return:
        Corrected returns.
    """

    if periods is not None:
        # deannualize
        rf = np.power(1 + risk_free_rate, 1 / periods) - 1

    if rf > 0:
        return returns - rf
    else:
        return returns


def _annualize_result(result: float, periods: int | None = None) -> float:
    """Annualize the given periods if necessary.

    :param result:
        Result to annualize.

    :param periods:
        Periods to annualize.

    :param annualize:
        Whether to annualize or not.

    :return:
        Annualized periods.
    """
    return result * np.sqrt(1 if periods is None else periods)
