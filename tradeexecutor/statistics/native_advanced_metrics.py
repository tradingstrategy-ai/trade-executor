"""Aims to be our own native implementation of the advanced statistics for which we currently use quantstats. Code is based on quantstats."""

import pandas as pd
import numpy as np


def _ratio_decorator(ratio_func):
    """Decorator for the ratio functions. It prepares the returns and annualizes the result if necessary.
    
    :param ratio_func:
        The ratio function to decorate.
        
    :return:
        The decorated function.
    """
    def wrapper(returns: pd.Series, risk_free_rate: float = 0, periods = 365, annualize: bool = True) -> float:
        assert risk_free_rate >= 0, "Risk free rate must be positive."
        prepared_returns = _prepare_returns(returns, risk_free_rate)
        result = ratio_func(prepared_returns, risk_free_rate, periods, annualize)
        return _annualize_result(result, periods, annualize) if annualize else result
    return wrapper
    

@_ratio_decorator
def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0, periods = 365, annualize: bool = True) -> float:
    """Calculate the Sharpe ratio for the given returns.

    See https://www.investopedia.com/terms/s/sharperatio.asp

    :param returns:
        Returns of the strategy. Use py:func:`calculate_daily_returns` (tradeexecutor.visual.equity_curve) to calculate them.

    :param risk_free_rate:
        Risk free rate of return. Currently using a default of zero.

    :return:
        Sharpe ratio.
    """

    return returns.mean() / returns.std(ddof=1)

    
@_ratio_decorator
def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0, periods = 365, annualize: bool = True) -> float:
    """Calculate the Sortino ratio for the given returns.
    
    See https://www.investopedia.com/terms/s/sortinoratio.asp

    :param Returns:
        Returns of the strategy. Use py:func:`calculate_daily_returns` (tradeexecutor.visual.equity_curve) to calculate them.

    :param risk_free_rate:
        Risk free rate of return. Currently using a default of zero.

    :return:
        Sortino ratio.
    """

    downside = np.sqrt((returns[returns < 0] ** 2).sum() / len(returns))

    return returns.mean() / downside



def calculate_profit_factor(returns: pd.Series) -> float:
    """Measures the profit ratio (wins/loss). Does not have to be daily returns.
    
    :param returns:
        Returns of the strategy. Does not have to be daily returns.
        
    :return:
        Profit factor."""
    return abs(returns[returns >= 0].sum() / returns[returns < 0].sum())


def _prepare_returns(returns: pd.Series, risk_free_rate: float = 0) -> pd.Series:
    """Get the returns corrected for the risk free rate.

    :param returns:
        Returns to correct.

    :param risk_free_rate:
        Risk free rate of return. Currently using a default of zero.

    :return:
        Corrected returns.
    """

    if risk_free_rate > 0:
        return returns - risk_free_rate
    else:
        return returns
    

def _annualize_result(result: float, periods: int, annualize: bool) -> float:
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
    return result * np.sqrt(1 if annualize else periods)




