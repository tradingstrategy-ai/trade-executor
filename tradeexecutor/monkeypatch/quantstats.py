import pandas as _pd
import numpy as _np

from quantstats import utils as _utils
from quantstats import stats as _stats
from quantstats.stats import comp

# CAGR calculation periods cannot be passed through the stack,
# so we hardcode it for crypto year
def _fixed_cagr(returns, rf=0.0, compounded=True, periods=365):
    """
    Calculates the communicative annualized growth return
    (CAGR%) of access returns

    If rf is non-zero, you must specify periods.
    In this case, rf is assumed to be expressed in yearly (annualized) terms
    """
    total = _utils._prepare_returns(returns, rf)
    if compounded:
        total = comp(total)
    else:
        total = _np.sum(total)

    years = (returns.index[-1] - returns.index[0]).days / periods

    res = abs(total + 1.0) ** (1.0 / years) - 1

    if isinstance(returns, _pd.DataFrame):
        res = _pd.Series(res)
        res.index = returns.columns

    return res

_stats.cagr = _fixed_cagr