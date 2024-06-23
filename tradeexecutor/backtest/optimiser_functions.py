"""Functions the optimiser would be looking for.

- You can also write your own optimiser functions
"""

from .optimiser import GridSearchResult, SearchResult


def optimise_profit(result: GridSearchResult) -> SearchResult:
    """Search for the best CAGR value."""
    return SearchResult(-result.get_cagr(), negative=True)


def optimise_sharpe(result: GridSearchResult) -> SearchResult:
    """Search for the best Sharpe value."""
    return SearchResult(-result.get_sharpe(), negative=True)


def optimise_win_rate(result: GridSearchResult) -> SearchResult:
    """Search for the best trade win rate."""
    return SearchResult(-result.get_win_rate(), negative=True)


def optimise_max_drawdown(result: GridSearchResult) -> SearchResult:
    """Search for the lowest max drawdown.

    - Return absolute value of drawdown (negative sign removed).

    - Lower is better.
    """
    return SearchResult(abs(result.get_max_drawdown()), negative=False)


def optimise_sharpe_and_max_drawdown_ratio(result: GridSearchResult) -> SearchResult:
    """Search for the best sharpe / max drawndown ratio.

    - One of the attempts to try to find "balanced" strategies that do not
      take risky trades, but rather sit on the cash (which can be used elsewhere)

    - Search combined sharpe / max drawdown ratio.

    - Higher is better.

    - See also :py:func:`BalancedSharpeAndMaxDrawdownOptimisationFunction`
    """
    return SearchResult(-(result.get_sharpe() / abs(result.get_max_drawdown())), negative=True)


class BalancedSharpeAndMaxDrawdownOptimisationFunction:
    """Try to find a strategy with balanced Sharpe and max drawdown.

    - Both max drawdown and sharpe are giving weights (by default 50%)

    - Try to find a result where both of these varibles are maxed out

    - You can weight one more than other

    - See also :py:func:`optimise_sharpe_and_max_drawdown_ratio`
    """

    def __init__(
        self,
        sharpe_weight: Percent =0.5,
        max_drawdown_weight: Percent =0.5,
        max_sharpe: float =3.0,
        epsilon=0.01,
    ):
        self.sharpe_weight = sharpe_weight
        self.max_drawdown_weight = max_drawdown_weight
        self.max_sharpe = max_sharpe
        self.epsilon = epsilon

    def __call__(self, result: GridSearchResult) -> SearchResult:
        normalised_max_drawdown = 1 + result.get_max_drawdown()  # 0 drawdown get value of max 1
        normalised_sharpe = min(result.get_sharpe(), self.max_sharpe) / self.max_sharpe  # clamp sharpe to 3
        total_normalised = normalised_max_drawdown * self.max_drawdown_weight + self.normalised_sharpe * self.sharpe_weight
        assert total_normalised > 0, f"Got {total_normalised}"
        assert total_normalised < 1 + self.epsilon, f"Got {total_normalised} with normalised sharpe: {normalised_sharpe} and normalised max drawdown {normalised_max_drawdown}, weights sharpe: {self.sharpe_weight} / dd: {self.max_drawdown_weight}"
        return total_normalised


