"""Functions the optimiser would be looking for.

- You can also write your own optimiser functions, see :py:class:`tradeexecutor.backtest.optimiser.SearchFunction`.

Example:

.. code-block:: python

    import logging
    from tradeexecutor.backtest.optimiser import perform_optimisation
    from tradeexecutor.backtest.optimiser import prepare_optimiser_parameters
    from tradeexecutor.backtest.optimiser_functions import optimise_profit, optimise_sharpe
    from tradeexecutor.backtest.optimiser import MinTradeCountFilter

    # How many Gaussian Process iterations we do
    iterations = 6

    optimised_results = perform_optimisation(
        iterations=iterations,
        search_func=optimise_profit,
        decide_trades=decide_trades,
        strategy_universe=strategy_universe,
        parameters=prepare_optimiser_parameters(Parameters),  # Handle scikit-optimise search space
        create_indicators=create_indicators,
        result_filter=MinTradeCountFilter(50)
        # Uncomment for diagnostics
        # log_level=logging.INFO,
        # max_workers=1,
    )

    print(f"Optimise completed, optimiser searched {optimised_results.get_combination_count()} combinations")
"""

import numpy as np
import pandas as pd

from .optimiser import GridSearchResult, OptimiserSearchResult
from ..state.types import Percent
from ..visual.equity_curve import calculate_rolling_sharpe


def optimise_profit(result: GridSearchResult) -> OptimiserSearchResult:
    """Search for the best CAGR value."""
    return OptimiserSearchResult(-result.get_cagr(), negative=True)


def optimise_sharpe(result: GridSearchResult) -> OptimiserSearchResult:
    """Search for the best Sharpe value."""
    return OptimiserSearchResult(-result.get_sharpe(), negative=True)


def optimise_sortino(result: GridSearchResult) -> OptimiserSearchResult:
    """Search for the best Sortino value."""
    return OptimiserSearchResult(-result.get_sortino(), negative=True)


def optimise_win_rate(result: GridSearchResult) -> OptimiserSearchResult:
    """Search for the best trade win rate."""
    return OptimiserSearchResult(-result.get_win_rate(), negative=True)


def optimise_max_drawdown(result: GridSearchResult) -> OptimiserSearchResult:
    """Search for the lowest max drawdown.

    - Return absolute value of drawdown (negative sign removed).

    - Lower is better.
    """
    return OptimiserSearchResult(abs(result.get_max_drawdown()), negative=False)


def optimise_sharpe_and_max_drawdown_ratio(result: GridSearchResult) -> OptimiserSearchResult:
    """Search for the best sharpe / max drawndown ratio.

    - One of the attempts to try to find "balanced" strategies that do not
      take risky trades, but rather sit on the cash (which can be used elsewhere)

    - Search combined sharpe / max drawdown ratio.

    - Higher is better.

    - See also :py:func:`BalancedSharpeAndMaxDrawdownOptimisationFunction`
    """
    return OptimiserSearchResult(-(result.get_sharpe() / abs(result.get_max_drawdown())), negative=True)


class BalancedSharpeAndMaxDrawdownOptimisationFunction:
    """Try to find a strategy with balanced Sharpe and max drawdown.

    - Both max drawdown and sharpe are giving weights (by default 50%)

    - Try to find a result where both of these varibles are maxed out

    - You can weight one more than other

    - See also :py:func:`optimise_sharpe_and_max_drawdown_ratio`

    Example:

    .. code-block:: python

        import logging
        from tradeexecutor.backtest.optimiser import perform_optimisation
        from tradeexecutor.backtest.optimiser import prepare_optimiser_parameters
        from tradeexecutor.backtest.optimiser_functions import optimise_profit, optimise_sharpe, BalancedSharpeAndMaxDrawdownOptimisationFunction
        from tradeexecutor.backtest.optimiser import MinTradeCountFilter

        # How many Gaussian Process iterations we do
        iterations = 8

        optimised_results = perform_optimisation(
            iterations=iterations,
            search_func=BalancedSharpeAndMaxDrawdownOptimisationFunction(sharpe_weight=0.75, max_drawdown_weight=0.25),
            decide_trades=decide_trades,
            strategy_universe=strategy_universe,
            parameters=prepare_optimiser_parameters(Parameters),  # Handle scikit-optimise search space
            create_indicators=create_indicators,
            result_filter=MinTradeCountFilter(150),
            timeout=20*60,
            # Uncomment for diagnostics
            # log_level=logging.INFO,
            # max_workers=1,
        )

        print(f"Optimise completed, optimiser searched {optimised_results.get_combination_count()} combinations")
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

        assert self.sharpe_weight + self.max_drawdown_weight == 1

    def __call__(self, result: GridSearchResult) -> OptimiserSearchResult:
        normalised_max_drawdown = 1 + result.get_max_drawdown()  # 0 drawdown get value of max 1
        normalised_sharpe = min(result.get_sharpe(), self.max_sharpe) / self.max_sharpe  # clamp sharpe to 3
        total_normalised = normalised_max_drawdown * self.max_drawdown_weight + normalised_sharpe * self.sharpe_weight

        if pd.isna(total_normalised):
            total_normalised = 0
        error_message = f"Got {total_normalised} with normalised sharpe: {normalised_sharpe} and normalised max drawdown {normalised_max_drawdown}\nWeights sharpe: {self.sharpe_weight} / dd: {self.max_drawdown_weight}.\nRaw sharpe: {result.get_sharpe()}, raw max downdown: {result.get_max_drawdown()}"
        # Total normalised is allowed to go below zero if Sharpe is negative (loss making strategy)
        # assert total_normalised > 0, error_message
        assert total_normalised < 1 + self.epsilon, error_message
        return OptimiserSearchResult(-total_normalised, negative=True)



class RollingSharpeOptimisationFunction:
    """Find a rolling sharpe that's stable and high.

    - Rolling sharpe is not volatile but a constant line

    - This means the strategy produces constant results over the time

    - Higher rolling sharpe is better

    """

    def __init__(self, rolling_sharpe_window_days=180):
        self.rolling_sharpe_window_days = rolling_sharpe_window_days

    def __call__(self, result: GridSearchResult) -> OptimiserSearchResult:

        rolling_sharpe = calculate_rolling_sharpe(
            result.returns,
            freq="D",
            periods=self.rolling_sharpe_window_days,
        )

        # The ratio of mean divided by standard deviation is known by different names depending on the context, but it's most commonly referred to as the following:
        #
        # Coefficient of Variation (CV): This term is used when both the mean and standard deviation are positive. It's often expressed as a percentage.
        # Signal-to-Noise Ratio (SNR): This term is used in signal processing and statistics.
        #
        # In some contexts, particularly in finance, the inverse (standard deviation divided by mean) is called the Coefficient of Variation.
        # Interpretation of high and low values:
        # High values (mean >> standard deviation):
        #
        # Indicate that the mean is large relative to the variability in the data.
        # Suggest more consistent or stable data.
        # In finance, could indicate better risk-adjusted returns.
        # In signal processing, suggest a clearer signal relative to noise.
        #
        # Low values (mean << standard deviation):
        #
        # Indicate high variability relative to the mean.
        # Suggest more dispersed or volatile data.
        # In finance, could indicate worse risk-adjusted returns.
        # In signal processing, suggest a weaker signal relative to noise.
        #
        # It's important to note that the interpretation can vary depending on the specific field and context. For example:
        #
        # In manufacturing quality control, a lower CV typically indicates better process control.
        # In investment, a higher Sharpe ratio (which is based on this concept) indicates better risk-adjusted returns.
        # In experimental sciences, a lower CV might indicate more precise measurements.

        value = np.mean(rolling_sharpe) / np.std(rolling_sharpe)

        return OptimiserSearchResult(-value, negative=True)
