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


def optimise_probabilistic_sharpe(result: GridSearchResult) -> OptimiserSearchResult:
    """Search for the best probabilistic Sharpe ratio.

    Probabilistic Sharpe Ratio (PSR) converts the observed Sharpe ratio into
    a confidence-style probability that the true Sharpe exceeds a chosen
    threshold. This makes it a useful optimisation target when you care more
    about statistical credibility than raw point estimates.

    For glossary definitions of PSR and related risk metrics, see
    https://tradingstrategy.ai/glossary.
    """
    psr = result.get_metric("Prob. Sharpe Ratio")
    if pd.isna(psr):
        return OptimiserSearchResult(0, negative=False)
    return OptimiserSearchResult(-float(psr), negative=True)


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


def optimise_calmar(result: GridSearchResult) -> OptimiserSearchResult:
    """Search for the best Calmar ratio (CAGR / |max drawdown|).

    The Calmar ratio was introduced by Terry W. Young in 1991 and named after
    his California Managed Accounts Reports newsletter. It measures the
    tradeoff between compounded growth and the worst peak-to-trough loss
    experienced during the backtest.

    Theory and when to use:

    - The Calmar ratio directly addresses the common problem where optimising
      for Sharpe produces flat equity curves (the optimiser avoids trading to
      minimise volatility) whilst optimising for CAGR produces excessive
      drawdowns (growth at any cost).

    - By dividing CAGR by the worst drawdown, the optimiser is incentivised
      to find strategies that grow steadily without catastrophic losses.
      A Calmar ratio above 1.0 means annualised returns exceed the worst
      drawdown — generally considered good for systematic strategies.

    - Particularly well suited for vault/index strategies where investor
      psychology makes large drawdowns unacceptable even if long-term
      returns are high.

    - Limitation: relies on a single worst drawdown event. A strategy with
      one bad week but otherwise excellent performance will score poorly.
      For a metric that accounts for the pattern of drawdowns rather than
      just the worst, see :py:func:`optimise_ulcer_performance`.

    Further reading:

    - Young, T.W. (1991). "Calmar Ratio: A Smoother Tool." Futures Magazine.
    - https://en.wikipedia.org/wiki/Calmar_ratio
    - https://www.investopedia.com/terms/c/calmarratio.asp

    Example:

    .. code-block:: python

        from tradeexecutor.backtest.optimiser_functions import optimise_calmar

        optimised_results = perform_optimisation(
            iterations=8,
            search_func=optimise_calmar,
            ...
        )
    """
    cagr = result.get_cagr()
    max_dd = abs(result.get_max_drawdown())
    if pd.isna(cagr) or pd.isna(max_dd) or max_dd < 0.001:
        return OptimiserSearchResult(0, negative=False)
    return OptimiserSearchResult(-(cagr / max_dd), negative=True)


def optimise_gain_to_pain(result: GridSearchResult) -> OptimiserSearchResult:
    """Search for the best gain-to-pain ratio.

    The gain-to-pain ratio (GPR) was popularised by Jack Schwager in his
    "Market Wizards" series. It is the sum of all positive returns divided
    by the absolute sum of all negative returns over the backtest period.

    Theory and when to use:

    - Unlike Sharpe/Sortino which summarise the return distribution with
      mean and standard deviation (assuming approximate normality), GPR
      uses the raw mass of gains versus losses. This makes it non-parametric
      and robust to fat tails, skewed distributions, and other non-normal
      return characteristics common in crypto/DeFi strategies.

    - A GPR > 1.0 means total gains exceed total losses.
      A GPR > 2.0 means the strategy earns twice as much as it loses.

    - Good fit for strategies with non-normal return profiles, such as
      momentum/trend-following strategies that have many small losses
      and fewer large wins (positive skew). Sharpe penalises the upside
      volatility of such strategies; GPR does not.

    - Limitation: ignores the timing and sequence of gains and losses.
      A strategy could have all its losses concentrated in one period
      (a severe drawdown) and still score well if total gains are high.

    Further reading:

    - Schwager, J.D. (2012). "Market Wizards." John Wiley & Sons.
    - https://en.wikipedia.org/wiki/Gain%E2%80%93pain_ratio
    - https://analyzingalpha.com/gain-to-pain-ratio

    Example:

    .. code-block:: python

        from tradeexecutor.backtest.optimiser_functions import optimise_gain_to_pain

        optimised_results = perform_optimisation(
            iterations=8,
            search_func=optimise_gain_to_pain,
            ...
        )
    """
    gpr = result.get_metric("Gain/Pain Ratio")
    if pd.isna(gpr):
        return OptimiserSearchResult(0, negative=False)
    return OptimiserSearchResult(-float(gpr), negative=True)


def optimise_ulcer_index(result: GridSearchResult) -> OptimiserSearchResult:
    """Search for the lowest Ulcer Index.

    Unlike standard deviation based metrics, Ulcer Index measures the depth
    and duration of drawdowns only. Lower values are better.

    For glossary definitions of Ulcer Index and drawdown terms, see
    https://tradingstrategy.ai/glossary.
    """
    ulcer = result.get_metric("Ulcer Index")
    if pd.isna(ulcer):
        return OptimiserSearchResult(0, negative=False)
    return OptimiserSearchResult(float(ulcer), negative=False)


def optimise_ulcer_performance(result: GridSearchResult) -> OptimiserSearchResult:
    """Search for the best Ulcer performance index (return / Ulcer index).

    The Ulcer index was created by Peter Martin and Byron McCann in 1987,
    originally for mutual fund analysis. The Ulcer performance index (UPI)
    divides excess return by the Ulcer index, analogous to how the Sharpe
    ratio divides excess return by standard deviation — but using only
    downside risk.

    Theory and when to use:

    - The Ulcer index is the root-mean-square of percentage drawdowns from
      recent peaks. By squaring drawdowns before averaging, it penalises
      both the **depth** and **duration** of drawdowns — a long shallow
      drawdown is penalised similarly to a short deep one.

    - This addresses a key weakness of the Sharpe ratio: Sharpe treats
      upside and downside volatility equally, so a strategy that sits in
      cash (zero volatility) scores well. The Ulcer index only measures
      pain from being underwater, so the optimiser must generate returns
      to score well.

    - Particularly good fit for long-only portfolio/index strategies
      (like vault-of-vaults) where investors experience drawdowns
      asymmetrically — the pain of a 20% loss is felt more acutely
      than the joy of a 20% gain.

    - Compared to Calmar (which uses only the single worst drawdown),
      UPI accounts for the entire drawdown history. A strategy with
      many moderate drawdowns will score worse than one with a single
      bad day followed by smooth recovery.

    - Limitation: ignores upside volatility entirely, which may not suit
      strategies designed to capture explosive upside moves.

    Further reading:

    - Martin, P. & McCann, B. (1989). "The Investor's Guide to Fidelity Funds."
    - https://en.wikipedia.org/wiki/Ulcer_index
    - https://www.investopedia.com/terms/u/ulcerindex.asp
    - https://portfoliooptimizer.io/blog/ulcer-performance-index-optimization/
    - https://tradingstrategy.ai/glossary

    Example:

    .. code-block:: python

        from tradeexecutor.backtest.optimiser_functions import optimise_ulcer_performance

        optimised_results = perform_optimisation(
            iterations=8,
            search_func=optimise_ulcer_performance,
            ...
        )
    """
    ulcer = result.get_metric("Ulcer Index")
    cagr = result.get_cagr()
    if pd.isna(ulcer) or pd.isna(cagr) or ulcer < 0.001:
        return OptimiserSearchResult(0, negative=False)
    return OptimiserSearchResult(-(cagr / ulcer), negative=True)


def optimise_cvar(result: GridSearchResult) -> OptimiserSearchResult:
    """Search for the lowest absolute conditional value at risk.

    cVaR, also called Expected Shortfall, measures average losses in the tail
    once returns breach the VaR threshold. Lower absolute tail loss is better.

    For glossary definitions of cVaR and tail-risk metrics, see
    https://tradingstrategy.ai/glossary.
    """
    cvar = result.get_metric("Expected Shortfall (cVaR)")
    if pd.isna(cvar):
        return OptimiserSearchResult(0, negative=False)
    return OptimiserSearchResult(abs(float(cvar)), negative=False)


def optimise_recovery_factor(result: GridSearchResult) -> OptimiserSearchResult:
    """Search for the best recovery factor.

    Recovery Factor compares cumulative profit to the magnitude of the worst
    drawdown, rewarding strategies that earn back losses efficiently after a
    setback.

    For glossary definitions of recovery factor and drawdown metrics, see
    https://tradingstrategy.ai/glossary.
    """
    recovery = result.get_metric("Recovery Factor")
    if pd.isna(recovery):
        return OptimiserSearchResult(0, negative=False)
    return OptimiserSearchResult(-float(recovery), negative=True)


class BalancedCAGRAndSharpeOptimisationFunction:
    """Weighted blend of normalised CAGR and Sharpe ratio.

    This composite objective function addresses the fundamental tension
    between growth maximisation (CAGR) and risk-adjusted performance
    (Sharpe). Rather than optimising for one metric and accepting poor
    outcomes on the other, it normalises both to a [0, 1] scale and
    combines them with configurable weights.

    Theory and when to use:

    - Multi-objective optimisation is well established in portfolio
      construction theory (Markowitz, 1952). When two objectives conflict,
      the Pareto-optimal solutions form a frontier. This function collapses
      the frontier into a single scalar by weighting the objectives,
      letting you choose where on the frontier to land.

    - With ``cagr_weight=0.6, sharpe_weight=0.4`` (the default), the
      optimiser favours growth but with meaningful risk penalisation.
      Adjust towards more Sharpe weight for conservative strategies,
      or more CAGR weight for growth-oriented ones.

    - This is a good starting point when you are unsure which single
      metric to optimise for. By running multiple optimisations with
      different weight configurations, you can map out the
      risk-return tradeoff for your strategy.

    - Compared to the existing
      :py:class:`BalancedSharpeAndMaxDrawdownOptimisationFunction`,
      this blends growth (CAGR) with risk quality (Sharpe) rather than
      risk quality with worst-case loss (max drawdown). Use this when
      you want the optimiser to actively seek returns rather than just
      avoid losses.

    Further reading:

    - Markowitz, H. (1952). "Portfolio Selection." The Journal of Finance.
    - https://en.wikipedia.org/wiki/Multi-objective_optimization
    - https://www.quantconnect.com/docs/v2/cloud-platform/optimization/objectives

    Example:

    .. code-block:: python

        from tradeexecutor.backtest.optimiser_functions import BalancedCAGRAndSharpeOptimisationFunction

        # Growth-biased blend
        search_func = BalancedCAGRAndSharpeOptimisationFunction(
            cagr_weight=0.7,
            sharpe_weight=0.3,
        )

        # Conservative blend
        search_func = BalancedCAGRAndSharpeOptimisationFunction(
            cagr_weight=0.4,
            sharpe_weight=0.6,
        )

        optimised_results = perform_optimisation(
            iterations=8,
            search_func=search_func,
            ...
        )
    """

    def __init__(
        self,
        cagr_weight: Percent = 0.6,
        sharpe_weight: Percent = 0.4,
        max_cagr: float = 5.0,
        max_sharpe: float = 3.0,
    ):
        """
        :param cagr_weight:
            Weight for normalised CAGR in the combined score. Default 0.6.

        :param sharpe_weight:
            Weight for normalised Sharpe in the combined score. Default 0.4.

        :param max_cagr:
            CAGR values are clamped to this ceiling before normalising.
            Default 5.0 (500% annualised).

        :param max_sharpe:
            Sharpe values are clamped to this ceiling before normalising.
            Default 3.0.
        """
        self.cagr_weight = cagr_weight
        self.sharpe_weight = sharpe_weight
        self.max_cagr = max_cagr
        self.max_sharpe = max_sharpe

        assert abs(self.cagr_weight + self.sharpe_weight - 1.0) < 1e-9, \
            f"Weights must sum to 1.0, got {self.cagr_weight} + {self.sharpe_weight}"

    def __call__(self, result: GridSearchResult) -> OptimiserSearchResult:
        cagr = result.get_cagr()
        sharpe = result.get_sharpe()

        if pd.isna(cagr):
            cagr = 0.0
        if pd.isna(sharpe):
            sharpe = 0.0

        normalised_cagr = np.clip(cagr, -self.max_cagr, self.max_cagr) / self.max_cagr
        normalised_sharpe = np.clip(sharpe, -self.max_sharpe, self.max_sharpe) / self.max_sharpe

        combined = normalised_cagr * self.cagr_weight + normalised_sharpe * self.sharpe_weight

        return OptimiserSearchResult(-combined, negative=True)
