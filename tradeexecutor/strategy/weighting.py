"""Weighting based portfolio manipulation.

Various helper functions to calculate weights for assets, normalise them.
"""

from typing import Dict, TypeAlias

from tradeexecutor.state.types import PairInternalId

#: Raw trading signal strength.
#:
#: E.g. raw value of the momentum.
#:
#: Negative signal indicates short.
#:
#: Can be any number between ]-inf, inf[
#:
#: Set zero for pairs that are discarded, e.g. due to risk assessment.
#:
Signal: TypeAlias = float


#: Weight of an asset.
#:
#: Represents USD allocated to this position.
#:
#: For raw weights ``0...inf``, for normalised weights ``0...1``.
#:
#: Negative signals have positive weight.
#:
Weight: TypeAlias = float


class BadWeightsException(Exception):
    """Sum of weights not 1."""


def check_normalised_weights(weights: Dict[PairInternalId, Weight], epsilon=0.0001):
    """Check that the sum of weights is good.

    - If there are any entries in weights the sum must be one

    - If the weights are empty the sum must be zero
    """

    if not weights:
        return

    total = sum(weights.values())

    if abs(total - 1) > epsilon:
        raise BadWeightsException(f"Total sum of normalised portfolio weights was not 1.0\n"
                                  f"Sum: {total}")


def clip_to_normalised(
        weights: Dict[PairInternalId, Weight],
        epsilon=0.00003,
        very_small_subtract=0.00001,
) -> Dict[int, float]:
    """If the sum of the weights are not exactly 1, then decrease the largest member to make the same sum 1 precise.

    :param weights:
        Weights where the sum is almost 1.

        A dict of pair id -> weight mappings.

    :param epsilon:
        We check that our floating point sum is within this value.

    :param very_small_subtract:
        Use this value to substract so we never go above 1, always under.

    :return:

        A dict of pair id -> fixed weight mappings.

        New weights where the largest weight have been clipped to make exactly 1
    """

    # Empty weights
    if not weights:
        return weights

    for round_substract_helper in (0, very_small_subtract):
        total = sum(weights.values())
        diff = total - 1
        largest = max(weights.items(), key=lambda x: x[1])

        clipped = largest[1] - diff - round_substract_helper

        fixed = weights.copy()
        fixed[largest[0]] = clipped

        total = sum(fixed.values())

        if total > 1:
            # We somehow still ended above one
            # Try again with more subtract
            continue

        assert abs(total - 1) < epsilon, f"Assumed all weights total is 1, got {total}, epsilon is {epsilon}"
        return fixed

    raise AssertionError("Should never happen")


def normalise_weights(weights: Dict[PairInternalId, Weight]) -> Dict[PairInternalId, Weight]:
    """Normalise weight distribution so that the sum of weights is 1."""

    total = sum(weights.values())
    normalised_weights = {}
    if total == 0:
        # Avoid division by zero
        return normalised_weights

    for key, value in weights.items():
        normalised_weights[key] = value / total

    return clip_to_normalised(normalised_weights)


def weight_by_1_slash_n(alpha_signals: Dict[PairInternalId, Signal]) -> Dict[PairInternalId, Weight]:
    """Use 1/N (position) weighting system to generate portfolio weightings from the raw alpha signals.

    - The highest alpha gets portfolio allocation 1/1

    - The second-highest alpha gets portfolio allocation 1/2

    - etc.

    More information:

    `The Fallacy of 1/N and Static Weight Allocation <https://www.r-bloggers.com/2013/06/the-fallacy-of-1n-and-static-weight-allocation/>`__.
    """
    weighed_signals = {}

    presorted_alpha = [(pair_id, abs(signal)) for pair_id, signal in alpha_signals.items()]
    sorted_alpha = sorted(presorted_alpha, key=lambda t: t[1])

    for idx, tuple in enumerate(sorted_alpha, 1):
        pair_id, alpha = tuple
        weighed_signals[pair_id] = 1 / idx
    return weighed_signals


def weight_by_1_slash_signal(alpha_signals: Dict[PairInternalId, Signal]) -> Dict[PairInternalId, Weight]:
    """Use 1/signal weighting system to generate portfolio weightings from the raw alpha signals.

    - All signals are weighted 1/signal

    - Higher the signal, smaller the weight

    - E.g. volatility weighted (more volatility, less alloaction)

    Example:

    .. code-block:: python

        alpha_model = AlphaModel(timestamp)

        available_pairs = get_included_pairs_per_month(input)
        top_pairs = available_pairs[0:parameters.max_pairs_per_month]

        assets_chosen_count = 0

        for pair in top_pairs:

            volatility = indicators.get_indicator_value("volatility", pair=pair)
            if volatility is None:
                # Back buffer has not filled up yet with enough data,
                # skip to the next pair
                continue

            if volatility >= parameters.minimum_volatility_threshold:
                # Include this pair for the ranking for each tick
                alpha_model.set_signal(
                    pair,
                    volatility,
                )
                assets_chosen_count += 1
    """
    weighed_signals = {}
    for id, value in alpha_signals.items():
        weighed_signals[id] = 1 / value
    return weighed_signals


def weight_equal(alpha_signals: Dict[PairInternalId, Signal]) -> Dict[PairInternalId, Weight]:
    """Give equal weight to every asset, regardless of the signal strength.

    :return:
        Weight map where each pair has weight 1.
    """
    weighed_signals = {}
    for idx, tuple in enumerate(alpha_signals.items(), 1):
        pair_id, alpha = tuple
        weighed_signals[pair_id] = 1
    return weighed_signals


def weight_passthrouh(alpha_signals: Dict[PairInternalId, Signal]) -> Dict[PairInternalId, Weight]:
    """Use the given raw weight value as is as the portfolio weight."""

    # Sort by pair id so we are deterministic
    items = alpha_signals.items()
    items = sorted(items, key=lambda i: i[0])

    return {pair_id: abs(signal) for pair_id, signal in items}


def weight_by_softmax(
    alpha_signals: Dict[PairInternalId, Signal],
    temperature: float = 2.0,
) -> Dict[PairInternalId, Weight]:
    """Softmax-temperature weighting that smoothly interpolates between
    equal-weight and winner-take-all allocation.

    Applies the softmax function with a temperature parameter to transform
    raw signal values into portfolio weights:

        ``w_i = exp(signal_i / T) / sum_j(exp(signal_j / T))``

    The temperature T controls concentration:

    - ``T → ∞``: converges to equal weights (1/N)
    - ``T → 0``: converges to winner-take-all (100% in top signal)
    - ``T ≈ 1-2``: moderate tilt toward higher signals

    Weights always sum to 1.0, eliminating the cash drag problem inherent
    in equal weighting under greedy allocation loops.

    **Pros:**

    - Single tunable parameter with intuitive behaviour
    - Smooth, differentiable — no hard cutoffs or discontinuities
    - Numerically stable (uses max-subtraction trick)
    - Naturally handles any number of assets

    **Cons:**

    - Sensitive to signal scale — signals should be comparable magnitude
    - Low temperatures can over-concentrate into noisy top signals
    - Does not account for correlation between assets

    References:

    - `Zakamulin (2025), "Entropy-Regularized Portfolio Selection via
      Softmax Sharpe Allocation", SSRN 5539560
      <https://doi.org/10.2139/ssrn.5539560>`_
    - `Britten-Jones (1999), "The Sampling Error in Estimates of
      Mean-Variance Efficient Portfolio Weights", Journal of Finance
      <https://doi.org/10.1111/0022-1082.00120>`_
    - `Goodfellow, Bengio & Courville (2016), "Deep Learning",
      Ch. 6.2.2 — softmax as generalisation of logistic function
      <https://www.deeplearningbook.org/contents/mlp.html>`_

    :param alpha_signals:
        Signal objects keyed by pair ID.

    :param temperature:
        Controls equal↔concentrated tradeoff.
        Higher = more equal, lower = more concentrated.

    :return:
        Weight map summing to 1.0.
    """
    import math

    if not alpha_signals:
        return {}

    signals = {k: abs(v) for k, v in alpha_signals.items()}
    max_s = max(signals.values())

    # Numerically stable softmax: subtract max before exp
    exp_signals = {k: math.exp((v - max_s) / temperature) for k, v in signals.items()}
    total = sum(exp_signals.values())

    return {k: v / total for k, v in exp_signals.items()}


def weight_by_blend(
    alpha_signals: Dict[PairInternalId, Signal],
    blend_alpha: float = 0.5,
) -> Dict[PairInternalId, Weight]:
    """Linear blend of equal-weight and signal-proportional allocation.

    Computes weights as:

        ``w_i = alpha * (1/N) + (1 - alpha) * (signal_i / sum_j(signal_j))``

    This is equivalent to James-Stein shrinkage applied to the portfolio
    weight vector, shrinking signal-proportional weights toward the
    equal-weight prior. The literature consistently shows that shrinkage
    estimators outperform both pure equal-weight and pure optimised
    portfolios out of sample.

    The blend parameter alpha controls the shrinkage intensity:

    - ``alpha=1.0``: pure equal weight
    - ``alpha=0.0``: pure signal-proportional (same as :py:func:`weight_passthrouh`)
    - ``alpha=0.5``: 50/50 blend (recommended starting point)

    **Pros:**

    - Dead simple — one-line formula, easy to reason about
    - Grounded in shrinkage estimation theory (James & Stein, 1961)
    - Robust to signal noise — always partially diversified
    - Weights sum to 1.0

    **Cons:**

    - Linear blending may not be optimal — equal weight is a crude prior
    - Does not adapt to signal dispersion (same alpha regardless of
      whether signals are tightly clustered or widely spread)

    References:

    - `James & Stein (1961), "Estimation with Quadratic Loss",
      Proc. 4th Berkeley Symposium
      <https://projecteuclid.org/proceedings/berkeley-symposium-on-mathematical-statistics-and-probability/Proceedings-of-the-Fourth-Berkeley-Symposium-on-Mathematical-Statistics-and/Chapter/Estimation-with-Quadratic-Loss/bsmsp/1200512173>`_
    - `Jorion (1986), "Bayes-Stein Estimation for Portfolio Analysis",
      Journal of Financial and Quantitative Analysis
      <https://doi.org/10.2307/2331042>`_
    - `DeMiguel, Garlappi & Uppal (2009), "Optimal Versus Naive
      Diversification: How Inefficient is the 1/N Portfolio Strategy?",
      Review of Financial Studies — shows 1/N is hard to beat
      <https://doi.org/10.1093/rfs/hhm075>`_
    - `Carver (2015), "Systematic Trading", Ch. 11 — handcrafting
      portfolio weights as shrinkage toward equal weight
      <https://qoppac.blogspot.com/2018/12/portfolio-construction-through.html>`_

    :param alpha_signals:
        Signal objects keyed by pair ID.

    :param blend_alpha:
        Shrinkage intensity. 1.0 = pure equal, 0.0 = pure signal.

    :return:
        Weight map summing to 1.0.
    """
    if not alpha_signals:
        return {}

    n = len(alpha_signals)
    equal_w = 1.0 / n

    signals = {k: abs(v) for k, v in alpha_signals.items()}
    total_signal = sum(signals.values())

    if total_signal == 0:
        return {k: equal_w for k in alpha_signals}

    return {
        k: blend_alpha * equal_w + (1 - blend_alpha) * (v / total_signal)
        for k, v in signals.items()
    }


def weight_by_log(alpha_signals: Dict[PairInternalId, Signal]) -> Dict[PairInternalId, Weight]:
    """Use logarithmic weighting to dampen high signals.

    - Applies log(1 + signal) to each signal's absolute value

    - Higher signals still get more weight, but the relationship is logarithmic
      rather than linear

    - Useful when raw signal values span a wide range and you don't want
      top signals to dominate the portfolio
    """
    import math

    items = sorted(alpha_signals.items(), key=lambda i: i[0])
    return {pair_id: math.log(1 + abs(signal)) for pair_id, signal in items}
