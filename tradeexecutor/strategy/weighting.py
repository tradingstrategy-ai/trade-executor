"""Weighting based portfolio manipulation.

Various helper functions to calculate weights for assets, normalise them.
"""

from typing import Dict


class BadWeightsException(Exception):
    """Sum of weights not 1."""


def check_normalised_weights(weights: Dict[int, float], epsilon=0.0001):
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
        weights: Dict[int, float],
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


def normalise_weights(weights: Dict[int, float]) -> Dict[int, float]:
    """Normalise weight distribution so that the sum of weights is 1."""

    total = sum(weights.values())
    normalised_weights = {}
    for key, value in weights.items():
        normalised_weights[key] = value / total

    return clip_to_normalised(normalised_weights)


def weight_by_1_slash_n(alpha_signals: Dict[int, float]) -> Dict[int, float]:
    """Use 1/N weighting system to generate portfolio weightings from the raw alpha signals.

    - The highest alpha gets portfolio allocation 1/1

    - The second-highest alpha gets portfolio allocation 1/2

    - etc.

    More information:

    `The Fallacy of 1/N and Static Weight Allocation <https://www.r-bloggers.com/2013/06/the-fallacy-of-1n-and-static-weight-allocation/>`__.
    """
    weighed_signals = {}
    for idx, tuple in enumerate(alpha_signals.items(), 1):
        pair_id, alpha = tuple
        weighed_signals[pair_id] = 1 / idx
    return weighed_signals


def weight_passthrouh(alpha_signals: Dict[int, float]) -> Dict[int, float]:
    """Use the given raw weight value as is as the portfolio weight."""
    return alpha_signals

