"""Weighting based portfolio manipulation."""

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


def clip_to_normalised(weights: Dict[int, float]) -> Dict[int, float]:
    """If the sum of the weights are not exactly 1, then decrease the largest member to make the same sum 1 precise.

    """

    # Empty weights
    if not weights:
        return weights

    total = sum(weights.values())
    diff = total - 1
    largest = max(weights.items(), key=lambda x: x[1])

    clipped = largest[1] - diff

    fixed = weights.copy()
    fixed[largest[0]] = clipped

    total = sum(fixed.values())
    assert total == 1
    return fixed


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
    for idx, tuple in enumerate(alpha_signals, 1):
        pair_id, alpha = tuple
        weighed_signals[pair_id] = 1 / idx
    return weighed_signals
