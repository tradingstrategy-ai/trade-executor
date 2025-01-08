"""Tagging live strategies."""

import enum


class StrategyTag(enum.Enum):
    """Tags we can for a strategy.

    - Tags give the context of the strategy and its life cycle for the users
      and the development team

    - Tags are shown in the strategy explorer and are a sorting criteria
      for displaying strategies to the user: live > beta > alpha > prototype

    - Some strategy functionality e.g. displaying the risk metrics
      and additional disclaimer depends on the tags
    """

    #: This strategy module is part of internal unit test suite
    unit_testing = "unit_testing"

    #: Testing strategy in forward-testing
    alpha = "alpha"

    #: Users can deposit
    beta = "beta"

    #: The strategy is expected to make profit and has enough history to show this
    live = "live"

    # The strategy is archived and should not be visible on the strategy index by default
    archived = "archived"

    #: The strategy is not expected to make profit, but is only running as an infrastructure test
    internal_testing = "internal_testing"

    #: The strategy should appear on the front page hero box showcasing strategies
    hero = "hero"
