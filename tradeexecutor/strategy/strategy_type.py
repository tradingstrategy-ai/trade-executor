"""Declarate what kind of strategy types our strategy loader and backtesting framework supports.

#: See :ref:`strategy types` for more information.
"""

import enum


class StrategyType(enum.Enum):
    """Which kind of strategy types we support."""

    #: Pandas + position manager based strategy.
    #: Uses position_manager instance to tell what trades to do.
    #: The strategy contains a `decide_trades` function that
    #: takes strategy cycle timestamp, trading universe and state as an input.
    #: See :ref:`strategy types` for more information.
    managed_positions = "managed_positions"

    #: Alpha model based strategy.
    #: Return alpha weightings of assets it wants to buy.
    #: This is what old QSTrader based strategies do.
    #: **Deprecated**
    alpha_model = "alpha_model"