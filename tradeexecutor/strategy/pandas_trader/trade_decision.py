"""Pandas based strategy core function."""
import typing
from typing import Dict, List

import pandas as pd

from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.pricing_model import PricingModel
from tradingstrategy.universe import Universe


class TradeDecider(typing.Protocol):
    """A callable that decides on new trades for every cycle.

    This class provides `callable type hinting <https://stackoverflow.com/questions/68472236/type-hint-for-callable-that-takes-kwargs>`_.
    Called for each cycle and returns new trades to be executed.

    Can be given inline or is a function called `decide_trades`
    in a strategy module.
    """

    # Only accept kwargs as per https://www.python.org/dev/peps/pep-3102/
    def __call__(
        *ignore,
        timestamp: pd.Timestamp,
        universe: Universe,
        state: State,
        pricing_model: PricingModel,
        cycle_debug_data: Dict) -> List[TradeExecution]:
        """

        :param ignore:
            See https://www.python.org/dev/peps/pep-3102/

        :param timestamp:
            The current cycle timestamp.
            This is the timestamp of the last candle opening time.

        :param universe:
            The current trading universe.
            All pairs, exchanges, etc.

        :param state:
            The current state of the trade execution.
            All open positions, past trades.

        :param pricing_model:
            Pricing model can tell the buy/sell price of the particular asset at a particular moment.

        :param cycle_debug_data:
            Random debug and diagnostics information
            about the current execution cycle as Python dictionary.

        :return:
            New trades to be executed on this cycle.
            These must be added as the part of the state, usually using :py:class:`tradeexecutor.strategy.pandas_trader.PositionManager`.

        """
        pass
