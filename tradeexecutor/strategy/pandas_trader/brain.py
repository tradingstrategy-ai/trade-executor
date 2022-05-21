"""Pandas based strategy core function."""
import typing
from typing import Dict, List

import pandas as pd

from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverseModel
from tradingstrategy.universe import Universe





# For typing.Protocol see https://stackoverflow.com/questions/68472236/type-hint-for-callable-that-takes-kwargs
class StrategyBrain(typing.Protocol):
    """A callable that decides on new trades for every cycle.

    Called for each cycle and returns new trades to be executed.
    """

    # Only accept kwargs as per https://www.python.org/dev/peps/pep-3102/
    def __call__(
        *ignore,
        timestamp: pd.Timestamp,
        universe: Universe,
        state: State,
        position_manager: PositionManager,
        cycle_debug_data: Dict) -> typing.Tuple[List[TradeExecution], StrategyOutput]:
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

        :param position_manager:
            Position manager used to generate trades to open and close positions.

        :param cycle_debug_data:
            Random debug and diagnostics information
            about the current execution cycle as Python dictionary.

        :return:
            New trades to be executed.

        """
        pass
