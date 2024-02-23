"""Indicator definitions."""
from typing import Callable, Protocol

from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


class Indicators:
    pass



class IndicatorBuilder:

    def __init__(self, strategy_universe: TradingStrategyUniverse):
        self.strategy_universe = strategy_universe
        self.indicators = {}

    def create(
        self,
        func: Callable,
        parameters: dict,
    ):
        pass


class CreateIndicators(Protocol):
    """Call signature for create_indicators function"""

    def __call__(self, parameters: StrategyParameters, indicators: IndicatorBuilder):
        """Build technical indicators for the strategy.

        :param parameters:
            Passed from the backtest / live strategy parametrs.

            If doing a grid search, each paramter is simplified.

        :param indicators:
            Indicator builder helper class.

            Call :py:meth:`IndicatorBuilder.create` to add new indicators to the strategy.
        """

