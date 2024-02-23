"""Indicator definitions."""
from collections.abc import Iterable
from pathlib import Path
from typing import Callable, Protocol, Any

import pandas as pd

from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.utils import dataclass


class Indicators:
    pass


class IndicatorSource:
    """The data on which the indicator will be calculated."""

    close_price = "close_price"

    open_price = "open_price"



@dataclass
class IndicatorDefinition:
    """A definition for a single indicator."""

    name: str

    func: Callable

    #: Parameters for building this indicator.
    #:
    #: - Each key is a function argument name for :py:attr:`func`.
    #: - Each value is a single value
    #:
    parameters: dict

    #: On what trading universe data this indicator is calculated
    #:
    source: IndicatorSource

    def __post_init__(self):
        pass

    def get_cache_key(self) -> str:
        parameters = ", ".join([f"{k}={v}" for k, v in self.parameters.items()])
        return f"{self.name}({self.parameters})"


class IndicatorSet:

    def __init__(self):
        self.indicators: dict[str, IndicatorDefinition] = {}

    def add(
        self,
        name: str,
        func: Callable,
        parameters: dict,
        source: IndicatorSource=IndicatorSource.close_price,
    ):
        assert type(name) == str
        assert callable(func), f"{func} is not callable"
        assert type(parameters) == dict, f"parameters must be dictionary, we got {parameters.__class__}"
        assert isinstance(source, IndicatorSource)
        self.indicators[name] = IndicatorDefinition(name, func, parameters, source)



class CreateIndicators(Protocol):
    """Call signature for create_indicators function"""

    def __call__(self, parameters: StrategyParameters, indicators: IndicatorSet):
        """Build technical indicators for the strategy.

        :param parameters:
            Passed from the backtest / live strategy parametrs.

            If doing a grid search, each paramter is simplified.

        :param indicators:
            Indicator builder helper class.

            Call :py:meth:`IndicatorBuilder.create` to add new indicators to the strategy.
        """

class IndicatorResult:
    """One result of an indicator calculation we can store on a disk.

    - Allows storing and reading output of a single precalculated indicator

    - Parameters is a single combination of parameters
    """

    universe_key: str

    definition: IndicatorDefinition

    parameters: dict[str, Any]

    data: pd.Series



class IndicatorStorage:

    def __init__(self, path: Path, universe_key: str):
        self.path = path
        self.universe_key = universe_ky

    def get_cache_key(self) -> str:
        return f"{self.universe_key}-{self.definition.name}-{_serialise_parameters_for_cache_key(self.parameters)}"

    def get_indicator_path(self, ind: IndicatorDefinition) -> Path:
        return self.path / Path(self.universe_key) / Path(f"{}")


def _serialise_parameters_for_cache_key(parameters: dict) -> str:

    for k, v in parameters.items():
        assert type(k) == str
        assert type(v) not in (list, tuple)  # Don't leak test ranges here - must be a single value

    return "".join([f"{k}={v}" for k, v in parameters.items()])