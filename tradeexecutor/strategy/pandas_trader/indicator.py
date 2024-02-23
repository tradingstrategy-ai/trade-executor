"""Indicator definitions."""
import os
import shutil
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, Any

import pandas as pd

from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, UniverseCacheKey


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
    """Define the indicators that are needed by a trading strategy.

    - For backtesting, indicators are precalculated

    - For live trading, these indicators are recalculated for the each decision cycle
    """

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

@dataclass
class IndicatorResult:
    """One result of an indicator calculation we can store on a disk.

    - Allows storing and reading output of a single precalculated indicator

    - Parameters is a single combination of parameters
    """

    #: The universe for which we calculated the result
    #:
    #:
    universe_key: UniverseCacheKey

    #: For which indicator this result is
    #:
    definition: IndicatorDefinition

    #: Indicator output is one time series, but in some cases can be multiple as well.
    #:
    #: For example BB indicator calculates multiple series from one close price value.
    #:
    #:
    data: pd.DataFrame



class IndicatorStorage:
    """Store calculated indicator results on disk."""

    def __init__(self, path: Path, universe_key: UniverseCacheKey):
        assert isinstance(path, Path)
        assert type(universe_key) == str
        self.path = path
        self.universe_key = universe_key

    def get_indicator_path(self, ind: IndicatorDefinition) -> Path:
        """Get the Parquet file where the indicator data is stored."""
        return self.path / Path(self.universe_key) / Path(f"{ind.get_cache_key()}.parquet")

    def is_available(self, ind: IndicatorDefinition) -> bool:
        return self.get_indicator_path(ind).exists()

    def load(self, ind: IndicatorDefinition) -> IndicatorResult:
        """Load cached indicator data from the disk."""
        assert self.is_available(ind)
        path = self.get_indicator_path(ind)
        df = pd.read_parquet(path)
        return IndicatorResult(
            self.universe_key,
            ind,
            df,
        )

    def save(self, ind: IndicatorDefinition, df: pd.DataFrame):
        """Atomic replacement of the existing data.

        - Avoid leaving partially written files
        """
        assert isinstance(ind, IndicatorDefinition)
        assert isinstance(df, pd.DataFrame)
        path = self.get_indicator_path(ind)
        dirname, basename = os.path.split(path)
        temp = tempfile.NamedTemporaryFile(mode='wb', delete=False, dir=dirname)
        df.to_parquet(temp)
        temp.close()
        # https://stackoverflow.com/a/3716361/315168
        shutil.move(temp, path)


def _serialise_parameters_for_cache_key(parameters: dict) -> str:

    for k, v in parameters.items():
        assert type(k) == str
        assert type(v) not in (list, tuple)  # Don't leak test ranges here - must be a single value

    return "".join([f"{k}={v}" for k, v in parameters.items()])


def calculate_or_load_indicators(
    universe: TradingStrategyUniverse,
    indicators: IndicatorSet,
    max_workers=8,
) -> dict[IndicatorDefinition, IndicatorResult]:
    """Precalculate all indicators.

    - Calculate indicators using multiprocessing

    - Display TQDM progress bar

    - Use cached indicators if available
    """

