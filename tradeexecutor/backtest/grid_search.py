"""Perform a grid search ove strategy parameters to find optimal parameters."""
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Dict, List, Tuple, Any

import pandas as pd

from tradeexecutor.state.state import State
from tradeexecutor.strategy.strategy_module import DecideTradesProtocol


@dataclass
class GridParameter:
    name: str
    value: Any

    def __post_init__(self):
        pass

    @staticmethod
    def to_path(value: Any) -> str:
        """"""
        if type(value) in (float, int, str):
            return str(value)
        elif isinstance(value, pd.Timedelta):
            return value.seconds()  
        else:
            raise NotImplementedError(f"We do not support filename conversion for value {type(value)}={value}")


class GridCombination:
    parameters: List[GridParameter]


class DecideTradesGridSearchFactory(Protocol):
    """Define how to create different strategy bodies."""


    def __call__(self, **kwargs) -> DecideTradesProtocol:
        """Create a new decide_trades() strategy body based over the serach parameters.

        :param args:
        :param kwargs:
        :return:
        """


def prepare_search_matrix(parameters: Dict[str, List[Any]) -> List[GridCombination]:
    """Get iterable search matrix of all parameter combinations."""




def search_parameters(
        decide_trades_factory: DecideTradesGridSearchFactory,
        parameters: Dict[str, List],
        result_path: Path,
) -> Dict[Tuple, State]:
    """Search different strategy parameters over a grid.

    :param result_path:
        A folder where resulting state files will be stored.

    """

    assert result_path.exists() and result_path.is_dir(), f"Not a dir: {result_path}"

    combination_matrix = []



