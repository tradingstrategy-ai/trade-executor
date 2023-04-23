"""Perform a grid search ove strategy parameters to find optimal parameters."""
import itertools
import os
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

    def to_path(self) -> str:
        """"""
        value = self.value
        if type(value) in (float, int, str):
            return f"{self.name}={self.value}"
        else:
            raise NotImplementedError(f"We do not support filename conversion for value {type(value)}={value}")


@dataclass()
class GridCombination:
    """One combination line in grid search."""

    #: Alphabetically sorted list of parameters
    parameters: List[GridParameter]

    def __post_init__(self):
        """Always sort parameters alphabetically"""
        self.parameters = sorted(self.parameters, key=lambda p: p.name)

    def get_state_path(self) -> Path:
        """Get the path where the resulting state file is stored."""
        return Path(os.path.join(*[p.to_path() for p in self.parameters]))


class DecideTradesGridSearchFactory(Protocol):
    """Define how to create different strategy bodies."""


    def __call__(self, **kwargs) -> DecideTradesProtocol:
        """Create a new decide_trades() strategy body based over the serach parameters.

        :param args:
        :param kwargs:
        :return:
        """


def prepare_grid_combinations(parameters: Dict[str, List[Any]]) -> List[GridCombination]:
    """Get iterable search matrix of all parameter combinations."""

    args_lists: List[list] = []
    for name, values in parameters.items():
        args = [GridParameter(name, v) for v in values]
        args_lists.append(args)

    #
    combinations = itertools.product(*args_lists)

    return [GridCombination(c) for c in combinations]



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



