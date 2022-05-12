"""Strategy thinking output helpers."""
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class StrategyOutput:
    """
    This object is returned from the strategy execution cycle.
    It allows you to plot values, add debug messages, etc.
    It is not used in any trading, but can help and visualize
    trade backtesting and execution.
    """

    messages: List[str] = field(default_factory=list)

    # Extra line outputs
    plots: Dict[str, float] = field(default_factory=dict)

    def debug(self, msg: str):
        """Write a debug message."""
        self.messages.append(msg)

    def visualise(self, name, value, colour: Optional[Any]=None):
        """Add a value to the output data and diagram.

        :param color:
            Maplotlib color as a string or object
            https://matplotlib.org/stable/gallery/color/named_colors.html
            https://matplotlib.org/stable/gallery/color/named_colors.html
        """






