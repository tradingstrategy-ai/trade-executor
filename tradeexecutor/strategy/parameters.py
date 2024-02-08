"""Strategy input parameters handlding.

- Handle input parameters in single-run and grid search cases
"""
from web3.datastructures import ReadableAttributeDict


class StrategyParameters(ReadableAttributeDict):
    """Strategy parameters.

    These parameters may present

    - Individual constant parameters for the strategy, like indicators threshold levels:
      `rsi_low`, `rsi_high`

    - Parameters about the backtesting itself: `backtest_start`, `backtest_end`

    - Parameters about strategy execution: `cycle_duration`.

    The parameters are presented as attributed dict and
    are accessible using both dotted attribe access and dict access:

    .. code-block:: python

        assert parameters.rsi_low == parameters["rsi_low"]

    When dealing with grid search input, every parameter is assumed to be a list
    of potential grid search combination values. `decide_trades` function
    gets called with every single combination of the list.

    .. code-block:: python

        # TODO
    """

    @staticmethod
    def from_class(c: type, grid_search=False) -> "StrategyParameters":
        """Create parameter dict out from a class object.

        - Convert inlined class-style strategy parameter input to dictionary

        :param grid_search:
            Make grid-search style parameters.

            Every scalar input is converted to a single item list if set.
        """
        # https://stackoverflow.com/a/1939279/315168
        keys = [attr for attr in dir(c) if not callable(getattr(c, attr)) and not attr.startswith("__")]
        params = {k: getattr(c, k) for k in keys}

        if not grid_search:
            return StrategyParameters(params)

        # Convert single variable declarations to a list
        output = {}
        for k, v in params.items():
            if not type(v) in (list, tuple):
                v = [v]
            output[k] = v
        return StrategyParameters(output)