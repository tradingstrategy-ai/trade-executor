"""Decorator syntax for indicators.

- Allow shorter definition of indicators using Python decorators, over the manual methods provided in :py:mod:`tradeexecutor.strategy.pandas_trade.indicator` module.

- See :py:class:`IndicatorRegistry` for usage.


"""
import inspect
import itertools
import logging
from dataclasses import dataclass
import datetime
from functools import wraps
from typing import Callable, Iterable, Sequence

import pandas as pd

from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet, IndicatorSource, CreateIndicatorsProtocolV1, CreateIndicatorsProtocolV2
from tradeexecutor.strategy.parameters import StrategyParameters, RollingParameter
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


logger = logging.getLogger(__name__)


class NotYetDefined(Exception):
    """There is a dependency resolution order issue with indicators."""
    pass


class ParameterMissing(Exception):
    """Indicator asks for a parameter that is missing."""
    pass


#: List of function arguments for different indicator functions that are built-in,
#: and do not come from StrategyParameters definitions.
#:
#: Each indicator source may have different arguments available.
#:
DEFAULT_INDICATOR_ARGUMENTS = {
    "close",
    "open",
    "high",
    "low",
    "volume",
    "strategy_universe",
    "dependency_resolver",
    "pair",
    "execution_context",
    "timestamp"
}


@dataclass(slots=True, frozen=True)
class IndicatorDecoration:
    """Internal struct capturing information we need about indicators."""

    #: Indicator name
    name: str

    #: Indicator function
    func: Callable

    #: Source data type
    source: IndicatorSource | None

    #: Linerised order for the parallel execution
    order: int

    #: Name of function arguments sans :py:attr:`DEFAULT_INDICATOR_ARGUMENTS`
    args: set[str]

    #: Dependencies needed to run this indicator
    dependencies: list[Callable]


def extract_args(func) -> set[str]:
    """Get the names of Python function arguments."""
    signature = inspect.signature(func)
    return set(signature.parameters.keys()) - DEFAULT_INDICATOR_ARGUMENTS


def detect_source(func: Callable, source: IndicatorSource | None) -> IndicatorSource:
    """Based on indicator function arguments, detect what kind of data source it needs."""
    if source is not None:
        return source

    signature = inspect.signature(func)
    args = set(signature.parameters.keys())
    if "strategy_universe" in args:
        return IndicatorSource.strategy_universe
    elif ("open" in args) or ("high" in args) or ("low" in args) or ("volume" in args):
        return IndicatorSource.ohlcv
    elif ("close" in args):
        return IndicatorSource.close_price
    else:
        raise RuntimeError(f"Cannot detect IndicatorSource for function {func} - please manually specify using @indicators.define(source) argument")



class IndicatorRegistry:
    """Decorator-based helper class for defining strategy indicators.

    Use ``@indicatores.define()`` decorator to create strategy technical indicators in your notebooks.

    - Less typing than with manuaal ``indicator.add()`` syntax

    - `Parameters` class members are automatically mapped to the function argument with the same name

    - Indicators list other indicator functions they depend on using syntax ``@indicators.define(dependencies=(slow_sma, fast_sma))``,
      this is automatically resolved to the current dependency resolution order when the indicator calulation order is determined.

    - Indicator data can be looked up by indicator function or its string name

    Example:

    .. code-block:: python

        import pandas_ta
        from tradeexecutor.strategy.pandas_trader.indicator_decorator import IndicatorRegistry

        class Parameters:
            rsi_length = 20

        indicators = IndicatorRegistry()

        @indicators.define()
        def rsi(close, rsi_length):
            return pandas_ta.rsi(close, rsi_length)

        # Calculate indicators - will spawn multiple worker processed,
        # or load cached results from the disk
        parameters = StrategyParameters.from_class(Parameters)
        indicators = calculate_and_load_indicators_inline(
            strategy_universe=strategy_universe,
            parameters=parameters,
            create_indicators=indicators.create_indicators,
        )

    With indicators that depend on each other:

    .. code-block:: python

        import pandas_ta
        from tradeexecutor.strategy.pandas_trader.indicator_decorator import IndicatorRegistry


        class Parameters:
            fast_ma_length = 7
            slow_ma_length = 21

        indicators = IndicatorRegistry()

        @indicators.define()
        def slow_sma(close, slow_ma_length):
            return pandas_ta.sma(close, slow_ma_length)

        @indicators.define()
        def fast_sma(close, fast_ma_length):
            return pandas_ta.sma(close, fast_ma_length)

        @indicators.define(dependencies=(slow_sma, fast_sma))
        def ma_crossover(
            close: pd.Series,
            pair: TradingPairIdentifier,
            dependency_resolver: IndicatorDependencyResolver,
        ) -> pd.Series:
            # You can use name or function to look up previously calcualted indicators data
            slow_sma: pd.Series = dependency_resolver.get_indicator_data(slow_sma)
            fast_sma: pd.Series = dependency_resolver.get_indicator_data("fast_sma")
            return fast_sma > slow_sma

    """

    def __init__(self):
        self.registry: dict[str, IndicatorDecoration] = {}

    def define(
        self,
        source: IndicatorSource=None,
        order: None | int=None,
        dependencies: Sequence[Callable]=None,
    ):
        """Function decorator to define indicator functions in your notebook.

        For the usage see :py:class:`IndicatorRegistry`.

        Short example:

        .. code-block:: python

            class Parameters:
                slow_ma_length = 21

            indicators = IndicatorRegistry()

            @indicators.define()
            def slow_sma(close, slow_ma_length):
                return pandas_ta.sma(close, slow_ma_length)

        :param source:
            What kind of trading data this indicator needs as input.

            - Per-pair
            - World
            - Raw data
            - Data from earlier calculated indicators

        :param order:
            Manually define indicator order.

            Not usually needed.

        :param dependencies:
            List of functions which result value this indicator consumes.

            Needed with :py:attr:`~tradeexecutor.strategy.pandas_trader.indicator.IndicatorSource.dependencies_only_per_pair`
            and :py:attr:`~tradeexecutor.strategy.pandas_trader.indicator.IndicatorSource.dependencies_only_universe`.
        """
        def decorator(func):
            nonlocal dependencies
            name = func.__name__
            dependencies = list(dependencies) if dependencies else []
            resolved_order = order if order else self.detect_order(func, dependencies)

            self.registry[name] = IndicatorDecoration(
                name=name,
                func=func,
                source=detect_source(func, source),
                order=resolved_order,
                args=extract_args(func),
                dependencies=dependencies,
            )
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Call the original function with the original arguments
                return func(*args, **kwargs)
            return wrapper

        return decorator

    def detect_order(self, func: Callable, dependencies: list[Callable]) -> int:
        """Automatically resolve the order of indicators.

        :return:
            Linear order number for running the indicator.

            Can be used to parallerise the process.
        """

        if dependencies is None:
            return 1

        assert isinstance(dependencies, Iterable), f"Bad dependencies, does not look like a list: {dependencies} in {func}"

        for dependency in dependencies:
            assert callable(dependency), f"Dependency must be specified as a function, got {dependency} in {func}"
            if dependency.__name__ not in self.registry:
                already_defined = list(self.registry.keys())
                raise NotYetDefined(f"Function {func} asks for dependency {dependency} which is not yet defined.\nWe have defined {already_defined}")

        dep_names = [func.__name__ for func in dependencies]
        max_order = max((dec.order for dec in self.registry.values() if dec.name in dep_names), default=0)
        return max_order + 1

    def create_indicators(
        self,
        timestamp: datetime.datetime,
        parameters: StrategyParameters,
        strategy_universe: TradingStrategyUniverse,
        execution_context: ExecutionContext,
    ) -> IndicatorSet:
        """Hook function to be passed to :py:func:`calculate_and_load_indicators_inline`.

        Example:

        .. code-block:: python

            indicators = IndicatorRegistry()

            @indicators.define()
            def rsi(close, rsi_length):
                return pandas_ta.rsi(close, rsi_length)

            # Calculate indicators - will spawn multiple worker processed,
            # or load cached results from the disk
            parameters = StrategyParameters.from_class(Parameters)
            indicators = calculate_and_load_indicators_inline(
                strategy_universe=strategy_universe,
                parameters=parameters,
                create_indicators=indicators.create_indicators,
            )
        """
        indicators = IndicatorSet()
        for name, definition in self.registry.items():

            applied_parameters = {}
            rollable = False
            for parameter in definition.args:
                if parameter not in parameters:
                    raise ParameterMissing(f"Function {name} requires parameter {parameter}, but this is not defined in strategy parameters.\nWe have: {list(parameters.keys())}")
                value = parameters[parameter]
                rollable |= RollingParameter.is_rolling(value)
                if rollable:
                    applied_parameters[parameter] = list(value.values)
                else:
                    applied_parameters[parameter] = value

            if rollable:
                logger.info("Rollable parameters detected, unrolling")
                permutations = _flatten_dict_permutations(applied_parameters)
                for p in permutations:
                    indicators.add(
                        name=definition.name,
                        func=definition.func,
                        source=definition.source,
                        order=definition.order,
                        parameters=p,
                        variations=True,
                    )
            else:
                indicators.add(
                    name=definition.name,
                    func=definition.func,
                    source=definition.source,
                    order=definition.order,
                    parameters=applied_parameters,
                )

        return indicators

    def get_diagnostics(self) -> pd.DataFrame:
        """Return a table that explains registered indicators.

        :return:
            Human-readable table
        """

        data = []
        for entry in self.registry.values():
            data.append({
                "Order": entry.order,
                "Name": entry.name,
                "Source": entry.source.name,
                "Parameters": ", ".join(entry.args),
                "Dependencies": ", ".join(d.__name__ for d in entry.dependencies),
            })

        df = pd.DataFrame(data)
        df = df.sort_values(by=["Order", "Name"])
        df = df.set_index("Name")
        return df


def _flatten_dict_permutations(input_dict):
    # Separate scalar and list values
    scalar_values = {k: v for k, v in input_dict.items() if not isinstance(v, list)}
    list_values = {k: v for k, v in input_dict.items() if isinstance(v, list)}

    # Generate all combinations of list values
    keys = list_values.keys()
    values = list_values.values()
    combinations = itertools.product(*values)

    # Create dictionaries for each combination
    result = []
    for combination in combinations:
        new_dict = scalar_values.copy()
        new_dict.update(zip(keys, combination))
        result.append(new_dict)

    return result