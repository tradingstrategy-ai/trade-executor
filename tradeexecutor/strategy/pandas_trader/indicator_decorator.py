"""Decorator syntax for indicators.

- Allow shorter definition of indicators using Python deocators


"""
import inspect
from dataclasses import dataclass
import datetime
from functools import wraps
from typing import Callable, Iterable

from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet, IndicatorSource, CreateIndicatorsProtocolV1, CreateIndicatorsProtocolV2
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


class NotYetDefined(Exception):
    pass


class ParameterMissing(Exception):
    pass


DEFAULT_INDICATOR_ARGUMENTS = {
    "close",
    "open",
    "high",
    "low",
    "strategy_universe",
    "dependency_resolver",
    "pair",
}


@dataclass(slots=True)
class IndicatorDecoration:
    name: str
    func: Callable
    source: IndicatorSource | None
    order: int
    args: set[str]


def extract_args(func) -> set[str]:
    signature = inspect.signature(func)
    return set(signature.parameters.keys()) - DEFAULT_INDICATOR_ARGUMENTS


def detect_source(func: Callable, source: IndicatorSource | None) -> IndicatorSource:

    if source is not None:
        return source

    signature = inspect.signature(func)
    args = set(signature.parameters.keys())
    if "strategy_universe" in args:
        return IndicatorSource.strategy_universe
    elif ("open" in args) or ("high" in args) or ("low" in args):
        return IndicatorSource.ohlcv
    elif ("close" in args):
        return IndicatorSource.close_price
    else:
        raise RuntimeError(f"Cannot detect IndicatorSource for function {func} - please manually specify using @indicators.define(source) argument")



class IndicatorRegistry:
    """Decorator-based helper class for defining strategy indicators."""

    def __init__(self):
        self.registry: dict[str, IndicatorDecoration] = {}

    def define(
        self,
        source: IndicatorSource=None,
        order=1,
        dependencies: Iterable[Callable]=None,
    ):
        def decorator(func):
            name = func.__name__
            self.registry[name] = IndicatorDecoration(
                name=name,
                func=func,
                source=detect_source(func, source),
                order=self.detect_order(func, dependencies),
                args=extract_args(func),
            )

            @wraps(func)
            def wrapper(*args, **kwargs):
                # Call the original function with the original arguments
                return func(*args, **kwargs)
            return wrapper

        return decorator

    def detect_order(self, func: Callable, dependencies: set[Callable] | None) -> int:

        if dependencies is None:
            return 1

        assert isinstance(dependencies, Iterable), f"Bad dependencies, does not look like a list: {dependencies} in {func}"

        for dependency in dependencies:
            assert callable(dependency), f"Dependency must be specified as a function, got {dependency} in {func}"
            if dependency.__name__ not in self.registry:
                already_defined = list(self.registry.keys())
                raise NotYetDefined(f"Function {func} asks for dependency {dependency} which is not yet defined.\nWe have defined {already_defined}")

        dep_names = [func.__name__ for func in dependencies]
        max_order = max((dec.order for dec in self.registry.values() if dec.name in dep_names), default=1)
        return max_order + 1

    def create_indicators(
        self,
        timestamp: datetime.datetime,
        parameters: StrategyParameters,
        strategy_universe: TradingStrategyUniverse,
        execution_context: ExecutionContext,
    ) -> IndicatorSet:
        indicators = IndicatorSet()
        for name, definition in self.registry.items():

            applied_parameters = {}
            for parameter in definition.args:
                if parameter not in parameters:
                    raise ParameterMissing(f"Function {name} requires parameter {parameter}, but this is not defined in strategy parameters.\nWe have: {list(parameters.keys())}")
                applied_parameters[parameter] = parameters[parameter]

            indicators.add(
                name=definition.name,
                func=definition.func,
                source=definition.source,
                order=definition.order,
                parameters=applied_parameters,
            )

        return indicators
