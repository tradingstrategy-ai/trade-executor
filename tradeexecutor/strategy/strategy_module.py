"""Describe strategy modules and their loading."""
from dataclasses import dataclass
from typing import Callable, Dict, Protocol, List, Optional
import pandas

from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.reserve_currency import ReserveCurrency
from tradeexecutor.strategy.strategy_type import StrategyType
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradingstrategy.client import Client
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe


#: As set for StrategyModuleInformation.trading_strategy_engine_version
CURRENT_ENGINE_VERSION = "0.1"


class StrategyModuleNotValid(Exception):
    """Raised when we cannot load a strategy module."""


class DecideTradesProtocol(Protocol):
    """A call signature protocol for user's decide_trades() functions."""

    def __call__(self,
            timestamp: pandas.Timestamp,
            universe: Universe,
            state: State,
            price_model: PricingModel,
            cycle_debug_data: Dict) -> List[TradeExecution]:
            """

        The brain function to decide the trades on each trading strategy cycle.

        - Reads incoming execution state (positions, past trades)

        - Reads the current universe (candles)

        - Decides what to do next

        - Outputs strategy thinking for visualisation and debug messages

        :param timestamp:
            The Pandas timestamp object for this cycle. Matches
            trading_strategy_cycle division.
            Always truncated to the zero seconds and minutes, never a real-time clock.

        :param universe:
            Trading universe that was constructed earlier.

        :param state:
            The current trade execution state.
            Contains current open positions and all previously executed trades.

        :param pricing_model:
            Position manager helps to create trade execution instructions to open and close positions.

        :param cycle_debug_data:
            Python dictionary for various debug variables you can read or set, specific to this trade cycle.
            This data is discarded at the end of the trade cycle.

        :return:
            List of trade instructions in the form of :py:class:`TradeExecution` instances.
            The trades can be generated using `position_manager` but strategy could also handcraft its trades.
        """


class CreateTradingUniverseProtocol(Protocol):
    """A call signature protocol for user's create_trading_universe() functions."""

    def __call__(self,
            timestamp: pandas.Timestamp,
            client: Optional[Client],
            execution_context: ExecutionContext,
            candle_time_frame_override: Optional[TimeBucket]=None) -> TradingStrategyUniverse:
        """Creates the trading universe where the strategy trades.

        If `execution_context.live_trading` is true then this function is called for
        every execution cycle. If we are backtesting, then this function is
        called only once at the start of backtesting and the `decide_trades`
        need to deal with new and deprecated trading pairs.

        As we are only trading a single pair, load data for the single pair only.

        :param ts:
            The timestamp of the trading cycle. For live trading,
            `create_trading_universe` is called on every cycle.
            For backtesting, it is only called at the start

        :param client:
            Trading Strategy Python client instance.

        :param execution_context:
            Information how the strategy is executed. E.g.
            if we are live trading or not.

        :param candle_timeframe_override:
            Allow the backtest framework override what candle size is used to backtest the strategy
            without editing the strategy Python source code file.

        :return:
            This function must return :py:class:`TradingStrategyUniverse` instance
            filled with the data for exchanges, pairs and candles needed to decide trades.
            The trading universe also contains information about the reserve asset,
            usually stablecoin, we use for the strategy.
            """


def pregenerated_create_trading_universe(universe: TradingStrategyUniverse) -> CreateTradingUniverseProtocol:
    """Wrap existing trading universe, so it can be passed around for universe generators."""

    def _inner(timestamp: pandas.Timestamp,
            client: Optional[Client],
            execution_context: ExecutionContext,
            candle_time_frame_override: Optional[TimeBucket]=None):
        return universe

    return _inner


@dataclass
class StrategyModuleInformation:
    """Describe elements that we need to have in a strategy module."""
    trading_strategy_engine_version: str
    trading_strategy_type: StrategyType
    trading_strategy_cycle: CycleDuration
    trade_routing: TradeRouting
    reserve_currency: ReserveCurrency

    decide_trades: DecideTradesProtocol

    #: If `execution_context.live_trading` is true then this function is called for
    #: every execution cycle. If we are backtesting, then this function is
    #: called only once at the start of backtesting and the `decide_trades`
    #: need to deal with new and deprecated trading pairs.
    create_trading_universe: CreateTradingUniverseProtocol

    def validate(self):
        """

        :raise StrategyModuleNotValid:
            If we could not load/parse strategy module for some reason
        """

        if not self.trading_strategy_engine_version:
            raise StrategyModuleNotValid(f"trading_strategy_engine_version missing in the module")

        if not type(self.trading_strategy_engine_version) == str:
            raise StrategyModuleNotValid(f"trading_strategy_engine_version is not string")

        if self.trading_strategy_engine_version != "0.1":
            raise StrategyModuleNotValid(f"Only version 0.1 supported for now, got {self.trading_strategy_engine_version}")

        if not self.trading_strategy_type:
            raise StrategyModuleNotValid(f"trading_strategy_type missing in the module")

        if not isinstance(self.trading_strategy_type, StrategyType):
            raise StrategyModuleNotValid(f"trading_strategy_type not StrategyType instance")

        if not isinstance(self.trading_strategy_cycle, CycleDuration):
            raise StrategyModuleNotValid(f"trading_strategy_cycle not CycleDuration instance, got {self.trading_strategy_cycle}")

        if self.trade_routing is None:
            raise StrategyModuleNotValid(f"trade_routing missing on the strategy")

        if not isinstance(self.trade_routing, TradeRouting):
            raise StrategyModuleNotValid(f"trade_routing not TradeRouting instance, got {self.trade_routing}")

        if not isinstance(self.decide_trades, Callable):
            raise StrategyModuleNotValid(f"decide_trades function missing/invalid")

        if not isinstance(self.create_trading_universe, Callable):
            raise StrategyModuleNotValid(f"create_trading_universe function missing/invalid")


def parse_strategy_module(mod) -> StrategyModuleInformation:
    """Parse a loaded .py module that describes a trading strategy.

    :param mod:
        Python module
    """
    return StrategyModuleInformation(
        mod.get("trading_strategy_engine_version"),
        mod.get("trading_strategy_type"),
        mod.get("trading_strategy_cycle"),
        mod.get("trade_routing"),
        mod.get("reserve_currency"),
        mod.get("decide_trades"),
        mod.get("create_trading_universe"),
    )
