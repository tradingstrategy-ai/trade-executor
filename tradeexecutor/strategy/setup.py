"""Strategy setup description.

Describe input variables for a strategy.
"""
import datetime
from dataclasses import field, dataclass
from enum import Enum
from typing import TypedDict, List, Union, Type

import pandas as pd

from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.reserve_currency import ReserveCurrency
from tradeexecutor.strategy.strategy_module import CreateTradingUniverseProtocol, DecideTradesProtocol
from tradeexecutor.strategy.strategy_type import StrategyType
from tradingstrategy.pair import HumanReadableTradingPairDescription
from tradingstrategy.timebucket import TimeBucket


class UnexpectedStrategyInput(Exception):
    """Raised when there is an unexpected parameter in the strategy setup."""


class IndicatorDescription:

    def __init__(self, name: str, **kwargs):
        pass


@dataclass
class StrategySetup:

    #: Strategy module version.
    #:
    #: Describes how other input parameters are parsed.
    #:
    #: The current latest version is 0.3.
    #:
    version: str = field(default="0.3")

    #: What kind of trading strategy this is
    #:
    #: The default type is "managed positions" where
    #: buys and sells happen based on signals and technical indicators.
    #:
    type: StrategyType | str = StrategyType.managed_positions

    #: A list of trading pairs this strategy is going to trade.
    #:
    #: If you want to create a dynamic universe see :py:attr:`create_trading_universe`
    #:
    trading_pairs: List[HumanReadableTradingPairDescription] | None = field(default=None)

    #: A List of indicators we need in this strategy.
    indicators: List[IndicatorDescription] = field(default_factory=list)

    #: How often decide_trades() is called.
    #:
    #: This can be more frequent or less frequent than :py:attr:`candle_time_frame`
    #:
    decision_time_frame: str | CycleDuration = field(default=None)

    #: Which is the main candle time bucket we use when loading data
    candle_time_frame: str | TimeBucket = field(default=None)

    #: Which is the main liquidity time bucket we use when loading data.
    #:
    #: Leave to `None` if the strategy does not use liquidity data.
    #:
    liquidity_time_frame: str | TimeBucket = field(default=None)

    #: The function to run the strategy logic.
    #:
    decide_trades: DecideTradesProtocol = field(default=None)

    #: A callback for creating the trading univese.
    #:
    #: Optional: If a staitc list of :py:attr:`trading_pairs` is given, use it.
    #:
    create_trading_universe: CreateTradingUniverseProtocol | None = None

    #: How trades are routed for this strategy.
    #:
    #: For DEX trading, the execution engine must do the routing itself.
    #:
    trade_routing: TradeRouting | None = None

    #: What reserve currency we use for this strat3egy
    #:
    #: For DEX trading, the execution engine must do the routing itself.
    #:
    reserve_currency: ReserveCurrency | None = None

    def parse_enum(self, enum_type: Type[Enum], name: str) -> Enum:
        value = getattr(self, name, None)
        try:
            return enum_type(value)
        except Exception as e:
            options = [e.value for e in enum_type]
            raise UnexpectedStrategyInput(f"Did not understand the input parameter {name}={value}, expected options {options}:\n{e}")

    @staticmethod
    def parse_and_validate(self) -> "StrategySetup":

        result = StrategySetup()

        if not self.version:
            raise UnexpectedStrategyInput(f"version missing in the module")

        if self.version != "0.3":
            raise UnexpectedStrategyInput(f"Only version 0.1 supported for now, got {self.trading_strategy_engine_version}")

        result.version = version

        if not self.type:
            raise UnexpectedStrategyInput(f"trading_strategy_type missing in the module")

        result.type = self.parse_enum(StrategyType, type)

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

        if self.trade_routing is None:
            raise StrategyModuleNotValid(f"trade_routing missing on the strategy")

        if self.chain_id:
            assert isinstance(self.chain_id, ChainId), f"Strategy module chain_in varaible expected ChainId instance, got {self.chain_id}"


class BacktestSetup(StrategySetup):
    """Backtest specific add-ons for the strategy set-up."""

    #: When backtest starts
    #:
    star_at: datetime.datetime | pd.Timestamp | str = None

    #: When backtest ends
    #:
    end_at: datetime.datetime | pd.Timestamp | str = None

    #: Load trigger signal.
    #:
    #: A more granural time frame or tick data is used to backtest stop loss/take profit/such conditions.
    #: On a live trade execution environment this is the live tick data.
    #:
    trigger_time_frame: TimeBucket | str = None


