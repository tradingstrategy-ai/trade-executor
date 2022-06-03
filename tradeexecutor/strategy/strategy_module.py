"""Describe strategy modules and their loading."""
import enum
from typing import Callable

from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.utils import dataclass


class StrategyType(enum.Enum):
    """Which kind of strategy types we support."""

    #: Pandas + position manager based strategy.
    #: Uses position_manager instance to tell what trades to do.
    position_manager = "position_manager"

    #: Alpha model based strategy.
    #: Return alpha weightings of assets it wants to buy.
    alpha_model = "alpha_model"


class TradeRouting(enum.Enum):
    """Trade routing shortcuts."""

    #: Two or three legged trades on PancakeSwap
    pancakeswap_basic = "quickswap_basic"

    #: Two or three legged trades on Quickswap
    quickswap_basic = "quickswap_basic"


class ReserveCurrency(enum.Enum):
    """Default supported reserve currencies."""

    #: BUSD on BNB Chain
    busd = "busd"

    #: USDC on Polygon
    usdc = "usdc"


class StrategyModuleNotValid(Exception):
    """Raised when we cannot load a strategy module."""


@dataclass
class StrategyModuleInformation:
    """Describe elements that we need to have in a strategy module."""
    trading_strategy_engine_version: str
    trading_strategy_type: StrategyType
    trading_strategy_cycle: CycleDuration
    trade_routing: TradeRouting
    reserve_currency: ReserveCurrency
    decide_trade: Callable

    #: If `execution_context.live_trading` is true then this function is called for
    #: every execution cycle. If we are backtesting, then this function is
    #: called only once at the start of backtesting and the `decide_trades`
    #: need to deal with new and deprecated trading pairs.
    create_trading_universe: Callable

    def check_valid(self):
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

        if not isinstance(self.trading_strategy_type, CycleDuration):
            raise StrategyModuleNotValid(f"trading_strategy_cycle not CycleDuration instance")

        if not isinstance(self.trade_routing, TradeRouting):
            raise StrategyModuleNotValid(f"trade_routing not TradeRouting instance")

        if not type(self.decide_trade) == str:
            raise StrategyModuleNotValid(f"decide_trade function missing/invalid")

        if not type(self.create_trading_universe) == str:
            raise StrategyModuleNotValid(f"create_trading_universe function missing/invalid")


def parse_strategy_module(mod) -> StrategyModuleInformation:
    return StrategyModuleInformation(
        mod.get("trading_strategy_engine_version"),
        mod.get("trading_strategy_type"),
        mod.get("trading_strategy_cycle"),
        mod.get("trade_routing"),
        mod.get("reserve_currency"),
        mod.get("decide_trade"),
        mod.get("create_trading_universe"),
    )
