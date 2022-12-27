"""Execution modes.

- Are we doing live trading or backtesting

- Any instrumentation like task duration tracing needed for the run
"""
import enum

from tradeexecutor.utils.timer import timed_task


class ExchangeSlug(enum.Enum):
    """Different exchanges that can be used for the strategy."""

    pancakeswap_v2 = "pancakeswap-v2"

    uniswap_v2 = "uniswap-v2"

    uniswap_v3 = "uniswap-v3"

    trader_joe = "trader-joe"

    quickswap = "quickswap"

    def get_uniswap_v3_like(self) -> set:
        """Returns a set of all uniswap v3 like exchanges.
        
        This method needs to be kept updated with the latest uniswap v3 like exchanges
        """
        return {self.uniswap_v3.value}

    def is_uniswap_v2_like(self) -> bool:
        """Returns a true if value is uniswap v2 like, false if uniswap v3 like"""
        return self.value not in self.get_uniswap_v3_like()

    def is_uniswap_v3_like(self) -> bool:
        """Returns a true if value is uniswap v3 like, false if uniswap v2 like"""
        return self.value in self.get_uniswap_v3_like()
