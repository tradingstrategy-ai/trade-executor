"""Execution modes.

- Are we doing live trading or backtesting

- Any instrumentation like task duration tracing needed for the run
"""
import enum

from tradeexecutor.utils.timer import timed_task


class Exchange(enum.Enum):
    """Different exchanges that can be used for the strategy."""

    #: TODO documentation comments
    pancakeswap_v2 = "pancakeswap-v2"

    uniswap_v2 = "uniswap-v2"

    uniswap_v3 = "uniswap-v3"

    trader_joe = "trader-joe"

    quickswap = "quickswap"

    def is_uniswap_v2_like(self) -> bool:
        return True if self.value != self.uniswap_v3 else False

    def is_uniswap_v3_like(self) -> bool:
        return True if self.value == self.uniswap_v3 else False
