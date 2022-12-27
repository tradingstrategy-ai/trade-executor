"""Execution modes.

- Are we doing live trading or backtesting

- Any instrumentation like task duration tracing needed for the run
"""
import enum

from tradingstrategy.chain import ChainId
from tradeexecutor.utils.timer import timed_task

class ExchangeSlug(enum.Enum):
    """Different exchanges that can be used for the strategy.
    
    TODO add support in codebase for using this enum
    """

    pancakeswap_v2 = "pancakeswap-v2"

    uniswap_v2 = "uniswap-v2"

    uniswap_v3 = "uniswap-v3"

    trader_joe = "trader-joe"

    quickswap = "quickswap"

    def get_uniswap_v3_like(self) -> set:
        """Returns a set of all uniswap v3 like exchanges.
        
        This method needs to be kept updated with the latest uniswap v3 like exchanges
        """
        return {self.uniswap_v3}

    def is_uniswap_v2_like(self) -> bool:
        """Returns a true if value is uniswap v2 like, false if uniswap v3 like"""
        return self not in self.get_uniswap_v3_like()

    def is_uniswap_v3_like(self) -> bool:
        """Returns a true if value is uniswap v3 like, false if uniswap v2 like"""
        return self in self.get_uniswap_v3_like()
    
    def get_chain_id(self) -> ChainId:
        """Gets the chain id for the relevant exchange.
        
        TODO use to validate that user has inputted correct chainid/exchange.
         
        Note that we can't automatically get chain id without user input because some exchanges
        such as uniswap operate on multiple blockchains.
        """
        if self in [self.uniswap_v2, self.uniswap_v3]:
            return {ChainId.ethereum, ChainId.polygon}
        elif self in [self.pancakeswap_v2]:
            return {ChainId.bsc}
        elif self in [self.quickswap]:
            return {ChainId.polygon}
        elif self in [self.trader_joe]:
            return {ChainId.avalanche}
        else:
            raise TypeError()


