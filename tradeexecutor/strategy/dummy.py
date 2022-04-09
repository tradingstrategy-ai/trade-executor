"""Trader where all assets are maintained on a single hot wallet."""
from typing import List

from tradeexecutor.state.trade import TradeExecution


class DummyExecutionModel:
    """Trade executor that does not connect to anything."""

    def __init__(self, private_key: str):
        self.private_key = private_key

    def execute_trades(self, trade_instructions: List[TradeExecution]):
        pass