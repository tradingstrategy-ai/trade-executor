import datetime
from typing import List

from tradeexecutor.state.state import State
from tradeexecutor.trade.tradeinstruction import TradeInstruction


class StrategyRunner:

    def __init__(self):
        pass

    def preflight_check(self):
        pass

    def on_data_signal(self):
        pass

    def on_clock(self, clock: datetime.datetime, state: State) -> List[TradeInstruction]:
        pass