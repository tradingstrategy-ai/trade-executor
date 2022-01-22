import datetime
from typing import List

from tradingstrategy.client import Client

from tradeexecutor.state.state import State, TradeExecution


class StrategyRunner:

    def __init__(self):
        pass

    def load_datasets(self, client: Client):
        pass

    def preflight_check(self):
        pass

    def on_data_signal(self):
        pass

    def on_clock(self, clock: datetime.datetime, state: State) -> List[TradeExecution]:
        pass