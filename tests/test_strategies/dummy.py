import datetime
from decimal import Decimal
from typing import Optional

import pandas as pd

from tradeexecutor.state.trade import create_trade
from tradingstrategy.chain import ChainId
from tradingstrategy.pair import PairUniverse, PandasPairUniverse
from tradingstrategy.exchange import ExchangeUniverse
from tradingstrategy.client import Client

from tradeexecutor.state.state import State
from tradeexecutor.strategy.runner import StrategyRunner


class DummyStrategyRunner(StrategyRunner):
    """A strategy exercised by unit tests."""

    #: For filter out data sets
    chain_id = ChainId.polygon

    def __init__(self):
        self.exchange_universe: Optional[ExchangeUniverse] = None
        self.pair_universe: Optional[PairUniverse] = None

    def load_datasets(self, client: Client):
        pair_table = client.fetch_pair_universe()
        self.exchange_universe = client.fetch_exchange_universe()
        self.pair_universe = PairUniverse.create_from_pyarrow_table_with_filters(pair_table, chain_id_filter=self.chain_id)
        assert len(self.pair_universe.pairs) > 1000

    def on_clock(self, clock: datetime.datetime, state: State):
        """Always buy MATIC/USDC."""
        quickswap = self.exchange_universe.get_by_chain_and_slug(self.chain_id, "quickswap")
        assert quickswap
        pair = self.pair_universe.get_pair_by_ticker_by_exchange(quickswap.exchange_id, "WMATIC", "USDC")
        assert pair
        # Buy 100 MATIC
        instruction = create_trade(clock, state, pair, Decimal(100))
        return [instruction]


strategy_runner = DummyStrategyRunner()