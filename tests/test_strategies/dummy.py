import datetime
from decimal import Decimal

from tradeexecutor.state.trade import create_trade
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.chain import ChainId


from tradeexecutor.state.state import State
from tradeexecutor.strategy.runner import StrategyRunner, Dataset
from tradingstrategy.client import Client
from tradingstrategy.liquidity import GroupedLiquidityUniverse
from tradingstrategy.pair import PandasPairUniverse
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe
from tradingstrategy.utils.groupeduniverse import filter_for_single_pair


class DummyMATICUSDCStrategyRunner(StrategyRunner):
    """A strategy exercised by unit tests."""

    def get_strategy_time_frame(self):
        # Monthly candles, to minimize any download delays
        return TimeBucket.d30

    def construct_universe(self, dataset: Dataset) -> Universe:
        """Create a trading universe that only contains WMATIC-USDC pair."""
        chain_id = ChainId.polygon
        quickswap = dataset.exchanges.get_by_chain_and_slug(chain_id, "quickswap")
        pairs = PandasPairUniverse.create_single_pair_universe(dataset.pairs, quickswap, "WMATIC", "USDC")
        pair = pairs.get_single()
        candles = GroupedCandleUniverse(filter_for_single_pair(dataset.candles, pair))
        liquidity = GroupedLiquidityUniverse(filter_for_single_pair(dataset.candles, pair))
        return Universe(
            time_frame=dataset.time_frame,
            chains=[chain_id],
            exchanges=[quickswap],
            pairs=pairs,
            candles=candles,
            liquidity=liquidity,
        )

    def preflight_check(self, client: Client, universe: Universe, now_: datetime.datetime):
        super().preflight_check(client, universe, now_)
        # We are trading only one trading pair
        assert universe.pairs.get_count() == 1

    def on_clock(self, clock: datetime.datetime, universe: Universe, state: State):
        """Always buy MATIC/USDC."""
        # Buy 100 MATIC
        quickswap = universe.exchanges[0]
        assert quickswap.exchange_slug == "quickswap"
        pair = universe.pairs.get_single()
        assert pair.base_token_symbol == "WMATIC"
        assert pair.quote_token_symbol == "USDC"
        # Buy 100 MATIC
        instruction = create_trade(clock, state, pair, Decimal(100))
        return [instruction]


def strategy_executor_factory(**kwargs):
    strategy_runner = DummyMATICUSDCStrategyRunner(**kwargs)
    return strategy_runner
