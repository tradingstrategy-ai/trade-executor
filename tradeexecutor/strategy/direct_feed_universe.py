"""Trading universe where data comes directly from the blockchain node.

"""
import contextlib
from typing import Optional, Set, Tuple
import logging

from tradeexecutor.strategy.execution_context import ExecutionMode
from tradeexecutor.strategy.dataset import Dataset
from tradeexecutor.strategy.universe_model import UniverseModel
from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client
from tradingstrategy.timebucket import TimeBucket


logger = logging.getLogger(__name__)


class DirectFeedUniverseModel(UniverseModel):
    """Trading universe where candle data comes directly from a blockchain node.

    Exchange and pair metadata is loaded from oracle,
    as this is not available on-chain or would be cost prohibitive to figure out.

    Because we load invididual trades from the data,
    it does not matter which candle size we use -
    everything is 1 minute in the end and we just resample higher.
    """

    def __init__(self,
                client: Client,
                chain_id: ChainId,
                exchange_slug: str,
                pairs: Set[Tuple[str, str]]):
        self.client = client
        self.timed_task_context_manager = timed_task_context_manager
        self.chain_id

    def refresh_data(self):
        pass

    def load_data(self,
                  time_frame: TimeBucket,
                  mode: ExecutionMode,
                  backtest_stop_loss_time_frame: Optional[TimeBucket]=None) -> Dataset:
        """Loads the server-side data using the client.

        :param client:
            Client instance. Note that this cannot be stable across ticks, as e.g.
            API keys can change. Client is recreated for every tick.

        :param mode:
            Live trading or backtesting

        :param backtest_stop_loss_time_frame:
            Load more granular data for backtesting stop loss

        :return:
            None if not dataset for the strategy required
        """

        assert isinstance(mode, ExecutionMode), f"Expected ExecutionMode, got {mode}"

        client = self.client

        with self.timed_task_context_manager("load_data", time_bucket=time_frame.value):

            if mode.is_fresh_data_always_needed():
                # This will force client to redownload the data
                logger.info("Execution mode %s, purging trading data caches", mode)
                client.clear_caches()
            else:
                logger.info("Execution mode %s, not live trading, Using cached data if available", mode)

            exchanges = client.fetch_exchange_universe()
            pairs = client.fetch_pair_universe().to_pandas()

            candles = client.fetch_all_candles(time_frame).to_pandas()
            liquidity = client.fetch_all_liquidity_samples(time_frame).to_pandas()

            return Dataset(
                time_bucket=time_frame,
                backtest_stop_loss_time_bucket=backtest_stop_loss_time_frame,
                exchanges=exchanges,
                pairs=pairs,
                candles=candles,
                liquidity=liquidity,
            )