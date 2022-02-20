import datetime
from contextlib import AbstractContextManager
from typing import List, Type, Optional
import logging

import pandas as pd

from qstrader.alpha_model.alpha_model import AlphaModel
from qstrader.asset.universe.static import StaticUniverse
from qstrader.data.backtest_data_handler import BacktestDataHandler
from qstrader.portcon.optimiser.fixed_weight import FixedWeightPortfolioOptimiser
from tradeexecutor.strategy.qstrader.livealphamodel import LiveAlphaModel
from tradeexecutor.strategy.qstrader.ordersizer import CashBufferedOrderSizer

from tradingstrategy.client import Client
from tradingstrategy.frameworks.qstrader import TradingStrategyDataSource
from tradingstrategy.universe import Universe

from tradeexecutor.state.state import State, TradeExecution
from tradeexecutor.strategy.qstrader.pcm import PortfolioConstructionModel
from tradeexecutor.strategy.runner import StrategyRunner, PreflightCheckFailed


logger = logging.getLogger(__name__)


class QSTraderRunner(StrategyRunner):
    """A live trading executor for QSTrade based algorithm."""

    def __init__(self, *args, alpha_model: LiveAlphaModel, max_data_age: Optional[datetime.timedelta] = None, cash_buffer=0.05, **kwargs):
        """
        :param alpha_model:
        :param timed_task_context_manager:
        :param max_data_age: Allow to unit test on old datasets
        """
        super().__init__(*args, **kwargs)
        assert alpha_model
        self.alpha_model = alpha_model
        self.max_data_age = max_data_age
        # TODO: Make starter configuration
        self.cash_buffer = cash_buffer

    def on_data_signal(self):
        pass

    def on_clock(self, clock: datetime.datetime, universe: Universe, state: State) -> List[TradeExecution]:
        """Run one strategy tick."""

        logger.info("Processing on_clock %s", clock)

        # TODO: Most of QSTrader execuion parts need to be rewritten to support these things better
        data_source = TradingStrategyDataSource(
            universe.exchanges,
            universe.pairs,
            universe.candles)

        strategy_assets = list(data_source.asset_bar_frames.keys())
        optimiser = FixedWeightPortfolioOptimiser()
        order_sizer = CashBufferedOrderSizer(state, self.pricing_method, self.cash_buffer)
        pcm = PortfolioConstructionModel(
            universe=universe,
            state=state,
            order_sizer=order_sizer,
            optimiser=optimiser,
            alpha_model=self.alpha_model,
            pricing_method=self.pricing_method,
            risk_model=None,
            cost_model=None)

        debug_details = {"clock": clock}
        rebalance_orders = pcm(pd.Timestamp(clock), stats=None, debug_details=debug_details)
        return rebalance_orders

    def preflight_check(self, client: Client, universe: Universe, now_: datetime.datetime):
        """Check the data looks more or less sane."""

        if len(universe.exchanges) == 0:
            raise PreflightCheckFailed("Exchange count zero")

        if universe.pairs.get_count() == 0:
            raise PreflightCheckFailed("Pair count zero")

        # Don't assume we have candle or liquidity data e.g. for the testing strategies
        if universe.candles.get_candle_count() > 0:
            start, end = universe.get_candle_availability()

            if now_ - end > self.max_data_age:
                raise PreflightCheckFailed(f"We do not have up-to-date data for candles. Last candles are at {end}")








