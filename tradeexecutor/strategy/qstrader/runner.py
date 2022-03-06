import datetime
from typing import List, Optional
import logging

import pandas as pd

from qstrader.portcon.optimiser.fixed_weight import FixedWeightPortfolioOptimiser
from tradeexecutor.strategy.qstrader.alpha_model import AlphaModel
from tradeexecutor.strategy.qstrader.order_sizer import CashBufferedOrderSizer
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse

from tradeexecutor.state.state import State, TradeExecution
from tradeexecutor.strategy.qstrader.portfolio_construction_model import PortfolioConstructionModel
from tradeexecutor.strategy.runner import StrategyRunner, PreflightCheckFailed


logger = logging.getLogger(__name__)


class QSTraderRunner(StrategyRunner):
    """A live trading executor for QSTrade based algorithm."""

    def __init__(self, *args, alpha_model: AlphaModel, max_data_age: Optional[datetime.timedelta] = None, cash_buffer=0.05, **kwargs):
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

    def on_clock(self, clock: datetime.datetime, executor_universe: TradingStrategyUniverse, state: State, debug_details: dict) -> List[TradeExecution]:
        """Run one strategy tick."""

        universe = executor_universe.universe
        logger.info("QSTrader on_clock %s", clock)
        assert len(self.reserve_assets) == 1, f"We only support strategies with a single reserve asset, got {self.reserve_assets}"
        pricing_model = self.pricing_model_factory(self.execution_model, universe)
        optimiser = FixedWeightPortfolioOptimiser()
        order_sizer = CashBufferedOrderSizer(state, pricing_model, self.cash_buffer)
        pcm = PortfolioConstructionModel(
            universe=universe,
            state=state,
            order_sizer=order_sizer,
            optimiser=optimiser,
            alpha_model=self.alpha_model,
            pricing_model=pricing_model,
            reserve_currency=self.reserve_assets[0],
            risk_model=None,
            cost_model=None)

        rebalance_trades = pcm(pd.Timestamp(clock), stats=None, debug_details=debug_details)
        return rebalance_trades

    def pretick_check(self, ts: datetime.datetime, universe: TradingStrategyUniverse):
        """Check the data looks more or less sane."""

        assert isinstance(universe, TradingStrategyUniverse)
        universe = universe.universe

        now_ = ts

        if len(universe.exchanges) == 0:
            raise PreflightCheckFailed("Exchange count zero")

        if universe.pairs.get_count() == 0:
            raise PreflightCheckFailed("Pair count zero")

        # Don't assume we have candle or liquidity data e.g. for the testing strategies
        if universe.candles.get_candle_count() > 0:
            start, end = universe.get_candle_availability()

            if now_ - end > self.max_data_age:
                raise PreflightCheckFailed(f"We do not have up-to-date data for candles. Last candles are at {end}")








