import datetime
from io import StringIO
from typing import List, Optional
import logging

import pandas as pd

from qstrader.portcon.optimiser.fixed_weight import FixedWeightPortfolioOptimiser
from tradeexecutor.strategy.qstrader.alpha_model import AlphaModel
from tradeexecutor.strategy.qstrader.order_sizer import CashBufferedOrderSizer
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, translate_trading_pair

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
        assert isinstance(alpha_model, AlphaModel), f"We got {alpha_model}"
        self.alpha_model = alpha_model
        self.max_data_age = max_data_age
        # TODO: Make starter configuration
        self.cash_buffer = cash_buffer

    def report_strategy_thinking(self, clock: datetime.datetime, universe: TradingStrategyUniverse, state: State, trades: List[TradeExecution], debug_details: dict):
        """Report alpha model status."""
        buf = StringIO()
        universe = universe.universe

        data_start, data_end = universe.candles.get_timestamp_range()
        liquidity_start, liquidity_end = universe.liquidity.get_timestamp_range()

        print("Strategy thinking", file=buf)
        print("", file=buf)
        print("Dataset status:", file=buf)
        print("", file=buf)
        print(f"   Cash buffer: {self.cash_buffer * 100:.2f}%", file=buf)
        print(f"   Candle dataset: {data_start} - {data_end}", file=buf)
        print(f"   Liquidity dataset: {liquidity_start} - {liquidity_end}", file=buf)
        print("", file=buf)

        alpha_model_weights = debug_details["alpha_model_weights"]

        if alpha_model_weights:
            print("Alpha model weights:", file=buf)
            print("", file=buf)

            for pair_id, weight in alpha_model_weights.items():
                pair = universe.pairs.get_pair_by_id(pair_id)
                tp = translate_trading_pair(pair)
                link = tp.info_url or ""
                momentum = debug_details["extra_debug_data"][pair_id]["momentum"]
                print(f"    {tp.get_human_description()} weight:{weight:.2f}, momentum:{momentum*100:.2f}%", file=buf)
                print(f"    link: {link}", file=buf)
                print("", file=buf)
        else:
            print("Error: Could not calculate any momentum! Data missing?", file=buf)

        good_candle_count = debug_details["good_candle_count"]
        problem_candle_count = debug_details["problem_candle_count"]
        low_liquidity_count = debug_details["low_liquidity_count"]
        bad_momentum_count = debug_details["bad_momentum_count"]
        funny_price_count = debug_details["funny_price_count"]
        candle_range_start = debug_details["candle_range_start"]
        candle_range_end = debug_details["candle_range_end"]
        print("", file=buf)
        print("Alpha model data quality:", file=buf)
        print("", file=buf)
        print(f"   Evaluated momentum range: {candle_range_start} - {candle_range_end}", file=buf)
        print(f"   Pairs with good candles data: {good_candle_count}", file=buf)
        print(f"   Pairs with bad price value: {funny_price_count}", file=buf)
        print(f"   Pairs with negative momentum result: {bad_momentum_count}", file=buf)
        print(f"   Pairs with problems in candles data: {problem_candle_count}", file=buf)
        print(f"   Pairs with low liquidity: {low_liquidity_count}", file=buf)

        logger.trade(buf.getvalue())

    def on_data_signal(self):
        pass

    def on_clock(self, clock: datetime.datetime, executor_universe: TradingStrategyUniverse, state: State, debug_details: dict) -> List[TradeExecution]:
        """Run one strategy tick."""

        assert isinstance(executor_universe, TradingStrategyUniverse)

        universe = executor_universe.universe
        reserve_assets = executor_universe.reserve_assets
        logger.info("QSTrader on_clock %s", clock)
        pricing_model = self.pricing_model_factory(self.execution_model, executor_universe)
        optimiser = FixedWeightPortfolioOptimiser()
        order_sizer = CashBufferedOrderSizer(state, pricing_model, self.cash_buffer)
        pcm = PortfolioConstructionModel(
            universe=universe,
            state=state,
            order_sizer=order_sizer,
            optimiser=optimiser,
            alpha_model=self.alpha_model,
            pricing_model=pricing_model,
            reserve_currency=reserve_assets[0],
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

            if self.max_data_age is not None:
                if now_ - end > self.max_data_age:
                    raise PreflightCheckFailed(f"We do not have up-to-date data for candles. Last candles are at {end}")








