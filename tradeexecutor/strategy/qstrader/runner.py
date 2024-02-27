"""Strategy runner for QSTrader based strategies."""
import datetime
from io import StringIO
from typing import List, Optional
import logging

import pandas as pd

from qstrader.portcon.optimiser.fixed_weight import FixedWeightPortfolioOptimiser

from tradeexecutor.strategy.execution_context import unit_test_execution_context
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInputIndicators
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.qstrader.alpha_model import AlphaModel
from tradeexecutor.strategy.qstrader.order_sizer import CashBufferedOrderSizer
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, translate_trading_pair

from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.qstrader.portfolio_construction_model import PortfolioConstructionModel
from tradeexecutor.strategy.runner import StrategyRunner, PreflightCheckFailed


logger = logging.getLogger(__name__)


class QSTraderRunner(StrategyRunner):
    """A live trading executor for QSTrade based algorithm.

    .. warning::

        This is legacy alpha version code and will be deprecated.
        It is only used in unit testing.

    """

    def __init__(self, *args, alpha_model: AlphaModel, max_data_age: Optional[datetime.timedelta] = None, cash_buffer=0.05, **kwargs):
        """
        :param alpha_model:
        :param timed_task_context_manager:
        :param max_data_age: Allow to unit test on old datasets
        """


        # Legacy code, used in tests only
        kwargs["unit_testing"] = True

        super().__init__(*args, **kwargs)
        assert isinstance(alpha_model, AlphaModel), f"We got {alpha_model}"
        self.alpha_model = alpha_model
        self.max_data_age = max_data_age
        # TODO: Make starter configuration
        self.cash_buffer = cash_buffer

        assert kwargs.get("routing_model"), "Routing model missing"

    # def report_strategy_thinking(self, clock: datetime.datetime, universe: TradingStrategyUniverse, state: State, trades: List[TradeExecution], debug_details: dict):
    #     """Report alpha model status."""
    #     buf = StringIO()
    #     universe = universe.universe
    #
    #     # TODO: move report_strategy_thinking() to a separate reporter class
    #
    #     data_start, data_end = universe.candles.get_timestamp_range()
    #     liquidity_start, liquidity_end = universe.liquidity.get_timestamp_range()
    #
    #     print("Strategy thinking", file=buf)
    #     print("", file=buf)
    #     print("Dataset status:", file=buf)
    #     print("", file=buf)
    #     print(f"   Cash buffer: {self.cash_buffer * 100:.2f}%", file=buf)
    #     print(f"   Candle dataset: {data_start} - {data_end}", file=buf)
    #     print(f"   Liquidity dataset: {liquidity_start} - {liquidity_end}", file=buf)
    #     print("", file=buf)
    #
    #     # Alpha model weights does not contain zero weight entries
    #     alpha_model_weights = debug_details["alpha_model_weights"]
    #
    #     # Normalised weights do contain zero weight entries
    #     normalised_weights = debug_details.get("normalised_weights", {})
    #
    #     if alpha_model_weights:
    #         print("Alpha model weights:", file=buf)
    #         print("", file=buf)
    #
    #         for pair_id, weight in alpha_model_weights.items():
    #             norm_weight = normalised_weights.get(pair_id, weight)
    #             pair = universe.pairs.get_pair_by_id(pair_id)
    #             tp = translate_trading_pair(pair)
    #             link = tp.info_url or ""
    #             if "extra_debug_data" in debug_details:
    #                 momentum = debug_details["extra_debug_data"][pair_id]["momentum"]
    #                 print(f"    {tp.get_human_description()} weight:{norm_weight*100:.2f}%, momentum:{momentum*100:.2f}%", file=buf)
    #                 if link:
    #                     print(f"    link: {link}", file=buf)
    #                 print("", file=buf)
    #     else:
    #         print("Error: Could not calculate any momentum! Data missing?", file=buf)
    #
    #     good_candle_count = debug_details.get("good_candle_count")
    #     problem_candle_count = debug_details.get("problem_candle_count")
    #     low_liquidity_count = debug_details.get("low_liquidity_count")
    #     bad_momentum_count = debug_details.get("bad_momentum_count")
    #     funny_price_count = debug_details.get("funny_price_count")
    #     candle_range_start = debug_details.get("candle_range_start")
    #     candle_range_end = debug_details.get("candle_range_end")
    #     print("", file=buf)
    #     print("Alpha model data quality:", file=buf)
    #     print("", file=buf)
    #     print(f"   Evaluated momentum range: {candle_range_start} - {candle_range_end}", file=buf)
    #     print(f"   Pairs with good candles data: {good_candle_count}", file=buf)
    #     print(f"   Pairs with bad price value: {funny_price_count}", file=buf)
    #     print(f"   Pairs with negative momentum result: {bad_momentum_count}", file=buf)
    #     print(f"   Pairs with problems in candles data: {problem_candle_count}", file=buf)
    #     print(f"   Pairs with low liquidity: {low_liquidity_count}", file=buf)
    #
    #     logger.trade(buf.getvalue())

    def on_data_signal(self):
        pass

    def on_clock(self,
                 clock: datetime.datetime,
                 executor_universe: TradingStrategyUniverse,
                 pricing_model: PricingModel,
                 state: State,
                 debug_details: dict,
                 indicators: StrategyInputIndicators | None = None,
                 ) -> List[TradeExecution]:
        """Run one strategy cycle.

        - Takes universe, pricing model and state as an input

        - Generates a list of new trades to change the current state
        """

        assert isinstance(executor_universe, TradingStrategyUniverse)

        universe = executor_universe.data_universe
        reserve_assets = executor_universe.reserve_assets
        logger.info("QSTrader on_clock %s", clock)
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
        universe = universe.data_universe

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

        # Check that the web3 conneciontion works by doing eth_call to a smart contract
        # TODO








