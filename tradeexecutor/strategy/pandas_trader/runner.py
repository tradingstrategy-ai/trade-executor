"""A strategy runner that executes Trading Strategy Pandas type strategies."""

import datetime
from io import StringIO
from typing import List, Optional
import logging

import pandas as pd

from tradeexecutor.cli.discord import post_logging_discord_image
from tradeexecutor.strategy.pandas_trader.trade_decision import TradeDecider
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse

from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.runner import StrategyRunner, PreflightCheckFailed
from tradeexecutor.visual.image_output import render_plotly_figure_as_image_file
from tradeexecutor.visual.strategy_state import draw_single_pair_strategy_state


logger = logging.getLogger(__name__)


class PandasTraderRunner(StrategyRunner):
    """A trading executor for Pandas math based algorithm."""

    def __init__(self, *args, decide_trades: TradeDecider, max_data_age: Optional[datetime.timedelta] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.decide_trades = decide_trades
        self.max_data_age = max_data_age

    def on_data_signal(self):
        pass

    def on_clock(self,
                 clock: datetime.datetime,
                 executor_universe: TradingStrategyUniverse,
                 pricing_model: PricingModel,
                 state: State,
                 debug_details: dict) -> List[TradeExecution]:
        """Run one strategy tick."""

        assert isinstance(executor_universe, TradingStrategyUniverse)
        universe = executor_universe.universe
        pd_timestamp = pd.Timestamp(clock)

        assert len(executor_universe.reserve_assets) == 1

        # Call the strategy script decide_trades()
        # callback
        return self.decide_trades(
            timestamp=pd_timestamp,
            universe=universe,
            state=state,
            pricing_model=pricing_model,
            cycle_debug_data=debug_details,
        )

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

    def report_strategy_thinking(self,
                                 clock: datetime.datetime,
                                 universe: TradingStrategyUniverse,
                                 state: State,
                                 trades: List[TradeExecution],
                                 debug_details: dict):
        """Strategy admin helpers to understand a live running strategy.

        - Post latest variables

        - Draw the single pair strategy visualisation.
        """

        visualisation = state.visualisation

        if universe.is_empty():
            # TODO: Not sure how we end up here
            return

        if universe.is_single_pair_universe():
            # Single pair thinking

            # Post strategy thinking image to Discord
            small_figure = draw_single_pair_strategy_state(state, universe, height=512)
            small_image = render_plotly_figure_as_image_file(small_figure, width=512, height=512, format="png")

            post_logging_discord_image(small_image)

            if self.execution_state:

                # Draw the inline plot and expose them tot he web serber
                large_figure = draw_single_pair_strategy_state(state, universe, height=1920)
                large_image = render_plotly_figure_as_image_file(large_figure, width=1920, height=1920, format="svg")

                self.execution_state.visualisation.update_image_data(small_image, large_image)

            # Log state
            buf = StringIO()

            pair = universe.get_single_pair()
            candles = universe.universe.candles.get_candles_by_pair(pair.internal_id)
            last_candle = candles.iloc[-1]
            lag = pd.Timestamp.utcnow().tz_localize(None) - last_candle["timestamp"]

            print("Strategy thinking", file=buf)
            print("", file=buf)
            print(f"  Now: {datetime.datetime.utcnow()} UTC", file=buf)
            print(f"  Last candle at: {last_candle['timestamp']} UTC, market data and action lag: {lag}", file=buf)
            print(f"  Price open:{last_candle['open']} close:{last_candle['close']} {pair.base.token_symbol} / {pair.quote.token_symbol}", file=buf)

            # Draw indicators
            for name, plot in visualisation.plots.items():
                value = plot.get_last_value()
                print(f"  {name}: {value}", file=buf)

            logger.trade(buf.getvalue())

        else:
            raise NotImplementedError("Reporting of strategy thinking of multipair universes not supported yet")
