"""A strategy runner that executes Trading Strategy Pandas type strategies."""

import datetime
from io import StringIO
from typing import List, Optional
import logging

import pandas as pd

from tradeexecutor.cli.discord import post_logging_discord_image
from tradeexecutor.strategy.pandas_trader.trade_decision import TradeDecider
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.sync_model import SyncModel
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

        # Legacy assets
        sync_model = kwargs.get("sync_model")
        if sync_model is not None:
            assert isinstance(sync_model, SyncModel)

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

        assert state.sync.treasury.last_updated_at is not None, "Cannot do trades before treasury is synced at least once"
        # All sync models do not emit events correctly yet
        # assert len(state.sync.treasury.balance_update_refs) > 0, "No deposit detected. Please do at least one deposit before starting the strategy"
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
        if universe.candles is not None:
            if universe.candles.get_candle_count() > 0:
                start, end = universe.get_candle_availability()

                if self.max_data_age is not None:
                    if now_ - end > self.max_data_age:
                        raise PreflightCheckFailed(f"We do not have up-to-date data for candles. Last candles are at {end}")

    def report_strategy_thinking(self,
                                 strategy_cycle_timestamp: datetime.datetime,
                                 cycle: int,
                                 universe: TradingStrategyUniverse,
                                 state: State,
                                 trades: List[TradeExecution],
                                 debug_details: dict):
        """Strategy admin helpers to understand a live running strategy.

        - Post latest variables

        - Draw the single pair strategy visualisation.

        :param strategy_cycle_timestamp:
            real time lock

        :param cycle:
            Cycle number

        :param universe:
            Currnet trading universe

        :param trades:
            Trades executed on this cycle

        :param state:
            Current execution state

        :param debug_details:
            Dict of random debug stuff
        """

        visualisation = state.visualisation

        if universe.is_empty():
            # TODO: Not sure how we end up here
            logger.info("Strategy universe is empty - nothing to report")
            return

        if universe.is_single_pair_universe():
            # Single pair thinking

            # Post strategy thinking image to Discord
            small_figure = draw_single_pair_strategy_state(state, universe, height=512)
            small_image = render_plotly_figure_as_image_file(small_figure, width=768, height=512, format="png")

            small_figure.update_layout(template="plotly_dark")
            small_image_dark = render_plotly_figure_as_image_file(small_figure, width=512, height=512, format="png")

            post_logging_discord_image(small_image)

            if self.run_state:

                logger.info("Updating strategy thinking image data")

                # Draw the inline plot and expose them tot he web server
                # TODO: SVGs here are not very readable, have them as a stop gap solution
                large_figure = draw_single_pair_strategy_state(state, universe, height=1024)
                large_image = render_plotly_figure_as_image_file(large_figure, width=1024, height=1024, format="svg")

                large_figure.update_layout(template="plotly_dark")
                large_image_dark = render_plotly_figure_as_image_file(large_figure, width=1024, height=1024, format="svg")

                self.run_state.visualisation.update_image_data(
                    small_image,
                    large_image,
                    small_image_dark,
                    large_image_dark,
                )
            else:
                logger.info("Coudl not update strategy thinking image data, self.execution_state not available")

            # Log state
            buf = StringIO()

            pair = universe.get_single_pair()
            candles = universe.universe.candles.get_candles_by_pair(pair.internal_id)
            last_candle = candles.iloc[-1]
            lag = pd.Timestamp.utcnow().tz_localize(None) - last_candle["timestamp"]

            print("Strategy thinking", file=buf)
            print("", file=buf)
            print(f"  Strategy cycle #{cycle}: {strategy_cycle_timestamp} UTC, now is {datetime.datetime.utcnow()}", file=buf)
            print(f"  Last candle at: {last_candle['timestamp']} UTC, market data and action lag: {lag}", file=buf)
            print(f"  Price open:{last_candle['open']} close:{last_candle['close']} {pair.base.token_symbol} / {pair.quote.token_symbol}", file=buf)

            # Draw indicators
            for name, plot in visualisation.plots.items():
                value = plot.get_last_value()
                print(f"  {name}: {value}", file=buf)

            logger.trade(buf.getvalue())

        else:
            raise NotImplementedError("Reporting of strategy thinking of multipair universes not supported yet")
