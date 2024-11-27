"""A strategy runner that executes Trading Strategy Pandas type strategies."""

import datetime
import textwrap
from io import StringIO
from typing import List, Optional
import logging

import pandas as pd

from tradeexecutor.cli.discord import post_logging_discord_image
from tradeexecutor.ethereum.execution import EthereumExecution
from tradeexecutor.statistics.in_memory_statistics import refresh_live_strategy_images
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet, calculate_and_load_indicators, MemoryIndicatorStorage, call_create_indicators
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput, StrategyInputIndicators
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.strategy_module import DecideTradesProtocol, DecideTradesProtocol2, DecideTradesProtocol3, DecideTradesProtocol4
from tradeexecutor.strategy.sync_model import SyncModel
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, translate_trading_pair

from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.runner import StrategyRunner, PreflightCheckFailed
from tradeexecutor.utils.timestamp import convert_and_validate_timestamp_as_int
from tradeexecutor.visual.image_output import render_plotly_figure_as_image_file
from tradeexecutor.visual.strategy_state import draw_single_pair_strategy_state, draw_multi_pair_strategy_state


logger = logging.getLogger(__name__)


class PandasTraderRunner(StrategyRunner):
    """A trading executor for Pandas math based algorithm."""

    def __init__(
            self,
            *args,
            decide_trades: DecideTradesProtocol | DecideTradesProtocol2 | DecideTradesProtocol3 | DecideTradesProtocol4,
            max_data_age: datetime.timedelta = None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.decide_trades = decide_trades
        self.max_data_age = max_data_age

        # Legacy assets
        sync_model = kwargs.get("sync_model")
        if sync_model is not None:
            assert isinstance(sync_model, SyncModel)

    def on_data_signal(self):
        pass

    def on_clock(
        self,
        clock: datetime.datetime,
        strategy_universe: TradingStrategyUniverse,
        pricing_model: PricingModel,
        state: State,
        debug_details: dict,
        indicators:StrategyInputIndicators | None = None,
        ) -> List[TradeExecution]:
        """Run one strategy tick."""

        assert isinstance(strategy_universe, TradingStrategyUniverse)
        universe = strategy_universe.data_universe
        pd_timestamp = pd.Timestamp(clock)

        assert state.sync.treasury.last_updated_at is not None, "Cannot do trades before treasury is synced at least once"
        # All sync models do not emit events correctly yet
        # assert len(state.sync.treasury.balance_update_refs) > 0, "No deposit detected. Please do at least one deposit before starting the strategy"
        assert len(strategy_universe.reserve_assets) == 1

        # Call the strategy script decide_trades()
        # callback
        if self.execution_context.is_version_greater_or_equal_than(0, 5, 0):
            # DecideTradesProtocolV4

            if self.execution_context.mode.is_live_trading():
                # Indicators are recalculated for every tick in the live trading
                assert self.create_indicators is not None, "trading_strategy_engine_version > 0.5, but we lack create_indicators"
                assert self.parameters is not None, "trading_strategy_engine_version > 0.5, but we lack parameters"
                indicators = self.calculate_live_indicators(clock, strategy_universe, self.parameters)

            assert indicators is not None, "indicators not created when running trading_strategy_engine_version=0.5"
            indicators.prepare_decision_cycle(debug_details["cycle"], pd_timestamp)

            if isinstance(self.execution_model, EthereumExecution):
                # Need by fetch_quote_token_tvls()
                web3 = self.execution_model.web3
            else:
                # Backtesting, etc.
                web3 = None

            input = StrategyInput(
                cycle=debug_details["cycle"],
                timestamp=pd_timestamp,
                strategy_universe=strategy_universe,
                state=state,
                pricing_model=pricing_model,
                other_data=debug_details,
                indicators=indicators,
                parameters=self.parameters,
                execution_context=self.execution_context,
                web3=web3,
            )

            logger.info(
                "Running decide_trades(), timestamp %s, cash in hand %s",
                pd_timestamp,
                input.get_position_manager().get_current_cash()
            )
            return self.decide_trades(
                input
            )
        elif self.execution_context.is_version_greater_or_equal_than(0, 4, 0):
            parameters = self.execution_context.parameters
            # DecideTradesProtocolV3
            parameters["cycle"] = debug_details["cycle"]
            return self.decide_trades(
                timestamp=pd_timestamp,
                parameters=parameters,
                strategy_universe=strategy_universe,
                state=state,
                pricing_model=pricing_model,
            )
        elif self.execution_context.is_version_greater_or_equal_than(0, 3, 0):
            return self.decide_trades(
                timestamp=pd_timestamp,
                strategy_universe=strategy_universe,
                state=state,
                pricing_model=pricing_model,
                cycle_debug_data=debug_details,
            )
        else:
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
        universe = universe.data_universe

        now_ = ts

        if len(universe.exchanges) == 0:
            raise PreflightCheckFailed("Exchange count zero")

        if universe.pairs.get_count() == 0:
            raise PreflightCheckFailed("Pair count zero")

        # Don't assume we have candle or liquidity data e.g. for the testing strategies
        if universe.candles is not None:
            if universe.candles.get_candle_count() > 0:
                start, end = universe.candles.get_timestamp_range()

                if self.max_data_age is not None:
                    if now_ - end > self.max_data_age:
                        raise PreflightCheckFailed(f"We do not have up-to-date data for candles. Last candles are at {end}")

    def refresh_visualisations(self, state: State, universe: TradingStrategyUniverse):
        """Updates the visualisation images for the strategy.

        - Used in Discord (small)

        - Used on the frontend (large)

        This is automatically called on trade-executor console startup:

            docker compose run enzyme-polygon-eth-btc-usdc console

        To call this manually from the same console with pre-set up runner

        .. code-block:: shell

            runner.refresh_visualisations(state, strategy_universe)
        """

        if not self.run_state:
            # This strategy is not maintaining a run-state
            # Backtest, simulation, etc.
            logger.info("Could not update strategy thinking image data, self.run_state not available")
            return

        pair_count = universe.get_pair_count()

        logger.info("Refreshing strategy visualisations: %s, pair count is %d",
                    self.run_state.visualisation,
                    pair_count
                    )

        if not self.visualisation:
            logger.info("Visualisation disabled with VISUALISATION environment variable")

        execution_context = self.execution_context

        if universe.is_empty():
            # TODO: Not sure how we end up here
            logger.info("Strategy universe is empty - nothing to report")
            return

        try:

            if pair_count == 1:

                small_figure = draw_single_pair_strategy_state(state, execution_context, universe, height=512)
                # Draw the inline plot and expose them tot he web server
                # TODO: SVGs here are not very readable, have them as a stop gap solution
                large_figure = draw_single_pair_strategy_state(state, execution_context, universe, height=1024)

                self.update_strategy_thinking_image_data(small_figure, large_figure)

            elif 1 < pair_count <= 3:

                small_figure_combined = draw_multi_pair_strategy_state(state, execution_context, universe,  height=1024)
                large_figure_combined = draw_multi_pair_strategy_state(state, execution_context, universe, height=2048)

                self.update_strategy_thinking_image_data(small_figure_combined, large_figure_combined)

            elif 3 < pair_count <=5:

                small_figure_combined = draw_multi_pair_strategy_state(state, execution_context, universe, height=2048, detached_indicators = False)
                large_figure_combined = draw_multi_pair_strategy_state(state, execution_context, universe, height=3840, width = 2160, detached_indicators = False)

                self.update_strategy_thinking_image_data(small_figure_combined, large_figure_combined)

            else:
                logger.warning("Charts not yet available for this strategy type. Pair count: %d", pair_count)

        except Exception as e:
            # Don't take trade executor down if visualisations fail
            logger.warning("Could not draw visualisations in refresh_visualisations()")
            logger.warning("Visualisation exception %s", e, exc_info=e)

    def update_strategy_thinking_image_data(self, small_figure, large_figure):
        """Update the strategy thinking image data with small, small dark theme, large, and large dark theme images.
        
        :param small_image: 512 x 512 image
        :param large_image: 1920 x 1920 image
        """
        execution_context = self.execution_context
        refresh_live_strategy_images(self.run_state, execution_context, small_figure, large_figure)

    def report_strategy_thinking(
        self,
        strategy_cycle_timestamp: datetime.datetime,
        cycle: int,
        universe: TradingStrategyUniverse,
        state: State,
        trades: List[TradeExecution],
        debug_details: dict
    ):
        """Strategy admin helpers to understand a live running strategy.

        - Post latest variables

        - Draw the single pair strategy visualisation.

        To manually test the visualisation see: `manual-visualisation-test.py`.

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

        # Update charts
        self.refresh_visualisations(state, universe)

        visualisation = state.visualisation

        if universe.is_empty():
            # TODO: Not sure how we end up here
            logger.info("Strategy universe is empty - nothing to report")
            return

        if universe.is_single_pair_universe():
            # Log state
            buf = StringIO()

            pair = universe.get_single_pair()
            candles = universe.data_universe.candles.get_candles_by_pair(pair.internal_id)
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

            small_image = self.run_state.visualisation.small_image_png
            if small_image is not None:
                post_logging_discord_image(small_image)
            else:
                logger.warning("Chart visualisation missing")

        else:
            
             # Log state
            buf = StringIO()

            print("Strategy thinking", file=buf)
            print(f"  Strategy cycle #{cycle}: {strategy_cycle_timestamp} UTC, now is {datetime.datetime.utcnow()}", file=buf)

            for pair_id, candles in universe.data_universe.candles.get_all_pairs(max_count=3):
                
                pair = universe.data_universe.pairs.get_pair_by_id(pair_id)
                pair_slug = f"{pair.base_token_symbol} / {pair.quote_token_symbol}"

                print(f"\n  {pair_slug}", file=buf)

                lag = None
                timestamp = None
                last_candle = None
                if len(candles) > 0:
                    last_candle = candles.iloc[-1]
                    try:
                        timestamp = last_candle["timestamp"]
                        lag = pd.Timestamp.utcnow().tz_localize(None) - timestamp
                    except:
                        logger.warning("Cannot read timestamp")
                else:
                    logger.warning("Pair %s had not candle data", pair)

                dex_pair = universe.data_universe.pairs.get_pair_by_id(pair_id)
                pair = translate_trading_pair(dex_pair)

                if not pair:
                    logger.warning(f"  Pair missing: {dex_pair} - should not happen")
                else:
                    print(f"  Last candle at: {timestamp} UTC, market data and action lag: {lag}", file=buf)
                    if last_candle is not None:
                        print(f"  Price open:{last_candle['open']}", file=buf)
                        print(f"  Close:{last_candle['close']}", file=buf)

                # Draw indicators
                for name, plot in visualisation.plots.items():
                    
                    if getattr(plot.pair, "internal_id", None) is None:
                        logger.warning(f"  Plot {name} has no pair argument. To see indicator values for individual pairs in a multipair strategy, add pair argument to the `plot_indicator` function in your strategy file.")
                        continue
                    
                    if plot.pair.internal_id != pair_id:
                        continue

                    value = plot.get_last_value()
                    print(f"  {name}: {value}", file=buf)
                
            logger.trade(buf.getvalue())

            # there is already a warning in refresh_visualisations for pair count > 3
            if universe.get_pair_count() <= 5:
                large_image = self.run_state.visualisation.large_image_png
                post_logging_discord_image(large_image)
            else:
                logger.info(f"Strategy visualisation not posted to Discord because pair count of {universe.get_pair_count()} is greater than 5.")

        # Print the thinking message decide_trades() can add with
        # visualisation.add_message()
        unix_timestamp = convert_and_validate_timestamp_as_int(strategy_cycle_timestamp)
        messages = visualisation.messages.get(unix_timestamp)
        if messages:
            thinking_message = messages[0]
            logger.trade(textwrap.indent(thinking_message, "    "))

    def calculate_live_indicators(
        self,
        timestamp: datetime.datetime,
        strategy_universe: TradingStrategyUniverse,
        parameters: StrategyParameters,
    ) -> StrategyInputIndicators:
        """Calculate and recalculate indicators in a live trading.

        - Calculated just before `decide_trades` is called

        - Recalculated for every cycle

        :return:
            Freshly calculated indicators
        """
        # storage = self.indicator_storage
        logger.info("Calculating live indicators for %s", timestamp)

        storage = MemoryIndicatorStorage(strategy_universe.get_cache_key())

        indicators = call_create_indicators(
            self.create_indicators,
            parameters,
            strategy_universe,
            self.execution_context,
            timestamp,
        )

        indicator_results = calculate_and_load_indicators(
            strategy_universe=strategy_universe,
            storage=storage,
            execution_context=self.execution_context,
            indicators=indicators,
            parameters=self.parameters,
            timestamp=timestamp,
        )

        strategy_input_indicators = StrategyInputIndicators(
            strategy_universe,
            indicator_results=indicator_results,
            available_indicators=indicators,
        )

        return strategy_input_indicators
