"""Equally-weighted index to be used in preprocessed backtest reports

- See :py:mod:`preprocesed_backtest`
"""
from datetime import datetime

import pandas as pd

from tradeexecutor.backtest.backtest_runner import run_backtest_inline
from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.types import USDollarAmount
from tradeexecutor.strategy.alpha_model import AlphaModel
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSource, IndicatorDependencyResolver, calculate_and_load_indicators_inline
from tradeexecutor.strategy.pandas_trader.indicator_decorator import IndicatorRegistry
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.strategy.tvl_size_risk import USDTVLSizeRiskModel
from tradeexecutor.strategy.weighting import weight_passthrouh
from tradingstrategy.client import Client
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.utils.forward_fill import forward_fill


def _run_equally_weighted_index(
    client: Client,
    strategy_universe: TradingStrategyUniverse,
    dataset: "tradeexecutor.backtest.preprocessed_backtest.SavedDataset"
):
    class Parameters:
        id = "_run_equally_weighted_index"

        # We trade 1h candle
        candle_time_bucket = TimeBucket.h4
        cycle_duration = CycleDuration.cycle_4h

        # Coingecko categories to include
        # s
        # See list here: TODO
        #
        chain_id = dataset.set.chain
        exchanges = dataset.set.exchanges

        #
        # Basket construction and rebalance parameters
        #
        min_asset_universe = 5  # How many assets we need in the asset universe to start running the index
        max_assets_in_portfolio = 10  # How many assets our basket can hold once
        allocation = 0.95  # Allocate all cash to volatile pairs
        # min_rebalance_trade_threshold_pct = 0.05  # % of portfolio composition must change before triggering rebalacne
        individual_rebalance_min_threshold_usd = 75.0  # Don't make buys less than this amount
        per_position_cap_of_pool = 0.01  # Never own more than % of the lit liquidity of the trading pool
        max_concentration = 0.20  # How large % can one asset be in a portfolio once
        min_portfolio_weight = 0.0050  # Close position / do not open if weight is less than 50 BPS

        # For the length of trailing sharpe used in inclusion criteria
        rolling_volume_bars = pd.Timedelta("7d") // candle_time_bucket.to_timedelta()
        rolling_volatility_bars = pd.Timedelta("7d") // candle_time_bucket.to_timedelta()
        tvl_ewm_span = 7 * 24  # Smooth TVL inclusin criteria
        min_volume = 125_000  # USD
        min_tvl = dataset.set.min_tvl
        min_token_sniffer_score = 20  # 20 = AAVE

        #
        # Yield on cash
        #
        use_aave = False
        credit_flow_dust_threshold = 5.0  # Min deposit USD to Aave

        #
        #
        # Backtesting only
        # Limiting factor: Aave v3 on Base starts at the end of DEC 2023
        #
        backtest_start = dataset.set.start
        backtest_end = dataset.set.end
        initial_cash = 100_000

        #
        # Live only
        #
        routing = TradeRouting.default
        required_history_period = datetime.timedelta(days=2 * 14 + 1)
        slippage_tolerance = 0.0060  # 0.6%
        assummed_liquidity_when_data_missings = 10_000

    parameters = StrategyParameters.from_class(Parameters)

    indicators = IndicatorRegistry()

    empty_series = pd.Series([], index=pd.DatetimeIndex([]))

    @indicators.define()
    def volatility(close: pd.Series, rolling_volatility_bars: int) -> pd.Series:
        """Calculate the rolling volatility for rebalancing the index for each decision cycle."""
        price_diff = close.pct_change()
        rolling_std = price_diff.rolling(window=rolling_volatility_bars).std()
        return rolling_std

    @indicators.define()
    def rolling_cumulative_volume(volume: pd.Series, rolling_volume_bars: int) -> pd.Series:
        """Calculate rolling volume of the pair.

        - Used in inclusion criteria
        """
        rolling_volume = volume.rolling(window=rolling_volume_bars).sum()
        return rolling_volume


    @indicators.define()
    def rolling_liquidity_avg(close: pd.Series, rolling_volume_bars: int) -> pd.Series:
        """Calculate rolling liquidity average

        - This is either TVL or XY liquidity (one sided) depending on the trading pair DEX type

        - Used in inclusion criteria
        """
        rolling_liquidity_close = close.rolling(window=rolling_volume_bars).mean()
        return rolling_liquidity_close


    @indicators.define(dependencies=(rolling_cumulative_volume,), source=IndicatorSource.strategy_universe)
    def volume_inclusion_criteria(
            strategy_universe: TradingStrategyUniverse,
            min_volume: USDollarAmount,
            rolling_volume_bars: int,
            dependency_resolver: IndicatorDependencyResolver,
    ) -> pd.Series:
        """Calculate pair volume inclusion criteria.

        - Avoid including illiquid / broken pairs in the set: Pair is included when it has enough volume

        TODO: Add liquidity check later

        :return:
            Series where each timestamp is a list of pair ids meeting the criteria at that timestamp
        """

        series = dependency_resolver.get_indicator_data_pairs_combined(
            rolling_cumulative_volume,
            parameters={"rolling_volume_bars": rolling_volume_bars},
        )

        # Get mask for days when the rolling volume meets out criteria
        mask = series >= min_volume
        mask_true_values_only = mask[mask == True]

        # Turn to a series of lists
        series = mask_true_values_only.groupby(level='timestamp').apply(lambda x: x.index.get_level_values('pair_id').tolist())
        return series


    @indicators.define(source=IndicatorSource.tvl)
    def tvl(
        close: pd.Series,
        execution_context: ExecutionContext,
        timestamp: datetime.datetime,
    ) -> pd.Series:
        """Get TVL series for a pair.

        - Because TVL data is 1d and we use 1h everywhere else, we need to forward fill

        - Use previous hourly close as the value
        """
        if execution_context.live_trading:
            # TVL is daily data.
            # We need to forward fill until the current hour.
            # Use our special ff function.
            assert isinstance(timestamp, pd.Timestamp), f"Live trading needs forward-fill end time, we got {timestamp}"
            df = pd.DataFrame({"close": close})
            df_ff = forward_fill(
                df,
                Parameters.candle_time_bucket.to_frequency(),
                columns=("close",),
                forward_fill_until=timestamp,
            )
            series = df_ff["close"]
            return series
        else:
            return close.resample("1h").ffill()


    @indicators.define(dependencies=(tvl,), source=IndicatorSource.dependencies_only_per_pair)
    def tvl_ewm(
            pair: TradingPairIdentifier,
            tvl_ewm_span: float,
            dependency_resolver: IndicatorDependencyResolver,
    ) -> pd.Series:
        """Get smoothed TVL series for a pair.

        - Interpretation: If you set span=5, for example, the ewm function will compute an exponential moving average where the weight of the most recent observation is about 33.3% (since α=2/(5+1)≈0.333) and this weight decreases exponentially for older observations.

        - We forward fill gaps, so there is no missing data in decide_trades()

        - Currently unused in the strategy itself
        """
        tvl_ff = dependency_resolver.get_indicator_data(
            tvl,
            pair=pair,
        )
        return tvl_ff.ewm(span=tvl_ewm_span).mean()


    @indicators.define(dependencies=(tvl_ewm, tvl), source=IndicatorSource.dependencies_only_universe)
    def tvl_inclusion_criteria(
            min_tvl: USDollarAmount,
            dependency_resolver: IndicatorDependencyResolver,
    ) -> pd.Series:
        """The pair must have min XX,XXX USD one-sided TVL to be included.

        - If the Uniswap pool does not have enough ETH or USDC deposited, skip the pair as a scam

        :return:
            Series where each timestamp is a list of pair ids meeting the criteria at that timestamp
        """

        series = dependency_resolver.get_indicator_data_pairs_combined(tvl)
        mask = series >= min_tvl
        # Turn to a series of lists
        mask_true_values_only = mask[mask == True]
        series = mask_true_values_only.groupby(level='timestamp').apply(lambda x: x.index.get_level_values('pair_id').tolist())
        return series


    @indicators.define(
        source=IndicatorSource.strategy_universe
    )
    def trading_availability_criteria(
            strategy_universe: TradingStrategyUniverse,
    ) -> pd.Series:
        """Is pair tradeable at each hour.

        - The pair has a price candle at that
        - Mitigates very corner case issues that TVL/liquidity data is per-day whileas price data is natively per 1h
          and the strategy inclusion criteria may include pair too early hour based on TVL only,
          leading to a failed attempt to rebalance in a backtest
        - Only relevant for backtesting issues if we make an unlucky trade on the starting date
          of trading pair listing

        :return:
            Series with with index (timestamp) and values (list of pair ids trading at that hour)
        """
        # Trading pair availability is defined if there is a open candle in the index for it.
        # Because candle data is forward filled, we should not have any gaps in the index.
        candle_series = strategy_universe.data_universe.candles.df["open"]
        pairs_per_timestamp = candle_series.groupby(level='timestamp').apply(lambda x: x.index.get_level_values('pair_id').tolist())
        return pairs_per_timestamp


    @indicators.define(
        dependencies=[
            volume_inclusion_criteria,
            tvl_inclusion_criteria,
            trading_availability_criteria
        ],
        source=IndicatorSource.strategy_universe
    )
    def inclusion_criteria(
            strategy_universe: TradingStrategyUniverse,
            min_volume: USDollarAmount,
            rolling_volume_bars: int,
            min_tvl: USDollarAmount,
            dependency_resolver: IndicatorDependencyResolver
    ) -> pd.Series:
        """Pairs meeting all of our inclusion criteria.

        - Give the tradeable pair set for each timestamp

        :return:
            Series where index is timestamp and each cell is a list of pair ids matching our inclusion criteria at that moment
        """

        # Filter out benchmark pairs like WETH in the tradeable pair set
        benchmark_pair_ids = set(strategy_universe.get_pair_by_human_description(desc).internal_id for desc in SUPPORTING_PAIRS)

        volume_series = dependency_resolver.get_indicator_data(
            volume_inclusion_criteria,
            parameters={
                "min_volume": min_volume,
                "rolling_volume_bars": rolling_volume_bars,
            },
        )

        tvl_series = dependency_resolver.get_indicator_data(
            tvl_inclusion_criteria,
            parameters={
                "min_tvl": min_tvl,
            },
        )

        trading_availability_series = dependency_resolver.get_indicator_data(trading_availability_criteria)

        #
        # Process all pair ids as a set and the final inclusion
        # criteria is union of all sub-criterias
        #

        df = pd.DataFrame({
            "tvl_pair_ids": tvl_series,
            "volume_pair_ids": volume_series,
            "trading_availability_pair_ids": trading_availability_series,
        })

        # https://stackoverflow.com/questions/33199193/how-to-fill-dataframe-nan-values-with-empty-list-in-pandas
        df = df.fillna("").apply(list)

        def _combine_criteria(row):
            final_set = set(row["volume_pair_ids"]) & \
                        set(row["tvl_pair_ids"]) & \
                        set(row["trading_availability_pair_ids"])
            return final_set - benchmark_pair_ids

        union_criteria = df.apply(_combine_criteria, axis=1)

        # Inclusion criteria data can be spotty at the beginning when there is only 0 or 1 pairs trading,
        # so we need to fill gaps to 0
        full_index = pd.date_range(
            start=union_criteria.index.min(),
            end=union_criteria.index.max(),
            freq=Parameters.candle_time_bucket.to_frequency(),
        )
        reindexed = union_criteria.reindex(full_index, fill_value=[])
        return reindexed


    @indicators.define(dependencies=(volume_inclusion_criteria,), source=IndicatorSource.dependencies_only_universe)
    def volume_included_pair_count(
            min_volume: USDollarAmount,
            rolling_volume_bars: int,
            dependency_resolver: IndicatorDependencyResolver
    ) -> pd.Series:
        series = dependency_resolver.get_indicator_data(
            volume_inclusion_criteria,
            parameters={"min_volume": min_volume, "rolling_volume_bars": rolling_volume_bars},
        )
        return series.apply(len)


    @indicators.define(dependencies=(tvl_inclusion_criteria,), source=IndicatorSource.dependencies_only_universe)
    def tvl_included_pair_count(
            min_tvl: USDollarAmount,
            dependency_resolver: IndicatorDependencyResolver
    ) -> pd.Series:
        """Calculate number of pairs in meeting volatility criteria on each timestamp"""
        series = dependency_resolver.get_indicator_data(
            tvl_inclusion_criteria,
            parameters={"min_tvl": min_tvl},
        )
        series = series.apply(len)

        # TVL data can be spotty at the beginning when there is only 0 or 1 pairs trading,
        # so we need to fill gaps to 0
        full_index = pd.date_range(
            start=series.index.min(),
            end=series.index.max(),
            freq=Parameters.candle_time_bucket.to_frequency(),
        )
        # Reindex and fill NaN with zeros
        reindexed = series.reindex(full_index, fill_value=0)
        return reindexed


    @indicators.define(dependencies=(inclusion_criteria,), source=IndicatorSource.dependencies_only_universe)
    def all_criteria_included_pair_count(
            min_volume: USDollarAmount,
            min_tvl: USDollarAmount,
            rolling_volume_bars: int,
            dependency_resolver: IndicatorDependencyResolver
    ) -> pd.Series:
        """Series where each timestamp is the list of pairs meeting all inclusion criteria.

        :return:
            Series with pair count for each timestamp
        """
        series = dependency_resolver.get_indicator_data(
            "inclusion_criteria",
            parameters={
                "min_volume": min_volume,
                "min_tvl": min_tvl,
                "rolling_volume_bars": rolling_volume_bars,
            },
        )
        return series.apply(len)


    @indicators.define(source=IndicatorSource.strategy_universe)
    def trading_pair_count(
            strategy_universe: TradingStrategyUniverse,
    ) -> pd.Series:
        """Get number of pairs that trade at each timestamp.

        - Pair must have had at least one candle before the timestamp to be included

        - Exclude benchmarks pairs we do not trade

        :return:
            Series with pair count for each timestamp
        """

        benchmark_pair_ids = {strategy_universe.get_pair_by_human_description(desc).internal_id for desc in SUPPORTING_PAIRS}

        # Get pair_id, timestamp -> timestamp, pair_id index
        series = strategy_universe.data_universe.candles.df["open"]
        swap_index = series.index.swaplevel(0, 1)

        seen_pairs = set()
        seen_data = {}

        for timestamp, pair_id in swap_index:
            if pair_id in benchmark_pair_ids:
                continue
            seen_pairs.add(pair_id)
            seen_data[timestamp] = len(seen_pairs)

        series = pd.Series(seen_data.values(), index=list(seen_data.keys()))
        return series


    @indicators.define(dependencies=(volatility,), source=IndicatorSource.strategy_universe)
    def avg_volatility(
            strategy_universe,
            rolling_volatility_bars: int,
            dependency_resolver: IndicatorDependencyResolver
    ) -> pd.Series:
        """Calculate index avg volatility across all trading pairs.

        :return:
            Series with pair count for each timestamp
        """

        volatility = dependency_resolver.get_indicator_data_pairs_combined(
            "volatility",
            parameters={"rolling_volatility_bars": rolling_volatility_bars},
        )

        n_std = 3

        def remove_outliers_group(group):
            mean = group.mean()
            std = group.std()
            lower_bound = mean - n_std * std
            upper_bound = mean + n_std * std
            return group[(group >= lower_bound) & (group <= upper_bound)]

        cleaned = volatility.groupby(level='timestamp').apply(remove_outliers_group)

        # Group by timestamp, remove outliers within each group, then calculate mean
        cleaned_volatility = cleaned.groupby(level=0).mean()

        return cleaned_volatility


    # Calculate all indicators where parameters have changed and store the result on disk
    indicator_data = calculate_and_load_indicators_inline(
        strategy_universe=strategy_universe,
        create_indicators=indicators.create_indicators,
        parameters=parameters,
        max_workers=1,
    )


    def _decide_trades(
            input: StrategyInput
    ) -> list[TradeExecution]:
        """For each strategy tick, generate the list of trades."""
        parameters = input.parameters
        position_manager = input.get_position_manager()
        state = input.state
        timestamp = input.timestamp
        indicators = input.indicators
        strategy_universe = input.strategy_universe

        # Build signals for each pair
        alpha_model = AlphaModel(
            timestamp,
            close_position_weight_epsilon=parameters.min_portfolio_weight,  # 10 BPS is our min portfolio weight
        )

        # Get pairs included in this rebalance cycle.
        # This includes pair that have been pre-cleared in inclusion_criteria()
        # with volume, volatility and TVL filters
        included_pairs = indicators.get_indicator_value(
            "inclusion_criteria",
            na_conversion=False,
        )
        if included_pairs is None:
            included_pairs = []

        # Set signal for each pair
        signal_count = 0
        for pair_id in included_pairs:
            pair = strategy_universe.get_pair_by_id(pair_id)

            pair_signal = indicators.get_indicator_value("signal", pair=pair)
            if pair_signal is None:
                continue

            weight = pair_signal

            if weight < 0:
                continue

            alpha_model.set_signal(
                pair,
                weight,
            )

            # Diagnostics reporting
            signal_count += 1

        # Calculate how much dollar value we want each individual position to be on this strategy cycle,
        # based on our total available equity
        portfolio = position_manager.get_current_portfolio()
        portfolio_target_value = portfolio.get_total_equity() * parameters.allocation

        # Select max_assets_in_portfolio assets in which we are going to invest
        # Calculate a weight for ecah asset in the portfolio using 1/N method based on the raw signal
        alpha_model.select_top_signals(count=parameters.max_assets_in_portfolio)
        alpha_model.assign_weights(method=weight_passthrouh)
        # alpha_model.assign_weights(method=weight_by_1_slash_n)

        #
        # Normalise weights and cap the positions
        #
        size_risk_model = USDTVLSizeRiskModel(
            pricing_model=input.pricing_model,
            per_position_cap=parameters.per_position_cap_of_pool,  # This is how much % by all pool TVL we can allocate for a position
            missing_tvl_placeholder_usd=parameters.assummed_liquidity_when_data_missings,  # Placeholder for missing TVL data until we get the data off the chain
        )

        alpha_model.normalise_weights(
            investable_equity=portfolio_target_value,
            size_risk_model=size_risk_model,
            max_weight=parameters.max_concentration,
        )

        # Load in old weight for each trading pair signal,
        # so we can calculate the adjustment trade size
        alpha_model.update_old_weights(
            state.portfolio,
            ignore_credit=True,
        )
        alpha_model.calculate_target_positions(position_manager)

        # Shift portfolio from current positions to target positions
        # determined by the alpha signals (momentum)

        # rebalance_threshold_usd = portfolio_target_value * parameters.min_rebalance_trade_threshold_pct
        rebalance_threshold_usd = parameters.individual_rebalance_min_threshold_usd

        assert rebalance_threshold_usd > 0.1, "Safety check tripped - something like wrong with strat code"
        trades = alpha_model.generate_rebalance_trades_and_triggers(
            position_manager,
            min_trade_threshold=rebalance_threshold_usd,  # Don't bother with trades under XXXX USD
            invidiual_rebalance_min_threshold=parameters.individual_rebalance_min_threshold_usd,
            execution_context=input.execution_context,
        )

        # Supply or withdraw cash to Aave if strategy is set to do so
        credit_trades = []
        if parameters.use_aave:
            credit_deposit_flow = position_manager.calculate_credit_flow_needed(
                trades,
                parameters.allocation,
            )
            if abs(credit_deposit_flow) > parameters.credit_flow_dust_threshold:
                credit_trades = position_manager.manage_credit_flow(credit_deposit_flow)
                trades += credit_trades
        else:
            credit_deposit_flow = 0

        return trades  # Return the list of trades we made in this cycle

    result = run_backtest_inline(
        name=parameters.id,
        engine_version="0.5",
        decide_trades=_decide_trades,
        create_indicators=indicators.create_indicators,
        client=client,
        universe=strategy_universe,
        parameters=parameters,
        # log_level=logging.INFO,
        max_workers=1,
        start_at=backtest_start,
        end_at=backtest_end,
    )

    state = result.state

