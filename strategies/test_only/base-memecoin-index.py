"""Base memecoin basket strategy.

Used in Velvet and Lagoon E2E test.
"""
import datetime

import numpy as np
import pandas as pd

from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.types import USDollarAmount
from tradeexecutor.strategy.alpha_model import AlphaModel
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorDependencyResolver
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet, IndicatorSource
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.tag import StrategyTag
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.strategy.trading_strategy_universe import (
    load_partial_data)
from tradeexecutor.strategy.tvl_size_risk import USDTVLSizeRiskModel
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradeexecutor.strategy.weighting import weight_by_1_slash_n, weight_passthrouh
from tradeexecutor.webhook.error import exception_response
from tradingstrategy.alternative_data.coingecko import CoingeckoUniverse, categorise_pairs
from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client
from tradingstrategy.pair import PandasPairUniverse
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.transport.cache import OHLCVCandleType
from tradingstrategy.utils.token_extra_data import filter_scams
from tradingstrategy.utils.token_filter import deduplicate_pairs_by_volume
from tradeexecutor.state.visualisation import PlotKind
from tradingstrategy.types import TokenSymbol
from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.utils.dedent import dedent_any


#
# Strategy parametrers
#


trading_strategy_engine_version = "0.5"


class Parameters:
    id = "02-base-refined"

    # We trade 1h candle
    candle_time_bucket = TimeBucket.h1
    cycle_duration = CycleDuration.cycle_4h

    # Coingecko categories to include
    #
    # See list here: TODO
    #
    chain_id = ChainId.base
    categories = {"Meme"}
    exchanges = {"uniswap-v2", "uniswap-v3"}

    #
    # Basket construction and rebalance parameters
    #
    min_asset_universe = 10  # How many assets we need in the asset universe to start running the index
    max_assets_in_portfolio = 99  # How many assets our basket can hold once
    allocation = 0.99  # Allocate all cash to volatile pairs
    min_rebalance_trade_threshold_pct = 0.05  # % of portfolio composition must change before triggering rebalacne
    individual_rebalance_min_threshold_usd = 10  # Don't make buys less than this amount
    min_volatility_threshold = 0.02  # Set to have Sharpe ratio threshold for the inclusion
    per_position_cap_of_pool = 0.01  # Never own more than % of the lit liquidity of the trading pool
    max_concentration = 0.50  # How large % can one asset be in a portfolio once

    #
    # Inclusion criteria parameters:
    # - We set the length of various indicators used in the inclusion criteria
    # - We set minimum thresholds neede to be included in the index to filter out illiquid pairs
    #

    # For the length of trailing sharpe used in inclusion criteria
    trailing_sharpe_bars = pd.Timedelta("14d") // candle_time_bucket.to_timedelta()  # How many bars to use in trailing sharpe indicator
    rebalance_volalitity_bars = pd.Timedelta("14d") // candle_time_bucket.to_timedelta()  # How many bars to use in volatility indicator
    rolling_volume_bars = pd.Timedelta("7d") // candle_time_bucket.to_timedelta()
    rolling_liquidity_bars = pd.Timedelta("7d") // candle_time_bucket.to_timedelta()
    ewm_span = 200  # How many bars to use in exponential moving average for trailing sharpe smoothing
    min_volume = 200_000  # USD
    min_liquidity = 200_000  # USD
    min_token_sniffer_score = 30  # Scam filter

    #
    # Backtesting only
    #
    backtest_start = datetime.datetime(2024, 9, 15)
    backtest_end = datetime.datetime(2024, 11, 10)
    initial_cash = 10_000

    #
    # Live only
    #
    routing = TradeRouting.default
    required_history_period = datetime.timedelta(days=2 * 14 + 1)
    slippage_tolerance = 0.0060  # 0.6%
    assummed_liquidity_when_data_missings = 10_000


#
# Trading universe
#

#: Assets used in routing and buy-and-hold benchmark values for our strategy, but not traded by this strategy.
SUPPORTING_PAIRS = [
    (ChainId.base, "uniswap-v3", "WETH", "USDC", 0.0005),  # TODO: Needed until we have universal routing
    (ChainId.base, "uniswap-v2", "WETH", "USDC", 0.0030),  # TODO: Needed until we have universal routing
    # (ChainId.ethereum, "uniswap-v2", "WETH", "USDC", 0.0030),  # TODO: Needed until we have universal routing
]

#: Which pair we use as the volatility benchmark for the inclusion criteria
VOLATILITY_BENCHMARK_PAIR = (ChainId.base, "uniswap-v2", "WETH", "USDC")


def create_trading_universe(
    timestamp: datetime.datetime,
    client: Client,
    execution_context: ExecutionContext,
    universe_options: UniverseOptions,
) -> TradingStrategyUniverse:
    """Create the trading universe.

    - Load Trading Strategy full pairs dataset

    - Load built-in Coingecko top 1000 dataset

    - Get all DEX tokens for a certain Coigecko category

    - Load OHCLV data for these pairs

    - Load also BTC and ETH price data to be used as a benchmark
    """

    chain_id = Parameters.chain_id
    categories = Parameters.categories

    coingecko_universe = CoingeckoUniverse.load()
    print("Coingecko universe is", coingecko_universe)

    exchange_universe = client.fetch_exchange_universe()
    pairs_df = client.fetch_pair_universe().to_pandas()

    # Drop other chains to make the dataset smaller to work with
    chain_mask = pairs_df["chain_id"] == Parameters.chain_id.value
    pairs_df = pairs_df[chain_mask]

    # Pull out our benchmark pairs ids.
    # We need to construct pair universe object for the symbolic lookup.
    pair_universe = PandasPairUniverse(pairs_df, exchange_universe=exchange_universe)
    benchmark_pair_ids = [pair_universe.get_pair_by_human_description(desc).pair_id for desc in SUPPORTING_PAIRS]

    # Assign categories to all pairs
    category_df = categorise_pairs(coingecko_universe, pairs_df)

    # Get all trading pairs that are memecoin, across all coingecko data
    mask = category_df["category"].isin(categories)
    category_pair_ids = category_df[mask]["pair_id"]

    our_pair_ids = list(category_pair_ids) + benchmark_pair_ids

    # From these pair ids, see what trading pairs we have on Ethereum mainnet
    pairs_df = pairs_df[pairs_df["pair_id"].isin(our_pair_ids)]

    # Limit by DEX
    pairs_df = pairs_df[pairs_df["exchange_slug"].isin(Parameters.exchanges)]

    # Never deduplicate supporrting pars
    supporting_pairs_df = pairs_df[pairs_df["pair_id"].isin(benchmark_pair_ids)]

    # Deduplicate trading pairs - Choose the best pair with the best volume
    deduplicated_df = deduplicate_pairs_by_volume(pairs_df)
    pairs_df = pd.concat([deduplicated_df, supporting_pairs_df]).drop_duplicates(subset='pair_id', keep='first')

    print(
        f"Total {len(pairs_df)} pairs to trade on {chain_id.name} for categories {categories}",
    )

    # Scam filter using TokenSniffer
    pairs_df = filter_scams(pairs_df, client, min_token_sniffer_score=Parameters.min_token_sniffer_score)
    pairs_df = pairs_df.sort_values("volume", ascending=False)

    uni_v2 = pairs_df.loc[pairs_df["exchange_slug"] == "uniswap-v2"]
    uni_v3 = pairs_df.loc[pairs_df["exchange_slug"] == "uniswap-v3"]
    print(f"Pairs on Uniswap v2: {len(uni_v2)}, Uniswap v3: {len(uni_v3)}")

    dataset = load_partial_data(
        client=client,
        time_bucket=Parameters.candle_time_bucket,
        pairs=pairs_df,
        execution_context=execution_context,
        universe_options=universe_options,
        liquidity=True,
        liquidity_time_bucket=TimeBucket.d1,
        liquidity_query_type=OHLCVCandleType.tvl_v2,
    )

    strategy_universe = TradingStrategyUniverse.create_from_dataset(
        dataset,
        reserve_asset="0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",  # USDC on Base
        forward_fill=True,  # We got very gappy data from low liquid DEX coins
    )

    # Tag benchmark/routing pairs tokens so they can be separated from the rest of the tokens
    # for the index construction.
    strategy_universe.warm_up_data()
    for pair_id in benchmark_pair_ids:
        pair = strategy_universe.get_pair_by_id(pair_id)
        pair.other_data["benchmark"] = True

    print("Universe is (including benchmark pairs):")
    for idx, pair in enumerate(strategy_universe.iterate_pairs()):
        benchmark = pair.other_data.get("benchmark")
        print(f"   {idx + 1}. pair #{pair.internal_id}: {pair.base.token_symbol} - {pair.quote.token_symbol} ({pair.exchange_name}), {'benchmark/routed token' if benchmark else 'traded token'}")

    return strategy_universe


#
# Strategy logic
#

def decide_trades(
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
    alpha_model = AlphaModel(timestamp)

    volume_inclusion_criteria_pairs = indicators.get_indicator_value(
        "volume_inclusion_criteria",
        na_conversion=False,
    )

    if volume_inclusion_criteria_pairs is None:
        volume_inclusion_criteria_pairs = []

    vol_pair = strategy_universe.get_pair_by_human_description(VOLATILITY_BENCHMARK_PAIR)
    vol_pair_vol = indicators.get_indicator_value("volatility_ewm", pair=vol_pair)

    if vol_pair_vol is None:
        # We do not have a benchmark volatility yet
        return []

    max_vol = (0, None)  # Diagnostics
    signal_count = 0

    for pair_id in volume_inclusion_criteria_pairs:

        pair = strategy_universe.get_pair_by_id(pair_id)
        volatility = indicators.get_indicator_value("volatility_ewm", pair=pair)
        if not volatility:
            continue

        if volatility < vol_pair_vol:
            # Volatility must be higher than the volatility benchmark pair (ETH, BTC)
            continue

        weight = 1 / volatility

        alpha_model.set_signal(
            pair,
            weight,
        )

        # Diagnostics reporting
        signal_count += 1
        if volatility > max_vol[0]:
            max_vol = (volatility, pair)

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
    rebalance_threshold_usd = portfolio_target_value * parameters.min_rebalance_trade_threshold_pct
    assert rebalance_threshold_usd > 0.1, "Safety check tripped - something like wrong with strat code"
    trades = alpha_model.generate_rebalance_trades_and_triggers(
        position_manager,
        min_trade_threshold=rebalance_threshold_usd,  # Don't bother with trades under XXXX USD
        individual_rebalance_min_threshold=parameters.individual_rebalance_min_threshold_usd,
    )

    # Add verbal report about decision made/not made,
    # so it is much easier to diagnose live trade execution.
    # This will be readable in Discord/Telegram logging.
    if input.is_visualisation_enabled():
        try:
            top_signal = next(iter(alpha_model.get_signals_sorted_by_weight()))
            if top_signal.normalised_weight == 0:
                top_signal = None
        except StopIteration:
            top_signal = None

        max_vol_pair = max_vol[1]
        if max_vol_pair:
            max_vol_signal = alpha_model.get_signal_by_pair(max_vol_pair)
            assert max_vol_signal.position_size_risk
        else:
            max_vol_signal = None

        report = dedent_any(f"""
        Rebalanced: {'👍' if alpha_model.is_rebalance_triggered() else '👎'}
        Max position value change: {alpha_model.max_position_adjust_usd:,.2f} USD
        Rebalance threshold: {alpha_model.position_adjust_threshold_usd}:,.2f USD
        Trades decided: {len(trades)}
        Pairs total: {strategy_universe.data_universe.pairs.get_count()}
        Pairs meeting volume inclusion criteria: {len(volume_inclusion_criteria_pairs)}
        Signals created: {signal_count}
        Total equity: {portfolio.get_total_equity():,.2f} USD
        Cash: {position_manager.get_current_cash():,.2f} USD
        Investable equity: {alpha_model.investable_equity:,.2f} USD
        Accepted investable equity: {alpha_model.accepted_investable_equity:,.2f} USD
        Allocated to signals: {alpha_model.get_allocated_value():,.2f} USD
        Discarted allocation because of lack of lit liquidity: {alpha_model.size_risk_discarded_value:,.2f} USD
        {vol_pair.base.token_symbol} volatility: {vol_pair_vol}        
        Most volatility pair: {max_vol_pair.get_ticker() if max_vol_pair else '-'}
        Most volatility pair vol: {max_vol[0]}
        Most volatility pair signal value: {max_vol_signal.signal if max_vol_signal else '-'}
        Most volatility pair signal weight: {max_vol_signal.raw_weight if max_vol_signal else '-'}
        Most volatility pair signal weight (normalised): {max_vol_signal.normalised_weight * 100 if max_vol_signal else '-'} % (got {max_vol_signal.position_size_risk.get_relative_capped_amount() * 100 if max_vol_signal else '-'} % of asked size)
        """)

        if top_signal:
            top_signal_vol = indicators.get_indicator_value("volatility_ewm", pair=top_signal.pair)
            assert top_signal.position_size_risk
            report += dedent_any(f"""
            Top signal pair: {top_signal.pair.get_ticker()}
            Top signal volatility: {top_signal_vol}
            Top signal value: {top_signal.signal}
            Top signal weight: {top_signal.raw_weight}
            Top signal weight (normalised): {top_signal.normalised_weight * 100:.2f} % (got {top_signal.position_size_risk.get_relative_capped_amount() * 100:.2f} % of asked size)
            """)

        for flag, count in alpha_model.get_flag_diagnostics_data().items():
            report += f"Signals with flag {flag.name}: {count}\n"

        state.visualisation.add_message(
            timestamp,
            report,
        )

    return trades  # Return the list of trades we made in this cycle


#
# Indicators
#

def trailing_sharpe(
    close: pd.Series,
    window_length_bars: int
) -> pd.Series:
    """Calculate trailing 30d or so returns / standard deviation.

    :param length:
        Trailing period.

    :return:
        Rolling cumulative returns / rolling standard deviation

        Note that this trailing sharpe is not annualised.
    """
    ann_factor = pd.Timedelta(days=365) / Parameters.candle_time_bucket.to_pandas_timedelta()
    returns = close.pct_change()
    mean_returns = returns.rolling(window=window_length_bars).mean()
    vol = returns.rolling(window=window_length_bars).std()
    return mean_returns / vol * np.sqrt(ann_factor)


def trailing_sharpe_ewm(
    close: pd.Series,
    window_length_bars: int,
    ewm_span: float,
    pair: TradingPairIdentifier,
    dependency_resolver: IndicatorDependencyResolver,
) -> pd.Series:
    """Expontentially weighted moving average for Sharpe.

    :param ewm_span:
        How many bars to consider in the EVM

    """
    trailing_sharpe = dependency_resolver.get_indicator_data(
        "trailing_sharpe",
        pair=pair,
        parameters={"window_length_bars": window_length_bars},
    )
    ewm = trailing_sharpe.ewm(span=ewm_span)
    return ewm.mean()


def volatility(close: pd.Series, window_length_bars: int) -> pd.Series:
    """Calculate the rolling volatility for rebalancing the index for each decision cycle."""
    price_diff = close.pct_change()
    rolling_std = price_diff.rolling(window=window_length_bars).std()
    return rolling_std


def volatility_ewm(close: pd.Series, window_length_bars: int) -> pd.Series:
    """Calculate the rolling volatility for rebalancing the index for each decision cycle."""
    # We are operating on 1h candles, 14d window
    price_diff = close.pct_change()
    rolling_std = price_diff.rolling(window=window_length_bars).std()
    ewm = rolling_std.ewm(span=14*8)
    return ewm.mean()


def mean_returns(close: pd.Series, window_length_bars: int) -> pd.Series:
    # Descripton: TODO
    returns = close.pct_change()
    mean_returns = returns.rolling(window=window_length_bars).mean()
    return mean_returns


def rolling_cumulative_volume(volume: pd.Series, window_length_bars: int) -> pd.Series:
    """Calculate rolling volume of the pair.

    - Used in inclusion criteria
    """
    rolling_volume = volume.rolling(window=window_length_bars).sum()
    return rolling_volume


def rolling_liquidity_avg(close: pd.Series, window_length_bars: int) -> pd.Series:
    """Calculate rolling liquidity average

    - This is either TVL or XY liquidity (one sided) depending on the trading pair DEX type

    - Used in inclusion criteria
    """
    rolling_liquidity_close = close.rolling(window=window_length_bars).mean()
    return  rolling_liquidity_close


def volume_inclusion_criteria(
    strategy_universe: TradingStrategyUniverse,
    min_volume: USDollarAmount,
    window_length_bars: int,
    dependency_resolver: IndicatorDependencyResolver,
) -> pd.Series:
    """Calculate pair volume inclusion criteria.

    - Avoid including illiquid / broken pairs in the set: Pair is included when it has enough volume

    TODO: Add liquidity check later

    :return:
        Series where each timestamp is a list of pair ids meeting the criteria at that timestamp
    """

    benchmark_pair_ids = [strategy_universe.get_pair_by_human_description(desc).internal_id for desc in SUPPORTING_PAIRS]

    series = dependency_resolver.get_indicator_data_pairs_combined(
        "rolling_cumulative_volume",
        parameters={"window_length_bars": window_length_bars},
    )

    # Benchmark pairs are never traded
    filtered_series = series[~series.index.get_level_values('pair_id').isin(benchmark_pair_ids)]

    # Get mask for days when the rolling volume meets out criteria
    mask = filtered_series >= min_volume

    # Turn to a series of lists
    data = mask.groupby(level='timestamp').apply(lambda x: x.index.get_level_values('pair_id').tolist())

    return data


def included_pair_count(
    strategy_universe: TradingStrategyUniverse,
    min_volume: USDollarAmount,
    window_length_bars: int,
    dependency_resolver: IndicatorDependencyResolver
) -> pd.Series:
    """Included pairs is a combination of available pairs and inclusion criteria.

    - At a given moment of time, which of all available pairs are tradeable

    :return:
        Series with pair count for each timestamp
    """
    series = dependency_resolver.get_indicator_data(
        "volume_inclusion_criteria",
        parameters={"min_volume": min_volume, "window_length_bars": window_length_bars},
    )
    return series.apply(len)


def create_indicators(
    timestamp: datetime.datetime | None,
    parameters: StrategyParameters,
    strategy_universe: TradingStrategyUniverse,
    execution_context: ExecutionContext
):
    """Create indicator descriptions.

    - Indicators are automatically recalculated and cached by the backtest runner,
      if indicator Python function body or parameters change
    """
    indicator_set = IndicatorSet()
    indicator_set.add(
        "trailing_sharpe",
        trailing_sharpe,
        {"window_length_bars": parameters.trailing_sharpe_bars},
        IndicatorSource.close_price,
    )

    indicator_set.add(
        "trailing_sharpe_ewm",
        trailing_sharpe_ewm,
        {"window_length_bars": parameters.trailing_sharpe_bars, "ewm_span": parameters.ewm_span},
        IndicatorSource.close_price,
        order=2,
    )

    indicator_set.add(
        "volatility",
        volatility,
        {"window_length_bars": parameters.rebalance_volalitity_bars},
        IndicatorSource.close_price,
    )


    indicator_set.add(
        "volatility_ewm",
        volatility_ewm,
        {"window_length_bars": parameters.rebalance_volalitity_bars},
        IndicatorSource.close_price,
    )


    indicator_set.add(
        "mean_returns",
        mean_returns,
        {"window_length_bars": parameters.rebalance_volalitity_bars},
        IndicatorSource.close_price,
    )

    indicator_set.add(
        "rolling_cumulative_volume",
        rolling_cumulative_volume,
        {"window_length_bars": parameters.rolling_volume_bars},
        IndicatorSource.ohlcv,
    )
    # TODO: Currently web API issues loading this data for certain pairs
    #
    # indicator_set.add(
    #     "rolling_liquidity_avg",
    #     rolling_liquidity_avg,
    #     {"window_length_bars": parameters.rolling_liquidity_bars,
    #     IndicatorSource.liquidity,
    # )

    indicator_set.add(
        "volume_inclusion_criteria",
        volume_inclusion_criteria,
        {"min_volume": parameters.min_volume, "window_length_bars": parameters.rolling_volume_bars},
        IndicatorSource.strategy_universe,
        order=3,
    )

    indicator_set.add(
        "included_pair_count",
        included_pair_count,
        {"min_volume": parameters.min_volume, "window_length_bars": parameters.rolling_volume_bars},
        IndicatorSource.strategy_universe,
        order=4,
    )

    return indicator_set


#
# Strategy metadata/UI data.
#


tags = {StrategyTag.beta}

name = "Memecoin index"

short_description = "Portfolio strategy for memecoins"

icon = ""

long_description = """
# Strategy description

The strategy employs a passive management – or indexing – investment approach and seeks to track the performance of the [CoinGecko memecoin category](https://www.coingecko.com/en/categories/meme-token?asset_platform_id=ethereum), scaled for volatility, filtered by Ethereum mainnet.

- The Index is comprised of liquid memecoins listed on Base mainnet.
- The strategy is only available to new deposits if the Liquidity Constraints are met and the pool is 100% invested.
- The weighting of the Index constituents is scaled for volatility.
- Performance will be denominated in USDC.

## Strategy logic

The strategy attempts to:
    1. Track the performance of the Index by investing in all Index Constituents subject to Liquidity Constraints.
    2. Scale the weightings of the Index Constituents by volatility (Volatility Scaling) to avoid concentration risk.
    3. Remain fully invested subject to Liquidity Constraints. A Cash position will emerge if liquidity constraints require < 100% investment.
    4. Any cash position will be invested in Ethena's USDe.
    5. Rebalance every 4 hours.
    
## Index eligibility criteria

Liquidity Contraints:
    1. Portfolio Concentration: A single asset can never be > 50% of the portfolio. Practically this means that there must always be >=2 assets in the portfolio.
    2. Single Asset Limits: The held value of any Index Constituent can never be > 50% of the pool in which it is traded. The Index will renormalise weights to ensure this constraint is met and a Cash Position will emerge.
    3. Index Inclusion: Assets will be included as an Index Constituent if they meet the following criteria:
        a. Volume Filtering: An Index Constituent must have total volume traded in the last 7 days >= $USDC 200k.
        b. Volatility Filtering: An Index Constituent must have volatility >= the volatility of Bitcoin (`Volatility(BTC/USDC, t)`) in addition to being greater than 10% annualised volatilty.

## Index weighting
        
Volatility Scaling of the Weights:
    - The weights of the Index Constituents are scaled for volatility.
    - The Volatility of each Index Constituent (`i`) at the current time (`t`) is calculated using the previous Volatility Estimator `Volatility(i, t) = VolatilityEstimator(i, t - 1) * sqrt(365 * DailyObservations)`. This is annualised for ease of analysis.
    - Equal Weights of each Index Constituent are calculated by `EqualWeight(i) = 1 / Number of Index Constituents` at the time of calculation.
    - The Weights of each Index Constituent (`Weights(i, t)`) are then scaled by the inverse of the volatility: `Weights(i, t) =  EqualWeight(i) / Volatility(i, t)`.

Volatility Estimator:
    - The Volatility Estimator of each Index Constituent, i, is calculated using an exponentially weighted moving average (EWMA) of the 1 hourly returns (such that `DailyObservations = 24`).
    - The Returns for each Index Constituent are calculated as `Return(i, t) = IndexConstituentPrice(i, t) / IndexConstituentPrice(i, t-1) - 1`.
    - The half-life of the EWMA is fixed at 14 days by setting `Alpha = 2 / (14 * DailyObservations + 1)`.
    - The Volatility Estimator is calculated at each observation time (`t`), of which there are `DailyObservations` in a day. At each observation time `t`, the Volatility Estimator is updated for each Index Constituent `i` as `VolatilityEstimator(i, t) = Alpha * Return(i, t) + (1 - Alpha) * VolatilityEstimator(i, t - 1)`.

## Further information

- Any questions are welcome in [the Discord community chat](https://tradingstrategy.ai/community)

"""
