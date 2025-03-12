"""Create trading universe using min_tvl filter.

- Mostly lifted strategy code to capture the test case
"""
import datetime

import pandas as pd

from eth_defi.token import WRAPPED_NATIVE_TOKEN, USDC_NATIVE_TOKEN
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.execution_context import unit_test_execution_context, ExecutionContext
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, load_partial_data
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client
from tradingstrategy.lending import LendingProtocolType
from tradingstrategy.pair import PandasPairUniverse
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.utils.token_extra_data import load_token_metadata
from tradingstrategy.utils.token_filter import add_base_quote_address_columns, filter_for_stablecoins, StablecoinFilteringMode, filter_for_derivatives, filter_by_token_sniffer_score, filter_for_exchange_slugs, filter_for_quote_tokens, \
    deduplicate_pairs_by_volume


class Parameters:
    id = "base-ath"

    # We trade 1h candle
    candle_time_bucket = TimeBucket.h1
    cycle_duration = CycleDuration.cycle_4h

    # Coingecko categories to include
    # s
    # See list here: TODO
    #
    chain_id = ChainId.base
    exchanges = {"uniswap-v2", "uniswap-v3"}

    min_tvl_prefilter = 1_500_000  # USD - to reduce number of trading pairs for backtest-purposes only
    min_tvl = 1_500_000  # USD - set to same as above if you want to avoid any survivorship bias
    min_token_sniffer_score = 50

    #
    #
    # Backtesting only
    # Limiting factor: Aave v3 on Base starts at the end of DEC 2023
    #
    backtest_start = datetime.datetime(2024, 1, 1)
    backtest_end = datetime.datetime(2024, 2, 4)



#: Assets used in routing and buy-and-hold benchmark values for our strategy, but not traded by this strategy.
SUPPORTING_PAIRS = [
    (ChainId.base, "uniswap-v2", "WETH", "USDC", 0.0030),
    (ChainId.base, "uniswap-v3", "WETH", "USDC", 0.0005),
    (ChainId.base, "uniswap-v3", "cbBTC", "WETH", 0.0030),    # Only trading since October
]

#: Needed for USDC credit
LENDING_RESERVES = [
    (Parameters.chain_id, LendingProtocolType.aave_v3, "USDC"),
]

PREFERRED_STABLECOIN = USDC_NATIVE_TOKEN[Parameters.chain_id.value].lower()

VOL_PAIR = (ChainId.base, "uniswap-v3", "WETH", "USDC", 0.0005)


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

    exchange_universe = client.fetch_exchange_universe()
    targeted_exchanges = [exchange_universe.get_by_chain_and_slug(ChainId.base, slug) for slug in Parameters.exchanges]

    # Pull out our benchmark pairs ids.
    # We need to construct pair universe object for the symbolic lookup.
    # TODO: PandasPairUniverse(buidl_index=True) - speed this up by skipping index building
    all_pairs_df = client.fetch_pair_universe().to_pandas()
    all_pairs_df = filter_for_exchange_slugs(all_pairs_df, Parameters.exchanges)
    pair_universe = PandasPairUniverse(
        all_pairs_df,
        exchange_universe=exchange_universe,
        build_index=False,
    )

    #
    # Do exchange and TVL prefilter pass for the trading universe
    #
    tvl_df = client.fetch_tvl(
        mode="min_tvl",
        bucket=TimeBucket.d1,
        start_time=Parameters.backtest_start,
        end_time=Parameters.backtest_end,
        exchange_ids=[exc.exchange_id for exc in targeted_exchanges],
        min_tvl=Parameters.min_tvl_prefilter,
    )

    tvl_filtered_pair_ids = tvl_df["pair_id"].unique()
    benchmark_pair_ids = [pair_universe.get_pair_by_human_description(desc).pair_id for desc in SUPPORTING_PAIRS]
    needed_pair_ids = set(benchmark_pair_ids) | set(tvl_filtered_pair_ids)
    pairs_df = all_pairs_df[all_pairs_df["pair_id"].isin(needed_pair_ids)]

    category_df = pairs_df
    category_df = add_base_quote_address_columns(category_df)
    category_df = filter_for_stablecoins(category_df, StablecoinFilteringMode.only_volatile_pairs)
    category_df = filter_for_derivatives(category_df)

    allowed_quotes = {
        PREFERRED_STABLECOIN,
        WRAPPED_NATIVE_TOKEN[chain_id.value].lower(),
    }

    category_df = filter_for_quote_tokens(category_df, allowed_quotes)
    category_pair_ids = category_df["pair_id"]
    our_pair_ids = list(category_pair_ids) + benchmark_pair_ids
    pairs_df = category_df[category_df["pair_id"].isin(our_pair_ids)]

    # Never deduplicate supporting pars
    supporting_pairs_df = pairs_df[pairs_df["pair_id"].isin(benchmark_pair_ids)]

    # Deduplicate trading pairs - Choose the best pair with the best volume
    deduplicated_df = deduplicate_pairs_by_volume(pairs_df)
    pairs_df = pd.concat([deduplicated_df, supporting_pairs_df]).drop_duplicates(subset='pair_id', keep='first')

    # Add benchmark pairs back to the dataset
    pairs_df = pd.concat([pairs_df, supporting_pairs_df]).drop_duplicates(subset='pair_id', keep='first')

    # Load metadata
    pairs_df = load_token_metadata(pairs_df, client)

    # Scam filter using TokenSniffer
    risk_filtered_pairs_df = filter_by_token_sniffer_score(
        pairs_df,
        risk_score=Parameters.min_token_sniffer_score,
    )

    # Check if we accidentally get rid of benchmark pairs we need for the strategy
    difference = set(benchmark_pair_ids).difference(set(risk_filtered_pairs_df["pair_id"]))
    if difference:
        first_dropped_id = next(iter(difference))
        first_dropped_data = pairs_df.loc[pairs_df.pair_id == first_dropped_id]
        assert len(first_dropped_data) == 1, f"Got {len(first_dropped_data)} entries: {first_dropped_data}"
        raise AssertionError(f"Benchmark trading pair dropped in filter_by_token_sniffer_score() check: {first_dropped_data.iloc[0]}")
    pairs_df = risk_filtered_pairs_df.sort_values("volume", ascending=False)

    dataset = load_partial_data(
        client=client,
        time_bucket=Parameters.candle_time_bucket,
        pairs=pairs_df,
        execution_context=execution_context,
        universe_options=universe_options,
        lending_reserves=LENDING_RESERVES,
        preloaded_tvl_df=tvl_df,
        liquidity_time_bucket=TimeBucket.d1,
    )

    reserve_asset = PREFERRED_STABLECOIN

    strategy_universe = TradingStrategyUniverse.create_from_dataset(
        dataset,
        reserve_asset=reserve_asset,
        forward_fill=True,  # We got very gappy data from low liquid DEX coins
    )

    return strategy_universe


def test_min_tvl_trading_universe(
    persistent_test_client: Client,
):
    """Create trading universe using fetch_tvl(min_tvl=...) filter."""
    client = persistent_test_client

    universe = create_trading_universe(
        None,
        client=client,
        execution_context=unit_test_execution_context,
        universe_options=UniverseOptions.from_strategy_parameters_class(Parameters, unit_test_execution_context)
    )

    # We have liquidity data correctly loaded
    pair = universe.get_pair_by_human_description(
        (ChainId.base, "uniswap-v3", "WETH", "USDC", 0.0005)
    )

    liquidity, _  = universe.data_universe.liquidity.get_liquidity_with_tolerance(
        pair_id=pair.internal_id,
        when=pd.Timestamp("2024-01-05"),
        tolerance=pd.Timedelta(days=1),
    )
    assert liquidity > 100_000

