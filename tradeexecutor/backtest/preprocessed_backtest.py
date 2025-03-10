"""Preprocessed datasets.

- Generate preprocessed backtest histories with certain parameters

- Generated sets are free from survivorship bias, by having inclusion criteria
  as historical TVL threshold

To export / update all exported data:

.. code-block:: shell

    python tradeexecutor/backtest/preprocessed_backtest.py ~/exported

"""
import logging
import os
import sys
from dataclasses import dataclass
import datetime
from pathlib import Path

import pandas as pd

from tradeexecutor.cli.log import setup_logging
from tradeexecutor.strategy.execution_context import python_script_execution_context
from tradeexecutor.strategy.trading_strategy_universe import load_partial_data
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradeexecutor.utils.dedent import dedent_any
from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client
from tradingstrategy.pair import PandasPairUniverse
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.transport.cache import OHLCVCandleType
from tradingstrategy.types import USDollarAmount, Percent
from tradingstrategy.utils.token_extra_data import load_token_metadata
from tradingstrategy.utils.token_filter import filter_pairs_default, filter_by_token_sniffer_score, deduplicate_pairs_by_volume, add_base_quote_address_columns
from tradingstrategy.utils.wrangle import fix_dex_price_data


logger = logging.getLogger(__name__)


@dataclass
class Dataset:
    """Predefined backtesting dataset"""
    slug: str
    name: str
    description: str
    chain: ChainId
    time_bucket: TimeBucket
    start: datetime
    end: datetime
    exchanges: set[str]

    #: Pair descriptions that are always included, regardless of min_tvl and category filtering
    always_included_pairs: list[tuple]

    #: Prefilter pairs with this liquidity before calling token sniffer
    min_tvl: USDollarAmount | None = None
    categories: list[str] | None = None
    max_fee: Percent | None = None
    min_tokensniffer_score: int | None = None


@dataclass
class SavedDataset:
    set: Dataset
    parquet_path: Path
    csv_path: Path
    df: pd.DataFrame
    pairs_df: pd.DataFrame


def make_full_ticker(row: pd.Series) -> str:
    """Generate a base-quote ticker for a pair."""
    return row["base_token_symbol"] + "-" + row["quote_token_symbol"] + "-" + row["exchange_slug"] + "-" + str(row["fee"]) + "bps"


def make_simple_ticker(row: pd.Series) -> str:
    """Generate a ticker for a pair with fee and DEX info."""
    return row["base_token_symbol"] + "-" + row["quote_token_symbol"]


def make_base_symbol(row: pd.Series) -> str:
    """Generate a base symbol."""
    return row["base_token_symbol"]


def make_link(row: pd.Series) -> str:
    """Get TradingStrategy.ai explorer link for the trading data"""
    chain_slug = ChainId(row.chain_id).get_slug()
    return f"https://tradingstrategy.ai/trading-view/{chain_slug}/{row.exchange_slug}/{row.pair_slug}"


def prepare_dataset(
    client: Client,
    dataset: Dataset,
    output_folder: Path,
    write_csv=True,
    write_parquet=True,
) -> SavedDataset:
    """Prepare a predefined backtesting dataset.

    - Download data
    - Clean it
    - Write to a parquet file
    """

    chain_id = dataset.chain
    time_bucket = dataset.time_bucket
    liquidity_time_bucket = TimeBucket.d1  # TVL data for Uniswap v3 is only sampled daily, more fine granular is not needed
    exchange_slugs = dataset.exchanges
    tokensniffer_threshold = dataset.min_tokensniffer_score
    min_liquidity_threshold = dataset.min_tvl  #

    #
    # Set out trading pair universe
    #

    logger.info("Downloading/opening exchange dataset")
    exchange_universe = client.fetch_exchange_universe()

    # Resolve uniswap-v3 internal id
    targeted_exchanges = [exchange_universe.get_by_chain_and_slug(chain_id, slug) for slug in exchange_slugs]
    exchange_ids = [exchange.exchange_id for exchange in targeted_exchanges]
    logger.info(f"Exchange {exchange_slugs} ids are {exchange_ids}")

    # We need pair metadata to know which pairs belong to Polygon
    logger.info("Downloading/opening pairs dataset")
    pairs_df = client.fetch_pair_universe().to_pandas()

    # Never deduplicate supporting pars
    pair_universe = PandasPairUniverse(
        pairs_df,
        exchange_universe=exchange_universe,
        build_index=False,
    )
    supporting_pair_ids = [pair_universe.get_pair_by_human_description(desc).pair_id for desc in dataset.always_included_pairs]
    supporting_pairs_df = pairs_df[pairs_df["pair_id"].isin(supporting_pair_ids)]
    logger.info("We have %d supporting pairs", supporting_pairs_df.shape[0])

    assert min_liquidity_threshold is not None, "Dataset creation only by min_tvl supported for now"

    tvl_df = client.fetch_tvl(
        mode="min_tvl",
        bucket=liquidity_time_bucket,
        start_time=dataset.start,
        end_time=dataset.end,
        exchange_ids=[exc.exchange_id for exc in targeted_exchanges],
        min_tvl=min_liquidity_threshold,
    )
    tvl_filtered_pair_ids = tvl_df["pair_id"].unique()
    logger.info("TVL filter gave us %d pairs", len(tvl_filtered_pair_ids))

    tvl_pairs_df = pairs_df[pairs_df["pair_id"].isin(tvl_filtered_pair_ids)]
    pairs_df = filter_pairs_default(
        tvl_pairs_df,
    )
    logger.info("After standard filters we have %d pairs left", len(tvl_filtered_pair_ids))

    pairs_df = add_base_quote_address_columns(pairs_df)

    pairs_df = load_token_metadata(pairs_df, client)
    # Scam filter using TokenSniffer
    if tokensniffer_threshold is not None:
        risk_filtered_pairs_df = filter_by_token_sniffer_score(
            pairs_df,
            risk_score=tokensniffer_threshold,
        )

    else:
        risk_filtered_pairs_df = pairs_df

    logger.info(
        "After risk filter we have %d pairs",
        len(risk_filtered_pairs_df),
    )

    deduplicated_df = deduplicate_pairs_by_volume(pairs_df)
    pairs_df = pd.concat([deduplicated_df, supporting_pairs_df]).drop_duplicates(subset='pair_id', keep='first')
    logger.info("After pairs deduplication we have %d pairs", len(pairs_df))

    universe_options = UniverseOptions(
        start_at=dataset.start,
        end_at=dataset.end,
    )

    # After we know pair ids that fill the liquidity criteria,
    # we can build OHLCV dataset for these pairs
    logger.info(f"Downloading/opening OHLCV dataset {time_bucket}")
    loaded_data = load_partial_data(
        client=client,
        time_bucket=time_bucket,
        pairs=pairs_df,
        execution_context=python_script_execution_context,
        universe_options=universe_options,
        liquidity=False,
        liquidity_time_bucket=TimeBucket.d1,
        preloaded_tvl_df=tvl_df,
    )
    logger.info("Wrangling DEX price data")
    price_df = loaded_data.candles
    price_df = price_df.set_index("timestamp", drop=False).groupby("pair_id")
    price_dfgb = fix_dex_price_data(
        price_df,
        freq=time_bucket.to_frequency(),
        forward_fill=True,
        forward_fill_until=dataset.end,
    )
    price_df = price_dfgb.obj

    # Add additional columns
    pairs_df = pairs_df.set_index("pair_id")
    pair_metadata = {pair_id: row for pair_id, row in pairs_df.iterrows()}
    price_df["ticker"] = price_df["pair_id"].apply(lambda pair_id: make_full_ticker(pair_metadata[pair_id]))
    price_df["link"] = price_df["pair_id"].apply(lambda pair_id: make_link(pair_metadata[pair_id]))
    price_df["base"] = price_df["pair_id"].apply(lambda pair_id: pair_metadata[pair_id]["base_token_symbol"])
    price_df["quote"] = price_df["pair_id"].apply(lambda pair_id: pair_metadata[pair_id]["quote_token_symbol"])
    price_df["fee"] = price_df["pair_id"].apply(lambda pair_id: pair_metadata[pair_id]["fee"])

    # Merge price and TVL data.
    # For this we need to resample TVL to whatever timeframe the price happens to be in.
    liquidity_df = tvl_df
    liquidity_df = liquidity_df.rename(columns={'bucket': 'timestamp'})
    liquidity_df = liquidity_df.groupby('pair_id').apply(lambda x: x.set_index("timestamp").resample(time_bucket.to_frequency()).ffill(), include_groups=False)
    liquidity_df = liquidity_df.rename(columns={'close': 'tvl'})

    merged_df = price_df.join(liquidity_df["tvl"].to_frame(), how='inner')

    unique_pair_ids = merged_df.index.get_level_values('pair_id').unique()
    logger.info(f"After price/TVL merge we have {len(unique_pair_ids)} unique pairs")

    # Export data, make sure we got columns in an order we want
    logger.info(f"Writing OHLCV files")
    del merged_df["timestamp"]
    del merged_df["pair_id"]
    merged_df = merged_df.reset_index()
    column_order = (
        "ticker",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "tvl",
        "base",
        "quote",
        "fee",
        "link",
        "pair_id",
    )
    merged_df = merged_df.reindex(columns=column_order)  # Sort columns in a specific order

    if write_csv:
        csv_file = output_folder / f"{dataset.slug}.csv"
        merged_df.to_csv(
            csv_file,
        )
        logger.info(f"Wrote {csv_file}, {csv_file.stat().st_size:,} bytes")
    else:
        csv_file = None

    if write_parquet:
        parquet_file = output_folder / f"{dataset.slug}.parquet"
        merged_df.to_csv(
            parquet_file,
        )
        logger.info(f"Wrote {csv_file}, {csv_file.stat().st_size:,} bytes")
    else:
        parquet_file = None

    return SavedDataset(
        set=dataset,
        csv_path=csv_file,
        parquet_path=parquet_file,
        df=merged_df,
        pairs_df=pairs_df,
    )


PREPACKAGED_SETS = [
    Dataset(
        chain=ChainId.binance,
        description=dedent_any("""
        PancakeSwap DEX daily trades.
        
        - Contains bull and bear market data with mixed set of tokens
        - Binance smart chain is home of many fly-by-night tokens, 
          and very few of tokens on this chain have long term prospects 
        """),
        slug="binance-chain-1d",
        name="Binance Chain, Pancakeswap, 2021-2025, daily",
        start=datetime.datetime(2021, 1, 1),
        end=datetime.datetime(2025, 1, 1),
        min_tvl=5_000_000,
        time_bucket=TimeBucket.d1,
        exchanges={"pancakeswap-v2"},
        always_included_pairs=[
            (ChainId.binance, "pancakeswap-v2", "WBNB", "USDT"),
        ]
    ),

    Dataset(
        chain=ChainId.binance,
        slug="binance-chain-1h",
        name="Binance Chain, Pancakeswap, 2021-2025, hourly",
        description=dedent_any("""
        PancakeSwap DEX hourly trades.
        
        - Contains bull and bear market data with mixed set of tokens
        - Binance smart chain is home of many fly-by-night tokens, 
          and very few of tokens on this chain have long term prospects 
        """),
        start=datetime.datetime(2021, 1, 1),
        end=datetime.datetime(2025, 1, 1),
        time_bucket=TimeBucket.h1,
        min_tvl=5_000_000,
        exchanges={"pancakeswap-v2"},
        always_included_pairs=[
            (ChainId.binance, "pancakeswap-v2", "WBNB", "USDT"),
        ]
    )
]


def export_all_main():
    """Export all preprocessed backtest sets.

    - Main entry point
    """

    setup_logging()

    client = Client.create_live_client(api_key=os.environ["TRADING_STRATEGY_API_KEY"])
    output_path = Path(sys.argv[1])

    assert output_path.exists(), f"{output_path} does not exist"
    assert output_path.is_dir(), f"{output_path} is not a directory"
    for ds in PREPACKAGED_SETS:
        prepare_dataset(
            client=client,
            dataset=ds,
            output_folder=output_path,
        )

    logger.info("All done")


if __name__ == "__main__":
    export_all_main()