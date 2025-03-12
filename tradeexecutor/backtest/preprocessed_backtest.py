"""Preprocessed datasets.

- Generate preprocessed backtest histories with certain parameters

- Generated sets are free from survivorship bias, by having inclusion criteria
  as historical TVL threshold

To export / update all exported data:

.. code-block:: shell

    python tradeexecutor/backtest/preprocessed_backtest_exporter.py ~/exported

Run using Docker. Created files will be placed in ``~/exported`` in the host FS:

.. code-block:: shell

    mkdir ~/exported
    # Get from https://github.com/tradingstrategy-ai/trade-executor/actions
    export TRADE_EXECUTOR_VERSION=latest
    docker run \
        -it \
        --entrypoint /usr/local/bin/python \
        --env TRADING_STRATEGY_API_KEY \
        -v ~/exported:/exported \
        -v ~/.cache:/root/.cache \
        ghcr.io/tradingstrategy-ai/trade-executor:${TRADE_EXECUTOR_VERSION} \
        /usr/src/trade-executor/tradeexecutor/backtest/preprocessed_backtest_exporter.py /exported

"""
import logging
import os
import pickle
import tempfile
from dataclasses import dataclass
import datetime
from pathlib import Path

import pandas as pd
from nbclient.exceptions import CellExecutionError
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor
import nbformat

from eth_defi.token import USDT_NATIVE_TOKEN, USDC_NATIVE_TOKEN
from tradeexecutor.backtest.tearsheet import BacktestReportRunFailed, DEFAULT_CUSTOM_CSS, _inject_custom_css_and_js, DEFAULT_CUSTOM_JS
from tradeexecutor.strategy.execution_context import python_script_execution_context
from tradeexecutor.strategy.trading_strategy_universe import load_partial_data, TradingStrategyUniverse, Dataset
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradeexecutor.utils.dedent import dedent_any
from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client
from tradingstrategy.pair import PandasPairUniverse
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.types import USDollarAmount, Percent
from tradingstrategy.utils.token_extra_data import load_token_metadata
from tradingstrategy.utils.token_filter import filter_pairs_default, filter_by_token_sniffer_score, deduplicate_pairs_by_volume, add_base_quote_address_columns
from tradingstrategy.utils.wrangle import fix_dex_price_data


logger = logging.getLogger(__name__)


DATASET_NOTEBOOK_TEMPLATE = os.path.join(os.path.dirname(__file__), "dataset_report_template.ipynb")


@dataclass
class BacktestDatasetDefinion:
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

    #: The main USDC/USDT token on the chain
    #:
    #: We use this to generate equally-weighted index report and as a reserve token in this index.
    reserve_token_address: str

    #: Prefilter pairs with this liquidity before calling token sniffer
    min_tvl: USDollarAmount | None = None

    #: Filter used in the reporting notebook.
    #:
    #: Note that you still need to do actual volum filtering in the
    #: dataset yourself, as volume 0 days are exported.
    min_weekly_volume: USDollarAmount | None = None

    categories: list[str] | None = None
    max_fee: Percent | None = None
    min_tokensniffer_score: int | None = None




@dataclass
class SavedDataset:
    set: BacktestDatasetDefinion
    parquet_path: Path
    csv_path: Path

    parquet_file_size: int
    csv_file_size: int
    pair_count: int
    row_count: int
    duration: datetime.timedelta | None

    def get_pair_count(self):
        return self.pair_count

    def get_info(self) -> pd.DataFrame:
        """Get human readable information of this dataset to be displayed in the notebook."""

        items = {
            "Dataset name": self.set.name,
            "Slug": self.set.slug,
            "Description": self.set.description,
            "Start": self.set.start,
            "End": self.set.end,
            "Chain": self.set.chain.get_name(),
            "Exchanges": ", ".join(self.set.exchanges),
            "Pair count (w/TVL criteria)": self.get_pair_count(),
            "Min TVL": f"{self.set.min_tvl:,} USD",
            "OHLCV timeframe": self.set.time_bucket.value,
            "OHLCV rows": f"{self.row_count:,}",
            "Parquet size": f"{self.parquet_file_size:,} bytes",
            "CSV size": f"{self.csv_file_size:,} bytes",
            "Job duration": {self.duration},
        }

        data = []
        for key, value in items.items():
            data.append({
                "Name": key,
                "Value": value,
            })
        df = pd.DataFrame(data)
        df = df.set_index("Name")
        return df


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


def run_and_write_report(
    output_html: Path,
    output_notebook: Path,
    dataset: SavedDataset,
    strategy_universe: TradingStrategyUniverse,
    custom_css=DEFAULT_CUSTOM_CSS,
    custom_js=DEFAULT_CUSTOM_JS,
    show_code=False,
    timeout=1800,
):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)

        universe_path = tmp_dir / "universe.pickle"
        dataset_path = tmp_dir / "dataset.pickle"

        with open(universe_path, "wb") as out:
            pickle.dump(strategy_universe, out)

        with open(dataset_path, "wb") as out:
            pickle.dump(dataset, out)

        # https://nbconvert.readthedocs.io/en/latest/execute_api.html
        with open(DATASET_NOTEBOOK_TEMPLATE) as f:
            nb = nbformat.read(f, as_version=4)

        # Replace the first cell that allows us to pass parameters
        # See
        # - https://github.com/nteract/papermill/blob/main/papermill/parameterize.py
        # - https://github.com/takluyver/nbparameterise/blob/master/nbparameterise/code.py
        # for inspiration
        cell = nb.cells[0]
        assert cell.cell_type == "code", f"Assumed first cell is parameter cell, got {cell}"
        assert "parameters =" in cell.source, f"Did not see parameters = definition in the cell source: {cell.source}"
        cell.source = f"""parameters = {{
            "universe_file": "{universe_path}", 
            "dataset_file": "{dataset_path}",
        }} """

        # Run the notebook
        universe_size = os.path.getsize(universe_path)
        dataset_size = os.path.getsize(dataset_path)
        logger.info(f"Starting backtest {dataset.set.slug}, dataset notebook execution, dataset size is {dataset_size:,}b, universe size is {universe_size:,}b")
        ep = ExecutePreprocessor(timeout=timeout, kernel_name='python3')

        try:
            ep.preprocess(nb, {'metadata': {'path': '.'}})
        except CellExecutionError as e:
            raise BacktestReportRunFailed(f"Could not run backtest reporter for {dataset_path}: {e}") from e

        logger.info("Notebook executed")

        # Write ipynb file that contains output cells created in place
        if output_notebook is not None:
            with open(output_notebook, 'w', encoding='utf-8') as f:
                nbformat.write(nb, f)

        # Write a static HTML file based on the notebook
        if output_html is not None:

            html_exporter = HTMLExporter(
                template_name='classic',
                embed_images=True,
                exclude_input=show_code is False,
                exclude_input_prompt=True,
                exclude_output_prompt=True,
            )
            # Image are inlined in the output
            html_content, resources = html_exporter.from_notebook_node(nb)

            # Inject our custom css
            if custom_css is not None:
                html_content = _inject_custom_css_and_js(html_content, custom_css, custom_js)

            with open(output_html, 'w', encoding='utf-8') as f:
                f.write(html_content)

            logger.info("Wrote HTML report to %s, total %d bytes", output_html, len(html_content))

        return nb


def prepare_dataset(
    client: Client,
    dataset: BacktestDatasetDefinion,
    output_folder: Path,
    write_csv=True,
    write_parquet=True,
    write_report=True,
    verbose=True,
) -> SavedDataset:
    """Prepare a predefined backtesting dataset.

    - Download data
    - Clean it
    - Write to a parquet file
    """

    started = datetime.datetime.utcnow()

    logger.info("Preparing dataset %s", dataset.slug)

    chain_id = dataset.chain
    time_bucket = dataset.time_bucket
    liquidity_time_bucket = TimeBucket.d1  # TVL data for Uniswap v3 is only sampled daily, more fine granular is not needed
    exchange_slugs = dataset.exchanges
    tokensniffer_threshold = dataset.min_tokensniffer_score
    min_liquidity_threshold = dataset.min_tvl  #
    max_tax = 0.06

    #
    # Set out trading pair universe
    #

    logger.info("Downloading/opening exchange dataset")
    exchange_universe = client.fetch_exchange_universe()

    # Resolve uniswap-v3 internal id
    targeted_exchanges = [exchange_universe.get_by_chain_and_slug(chain_id, slug) for slug in exchange_slugs]
    exchange_ids = [exchange.exchange_id for exchange in targeted_exchanges]
    exchange_universe = exchange_universe.limit_to_slugs(exchange_slugs)
    exchange_universe = exchange_universe.limit_to_chains({chain_id})
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

    # Server returns candles in random order
    tvl_df = tvl_df.sort_values("bucket")

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

    risk_filtered_pairs_df = risk_filtered_pairs_df[
        (risk_filtered_pairs_df["buy_tax"] < max_tax) | (risk_filtered_pairs_df["buy_tax"].isnull())
    ]
    risk_filtered_pairs_df = risk_filtered_pairs_df[
        (risk_filtered_pairs_df["sell_tax"] < max_tax) | (risk_filtered_pairs_df["sell_tax"].isnull())
    ]
    logger.info(
        "After tax tax filter %f we have %d pairs",
        max_tax,
        len(risk_filtered_pairs_df),
    )

    deduplicated_df = deduplicate_pairs_by_volume(risk_filtered_pairs_df)
    pairs_df = pd.concat([deduplicated_df, supporting_pairs_df]).drop_duplicates(subset='pair_id', keep='first')
    logger.info("After pairs deduplication we have %d pairs", len(deduplicated_df))

    # Supporting pairs lack metadata
    pairs_df.loc[pairs_df["token_metadata"].isna(), "token_metadata"] = None

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
    price_df["buy_tax"] = price_df["pair_id"].apply(lambda pair_id: pair_metadata[pair_id]["buy_tax"])
    price_df["sell_tax"] = price_df["pair_id"].apply(lambda pair_id: pair_metadata[pair_id]["sell_tax"])

    # Merge price and TVL data.x
    # For this we need to resample TVL to whatever timeframe the price happens to be in.
    liquidity_df = tvl_df
    liquidity_df = liquidity_df.rename(columns={'bucket': 'timestamp'})
    liquidity_df = liquidity_df.groupby('pair_id').apply(lambda x: x.set_index("timestamp").resample(time_bucket.to_frequency()).ffill(), include_groups=False)
    liquidity_df["tvl"] = liquidity_df["close"]

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
        "buy_tax",
        "sell_tax",
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
        merged_df.to_parquet(
            parquet_file,
            compression='zstd'
        )
        logger.info(f"Wrote {parquet_file}, {parquet_file.stat().st_size:,} bytes")
    else:
        parquet_file = None

    saved_dataset = SavedDataset(
        set=dataset,
        csv_path=csv_file,
        parquet_path=parquet_file,
        parquet_file_size=parquet_file.stat().st_size if parquet_file else None,
        csv_file_size=csv_file.stat().st_size if csv_file else None,
        pair_count=len(pairs_df),
        row_count=len(merged_df),
        duration=None,
        # df=merged_df,
        #pairs_df=pairs_df,
    )

    if write_report:

        dataset_pairs_df = pairs_df
        dataset_pairs_df["pair_id"] = dataset_pairs_df.index
        dataset_liquidty_df = tvl_df
        dataset_liquidty_df = dataset_liquidty_df.rename(columns={"bucket": "timestamp"})

        dataset_price_df = price_df

        universe_dataset = Dataset(
            time_bucket=time_bucket,
            exchanges=exchange_universe,
            pairs=dataset_pairs_df,
            candles=dataset_price_df,
            liquidity=dataset_liquidty_df,
            liquidity_time_bucket=liquidity_time_bucket,
            start_at=dataset.start,
            end_at=dataset.end,
        )

        strategy_universe = TradingStrategyUniverse.create_from_dataset(
            universe_dataset,
            reserve_asset=dataset.reserve_token_address,
            forward_fill_until=dataset.end,
            forward_fill=True,
        )

        # Check liquidity forward fill bug on Binance data
        # if dataset.slug == "binance-chain-1d":
        #     liquidity_universe = strategy_universe.data_universe.liquidity
        #     pair_id = 2184761
        #     ldf = strategy_universe.data_universe.liquidity.df
        #     pdf = ldf[ldf.pair_id == 2184761]
        #     l = liquidity_universe.get_liquidity_with_tolerance(
        #         pair_id,
        #         pd.Timestamp("2022-03-11"),
        #         tolerance=pd.Timedelta(days=1)
        #     )
        #     # assert strategy_universe.data_universe.liquidity.df.index.is_monotonic_increasing, "Liquidity was not monotonically increasing"

        output_html = output_folder / f"{dataset.slug}-report.html"
        output_notebook = output_folder / f"{dataset.slug}-report.ipynb"
        run_and_write_report(
            output_html=output_html,
            output_notebook=output_notebook,
            dataset=saved_dataset,
            strategy_universe=strategy_universe,
        )

    saved_dataset.duration = datetime.datetime.utcnow() - started

    return saved_dataset


BNB_QUOTE_TOKEN = USDT_NATIVE_TOKEN[ChainId.binance.value]

AVAX_QUOTE_TOKEN = USDC_NATIVE_TOKEN[ChainId.avalanche.value]

BASE_QUOTE_TOKEN = USDC_NATIVE_TOKEN[ChainId.base.value]

ETHEREUM_QUOTE_TOKEN = USDC_NATIVE_TOKEN[ChainId.ethereum.value]


PREPACKAGED_SETS = [
    BacktestDatasetDefinion(
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
        min_weekly_volume=200_000,
        time_bucket=TimeBucket.d1,
        exchanges={"pancakeswap-v2"},
        always_included_pairs=[
            (ChainId.binance, "pancakeswap-v2", "WBNB", "USDT"),
        ],
        reserve_token_address=BNB_QUOTE_TOKEN,
    ),

    BacktestDatasetDefinion(
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
        min_weekly_volume=200_000,
        exchanges={"pancakeswap-v2"},
        always_included_pairs=[
            (ChainId.binance, "pancakeswap-v2", "WBNB", "USDT"),
        ],
        reserve_token_address=BNB_QUOTE_TOKEN,
    ),

    BacktestDatasetDefinion(
        chain=ChainId.avalanche,
        slug="avalanche-1d",
        name="Avalanche C-Chain, LFG, 2021-2025, daily",
        description=dedent_any("""
        LFG, formerly known as Trader Joe, DEX daily trades.
        
        - Contains bull and bear market data with mixed set of tokens
        """),
        start=datetime.datetime(2021, 1, 1),
        end=datetime.datetime(2025, 1, 1),
        time_bucket=TimeBucket.d1,
        min_tvl=250_000,
        min_weekly_volume=250_000,
        exchanges={"trader-joe"},
        always_included_pairs=[
            (ChainId.avalanche, "trader-joe", "WAVAX", "USDT.e", 0.0030),
            (ChainId.avalanche, "trader-joe", "WETH.e", "WAVAX", 0.0030),  # Only trading since October

        ],
        reserve_token_address=AVAX_QUOTE_TOKEN,
    ),

    BacktestDatasetDefinion(
        chain=ChainId.avalanche,
        slug="avalanche-1h",
        name="Avalanche C-Chain, LFG, 2021-2025, hourly",
        description=dedent_any("""
        LFG, formerly known as Trader Joe, DEX hourly trades.
    
        - Contains bull and bear market data with mixed set of tokens
        """),
        start=datetime.datetime(2021, 1, 1),
        end=datetime.datetime(2025, 1, 1),
        time_bucket=TimeBucket.h1,
        min_tvl=250_000,
        min_weekly_volume=250_000,
        exchanges={"trader-joe"},
        always_included_pairs=[
            (ChainId.avalanche, "trader-joe", "WAVAX", "USDT.e", 0.0030),
            (ChainId.avalanche, "trader-joe", "WETH.e", "WAVAX", 0.0030),  # Only trading since October

        ],
        reserve_token_address=AVAX_QUOTE_TOKEN,
    ),

    BacktestDatasetDefinion(
        chain=ChainId.base,
        slug="base-1h",
        name="Base, Uniswap, 2024-2025/Q2, hourly",
        description=dedent_any("""
        - Base Uniswap v2 and v3 trading pairs with a minimum TVL threshold
        """),
        start=datetime.datetime(2024, 1, 1),
        end=datetime.datetime(2025, 3, 1),
        time_bucket=TimeBucket.h1,
        min_tvl=500_000,
        min_weekly_volume=500_000,
        exchanges={"uniswap-v2", "uniswap-v3"},
        always_included_pairs=[
            (ChainId.base, "uniswap-v2", "WETH", "USDC", 0.0030),
            (ChainId.base, "uniswap-v3", "cbBTC", "WETH", 0.0030),  # Only trading since October
        ],
        reserve_token_address=BASE_QUOTE_TOKEN,
    ),

    BacktestDatasetDefinion(
        chain=ChainId.base,
        slug="base-1d",
        name="Base, Uniswap, 2024-2025/Q2, hourly",
        description=dedent_any("""
    - Base Uniswap v2 and v3 trading pairs with a minimum TVL threshold
    """),
        start=datetime.datetime(2024, 1, 1),
        end=datetime.datetime(2025, 3, 1),
        time_bucket=TimeBucket.d1,
        min_tvl=500_000,
        min_weekly_volume=500_000,
        exchanges={"uniswap-v2", "uniswap-v3"},
        always_included_pairs=[
            (ChainId.base, "uniswap-v2", "WETH", "USDC", 0.0030),
            (ChainId.base, "uniswap-v3", "cbBTC", "WETH", 0.0030),  # Only trading since October
        ],
        reserve_token_address=BASE_QUOTE_TOKEN,
    ),

    BacktestDatasetDefinion(
        chain=ChainId.ethereum,
        slug="ethereum-1d",
        name="Ethereum mainnet, Uniswap and Sushiswap, 2020-2025/Q2, daily",
        description=dedent_any("""
        Ethereum Uniswap and Sushiswap DEX traeds.
    
        - Longest DEX history we have
        - Contains bull and bear market data with mixed set of tokens
        """),
        start=datetime.datetime(2020, 1, 1),
        end=datetime.datetime(2025, 3, 1),
        time_bucket=TimeBucket.d1,
        min_tvl=3_000_000,
        min_weekly_volume=100_000,
        exchanges={"uniswap-v2", "uniswap-v3", "sushi"},
        always_included_pairs=[
            (ChainId.ethereum, "uniswap-v2", "WETH", "USDC", 0.0030),
            (ChainId.ethereum, "uniswap-v3", "WBTC", "USDC", 0.0030),  # Only trading since October
        ],
        reserve_token_address=ETHEREUM_QUOTE_TOKEN,
    ),

    BacktestDatasetDefinion(
        chain=ChainId.ethereum,
        slug="ethereum-1h",
        name="Ethereum mainnet, Uniswap and Sushiswap, 2020-2025/Q2, hourly",
        description=dedent_any("""
        Ethereum Uniswap and Sushiswap DEX traeds.
        
        - Longest DEX history we have
        - Contains bull and bear market data with mixed set of tokens
        """),
        start=datetime.datetime(2020, 1, 1),
        end=datetime.datetime(2025, 3, 1),
        time_bucket=TimeBucket.h1,
        min_tvl=3_000_000,
        min_weekly_volume=100_000,
        exchanges={"uniswap-v2", "uniswap-v3", "sushi"},
        always_included_pairs=[
            (ChainId.ethereum, "uniswap-v2", "WETH", "USDC", 0.0030),
            (ChainId.ethereum, "uniswap-v3", "WBTC", "USDC", 0.0030),  # Only trading since October
        ],
        reserve_token_address=ETHEREUM_QUOTE_TOKEN,
    ),
]

