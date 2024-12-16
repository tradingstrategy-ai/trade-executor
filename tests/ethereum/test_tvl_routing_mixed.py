"""TVL routing with Uniswap v2 and v3 pairs mixed.

- Copied from ethereum-memecoin-vol-basket.py

"""

import datetime
import os
import secrets
from dataclasses import asdict
from decimal import Decimal

import pandas as pd
import pytest
from hexbytes import HexBytes

from tradeexecutor.cli.bootstrap import create_execution_and_sync_model, create_web3_config
from tradeexecutor.cli.log import setup_pytest_logging
from tradeexecutor.strategy.approval import UncheckedApprovalModel
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.execution_context import ExecutionContext, unit_test_trading_execution_context
from tradeexecutor.strategy.execution_model import AssetManagementMode
from tradeexecutor.strategy.pandas_trader.runner import PandasTraderRunner
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.trade_pricing import TradePricing
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, load_partial_data
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradeexecutor.utils.timer import timed_task
from tradingstrategy.alternative_data.coingecko import CoingeckoUniverse, categorise_pairs
from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client
from tradingstrategy.pair import PandasPairUniverse
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.utils.token_extra_data import filter_scams
from tradingstrategy.utils.token_filter import deduplicate_pairs_by_volume, filter_for_base_tokens

JSON_RPC_ETHEREUM = os.environ.get("JSON_RPC_ETHEREUM")

pytestmark = pytest.mark.skipif(not JSON_RPC_ETHEREUM, reason="Give JSON_RPC_ETHEREUM to run")

#: Assets used in routing and buy-and-hold benchmark values for our strategy, but not traded by this strategy.
SUPPORTING_PAIRS = [
    (ChainId.ethereum, "uniswap-v3", "WBTC", "USDT", 0.0005),
    (ChainId.ethereum, "uniswap-v3", "WETH", "USDC", 0.0005),
    (ChainId.ethereum, "uniswap-v2", "WETH", "USDC", 0.0030),  # TODO: Needed until we have universal routing
]


class Parameters:
    id = "ethereum-memecoin-vol-basket"

    # We trade 1h candle
    candle_time_bucket = TimeBucket.h1
    cycle_duration = CycleDuration.cycle_4h

    # Coingecko categories to include
    #
    # See list here: TODO
    #
    chain_id = ChainId.ethereum
    categories = {"Meme"}
    exchanges = {"uniswap-v2", "uniswap-v3"}

    #
    # Basket construction and rebalance parameters
    #
    min_asset_universe = 10  # How many assets we need in the asset universe to start running the index
    max_assets_in_portfolio = 10  # How many assets our basket can hold once
    allocation = 0.99  # Allocate all cash to volatile pairs
    min_rebalance_trade_threshold_pct = 0.10  # % of portfolio composition must change before triggering rebalacne
    individual_rebalance_min_threshold_usd = 10  # Don't make buys less than this amount
    min_volatility_threshold = 0.02  # Set to have Sharpe ratio threshold for the inclusion
    per_position_cap_of_pool = 0.01  # Never own more than % of the lit liquidity of the trading pool
    max_concentration = 0.25  # How large % can one asset be in a portfolio once
    min_signal_threshold = 0.25

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
    assumed_missing_liquidity = 100_000  # USD

    #
    # Backtesting only
    #
    backtest_start = datetime.datetime(2022, 8, 15)
    backtest_end = datetime.datetime(2024, 10, 20)
    initial_cash = 10_000
    too_little_equity = 0.01

    #
    # Live only
    #
    routing = TradeRouting.default
    required_history_period = datetime.timedelta(days=2)
    slippage_tolerance = 0.0060  # 0.6%


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
    # deduplicated_df = deduplicate_pairs_by_volume(pairs_df)
    test_pairs_df = filter_for_base_tokens(
        pairs_df,
        {
            "0x594daad7d77592a2b97b725a7ad59d7e188b5bfa",  # APU
            "0x812ba41e071c7b7fa4ebcfb62df5f45f6fa853ee",  # Neiro
        }
    )

    pairs_df = pd.concat([test_pairs_df, supporting_pairs_df]).drop_duplicates(subset='pair_id', keep='first')

    print(
        f"Total {len(pairs_df)} pairs to trade on {chain_id.name} for categories {categories}",
    )

    # Scam filter using TokenSniffer
    # pairs_df = filter_scams(pairs_df, client, min_token_sniffer_score=Parameters.min_token_sniffer_score)
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
    )

    strategy_universe = TradingStrategyUniverse.create_from_dataset(
        dataset,
        reserve_asset="0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",  # USDC
        forward_fill=False,  # We got very gappy data from low liquid DEX coins
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
        print(f"   {idx + 1}. pair #{pair.internal_id}: {pair.base.token_symbol} - {pair.quote.token_symbol} ({pair.exchange_name} w/ {pair.fee} fee), {'benchmark/routed' if benchmark else 'traded'}")

    return strategy_universe


@pytest.fixture(scope="module")
def logger(request):
    """Setup test logger."""
    return setup_pytest_logging(request, mute_requests=False)


def test_tvl_routing_mixed(persistent_test_client, logger):
    """Don't crash when getting TVL for APU/WETH.

    - Setup routing model as the live executor would do

    - Figure out what goes wrong when calling get_usd_tvl() and get_buy_price() when mixing Uniswap v2 and v3 pairs
    """
    client = persistent_test_client

    parameters = StrategyParameters.from_class(Parameters)

    execution_context = ExecutionContext(**asdict(unit_test_trading_execution_context))
    execution_context.engine_version = "0.5"

    strategy_universe = create_trading_universe(
        None,
        client,
        execution_context,
        universe_options=UniverseOptions.from_strategy_parameters_class(parameters, execution_context),
    )

    web3config = create_web3_config(
        json_rpc_ethereum=JSON_RPC_ETHEREUM,
        json_rpc_binance=None,
        json_rpc_polygon=None,
        json_rpc_avalanche=None,
        json_rpc_arbitrum=None,
        json_rpc_anvil=None,
        json_rpc_base=None,
    )

    web3config.set_default_chain(ChainId.ethereum)

    execution_model, sync_model, valuation_model_factory, pricing_model_factory = create_execution_and_sync_model(
        asset_management_mode=AssetManagementMode.hot_wallet,
        private_key=HexBytes(secrets.token_bytes(32)).hex(),
        web3config=web3config,
        confirmation_timeout=datetime.timedelta(seconds=1),
        confirmation_block_count=0,
        max_slippage=0.1,
        min_gas_balance=Decimal(0),
        vault_address=None,
        vault_adapter_address=None,
        vault_payment_forwarder_address=None,
        routing_hint=TradeRouting.default,
    )

    runner = PandasTraderRunner(
        timed_task_context_manager=timed_task,
        execution_model=execution_model,
        approval_model=UncheckedApprovalModel(),
        valuation_model_factory=valuation_model_factory,
        sync_model=sync_model,
        pricing_model_factory=pricing_model_factory,
        routing_model=None,  # Automatically generated later
        decide_trades=lambda x: x,
        execution_context=execution_context,
        trade_settle_wait=datetime.timedelta(seconds=1),
        unit_testing=True,
    )

    routing_state, pricing_model, valuation_model = runner.setup_routing(strategy_universe)

    # Test Uni v2 pair
    pair = strategy_universe.get_pair_by_human_description(
        (ChainId.ethereum, "uniswap-v2", "APU", "WETH")
    )

    tvl_usd = pricing_model.get_usd_tvl(None, pair)
    # print(pair, tvl_usd)
    assert tvl_usd > 0

    price_structure = pricing_model.get_buy_price(None, pair, Decimal(1000.00))
    assert price_structure is not None

    # Test second Uni v2 pair
    pair = strategy_universe.get_pair_by_human_description(
        (ChainId.ethereum, "uniswap-v2", "WETH", "USDC")
    )

    tvl_usd = pricing_model.get_usd_tvl(None, pair)
    # print(pair, tvl_usd)
    assert tvl_usd > 0

    # Test Uni v3 pair
    pair = strategy_universe.get_pair_by_human_description(
        (ChainId.ethereum, "uniswap-v3", "WETH", "USDC", 0.0005)
    )

    tvl_usd = pricing_model.get_usd_tvl(None, pair)
    # print(pair, tvl_usd)
    assert tvl_usd > 0

    # Test second Uni v3 pair
    pair = strategy_universe.get_pair_by_human_description(
        (ChainId.ethereum, "uniswap-v3", "Neiro", "WETH", 0.0030)
    )

    tvl_usd = pricing_model.get_usd_tvl(None, pair)
    # print(pair, tvl_usd)
    assert tvl_usd > 0

    price_structure = pricing_model.get_buy_price(None, pair, Decimal(1000.00))
    assert isinstance(price_structure, TradePricing)





