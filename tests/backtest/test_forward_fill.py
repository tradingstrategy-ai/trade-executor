"""Data forward fill tests."""
import datetime

import pandas as pd

from tradeexecutor.strategy.execution_context import unit_test_execution_context
from tradeexecutor.strategy.trading_strategy_universe import load_partial_data, TradingStrategyUniverse
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client
from tradingstrategy.pair import PandasPairUniverse
from tradingstrategy.timebucket import TimeBucket


def test_forward_fill_spot_only_forward_filled(persistent_test_client: Client):
    """Forward-will spot market data.

    - When dataset is loaded, forward-fill any gaps when the dataset is transformed to the trading universe
    """
    client = persistent_test_client

    # The pairs we are rading
    pair_ids = [
        (ChainId.polygon, "quickswap", "WBTC", "WETH", 0.0030),  # This data has a lot of gaps
        (ChainId.polygon, "uniswap-v3", "WMATIC", "USDC", 0.0005),
    ]

    start_at = datetime.datetime(2022, 1, 1)
    end_at = datetime.datetime(2024, 3, 15)

    universe_options = UniverseOptions(
        start_at=start_at,
        end_at=end_at,
    )

    # Load all trading and lending data on Polygon
    # for all lending markets on a relevant time period
    dataset = load_partial_data(
        client=client,
        time_bucket=TimeBucket.h1,
        pairs=pair_ids,
        execution_context=unit_test_execution_context,
        liquidity=False,
        stop_loss_time_bucket=None,
        universe_options=universe_options,
    )

    # Token debug
    pairs = PandasPairUniverse(dataset.pairs, exchange_universe=dataset.exchanges)
    for t in pairs.iterate_tokens():
        assert t.chain_id == ChainId.polygon

    strategy_universe = TradingStrategyUniverse.create_from_dataset(
        dataset,
        reserve_asset="0x2791bca1f2de4661ed88a30c99a7a9449aa84174",
        forward_fill=True,
    )

    assert strategy_universe.data_universe.time_bucket == TimeBucket.h1
    assert strategy_universe.data_universe.candles.time_bucket == TimeBucket.h1

    wbtc_weth = strategy_universe.get_pair_by_human_description(pair_ids[0])
    btc_df = strategy_universe.data_universe.candles.get_candles_by_pair(wbtc_weth.internal_id)
    assert "timestamp" in btc_df.columns
    btc_close = strategy_universe.data_universe.candles.get_candles_by_pair(wbtc_weth.internal_id)["close"]

    # From MultiIndex to single series index
    # Drop pair id and only have dates in the index
    # MultiIndex([(93807, '2022-01-01'),
    #        (93807, '2022-01-02'),
    #         (93807, '2022-01-03'),
    btc_close_index_flattened = btc_close.index

    # Check there are no 1h gaps in the data
    # https://stackoverflow.com/a/42555628/315168
    mask = btc_close_index_flattened.to_series().diff() > pd.Timedelta('01:00:00')
    assert any(mask) is False


def test_forward_fill_spot_only_gapped(persistent_test_client: Client):
    """Forward-will spot market data.

    - Do not use forward fill

    - See gaps in the data
    """
    client = persistent_test_client

    # The pairs we are rading
    pair_ids = [
        (ChainId.polygon, "quickswap", "WBTC", "WETH", 0.0030),  # This data has a lot of gaps
        (ChainId.polygon, "uniswap-v3", "WMATIC", "USDC", 0.0005),
    ]

    start_at = datetime.datetime(2022, 1, 1)
    end_at = datetime.datetime(2024, 3, 15)

    universe_options = UniverseOptions(
        start_at=start_at,
        end_at=end_at,
    )

    # Load all trading and lending data on Polygon
    # for all lending markets on a relevant time period
    dataset = load_partial_data(
        client=client,
        time_bucket=TimeBucket.h1,
        pairs=pair_ids,
        execution_context=unit_test_execution_context,
        liquidity=False,
        stop_loss_time_bucket=None,
        universe_options=universe_options,
    )

    # Token debug
    pairs = PandasPairUniverse(dataset.pairs, exchange_universe=dataset.exchanges)
    for t in pairs.iterate_tokens():
        assert t.chain_id == ChainId.polygon

    strategy_universe = TradingStrategyUniverse.create_from_dataset(
        dataset,
        reserve_asset="0x2791bca1f2de4661ed88a30c99a7a9449aa84174",
        forward_fill=False,
    )

    assert strategy_universe.data_universe.time_bucket == TimeBucket.h1
    assert strategy_universe.data_universe.candles.time_bucket == TimeBucket.h1

    wbtc_weth = strategy_universe.get_pair_by_human_description(pair_ids[0])
    btc_df = strategy_universe.data_universe.candles.get_candles_by_pair(wbtc_weth.internal_id)
    assert "timestamp" in btc_df.columns
    btc_close = strategy_universe.data_universe.candles.get_candles_by_pair(wbtc_weth.internal_id)["close"]


    # TODO: Why index here is different?
    btc_close_index_flattened = btc_close.index

    # Check there are some 1h gaps in the data
    # https://stackoverflow.com/a/42555628/315168
    mask = btc_close_index_flattened.to_series().diff() > pd.Timedelta('01:00:00')
    assert any(mask) is True
