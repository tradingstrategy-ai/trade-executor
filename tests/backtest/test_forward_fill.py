"""Data forward fill tests."""
import datetime

import flaky
import pandas as pd
import pytest

from tradeexecutor.strategy.execution_context import unit_test_execution_context
from tradeexecutor.strategy.trading_strategy_universe import load_partial_data, TradingStrategyUniverse, TradingStrategyUniverseModel
from tradeexecutor.strategy.universe_model import UniverseOptions, DataTooOld
from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client
from tradingstrategy.pair import PandasPairUniverse
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.utils.forward_fill import forward_fill


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

    start_at = datetime.datetime(2024, 1, 1)
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
        forward_fill_until=datetime.datetime(2024, 5, 1),
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

    # Check our DataTooOld alert ignores forward filcal
    assert "forward_filled" in strategy_universe.data_universe.candles.df.columns
    assert strategy_universe.data_universe.candles.is_forward_filled() == True

    # We should be able to get time range for forward filled data
    time_range = strategy_universe.data_universe.candles.get_timestamp_range(exclude_forward_fill=False)
    assert time_range == (pd.Timestamp('2024-01-01 00:00:00'), pd.Timestamp('2024-05-01 00:00:00'))

    # For alerts, we should be able to get time range that excludes any synthetic forward-filled values
    time_range = strategy_universe.data_universe.candles.get_timestamp_range(exclude_forward_fill=True)
    assert time_range == (pd.Timestamp('2024-01-01 00:00:00'), pd.Timestamp('2024-03-15 00:00:00'))



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

    # Check our DataTooOld alert ignores forward filcal
    assert strategy_universe.data_universe.candles.is_forward_filled() == False
    time_range = strategy_universe.data_universe.candles.get_timestamp_range(exclude_forward_fill=True)
    assert time_range == (pd.Timestamp('2022-01-01 00:00:00'), pd.Timestamp('2024-03-15 00:00:00'))



def test_forward_fill_too_old(persistent_test_client: Client):
    """Data too old warning triggers on forward filled data.

    - Data is forward-filled until today, but should be still too old
      as we would not have any data for the last 30 days.
    """
    client = persistent_test_client

    # The pairs we are rading
    pair_ids = [
        (ChainId.polygon, "quickswap", "WBTC", "WETH", 0.0030),  # This data has a lot of gaps
        (ChainId.polygon, "uniswap-v3", "WMATIC", "USDC", 0.0005),
    ]

    start_at = datetime.datetime(2024, 1, 1)
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

    now_= datetime.datetime.utcnow()
    strategy_universe = TradingStrategyUniverse.create_from_dataset(
        dataset,
        reserve_asset="0x2791bca1f2de4661ed88a30c99a7a9449aa84174",
        forward_fill=True,
        forward_fill_until=now_,
    )

    with pytest.raises(DataTooOld):
        TradingStrategyUniverseModel.check_data_age(
            ts=now_,
            strategy_universe=strategy_universe,
            best_before_duration=datetime.timedelta(days=30),
        )


def test_forward_fill_tvl_freq(persistent_test_client: Client):
    """Do TVL forward fill and increase granularity to 1h"""

    client = persistent_test_client

    pair_ids = [
        (ChainId.polygon, "quickswap", "WBTC", "WETH", 0.0030),  # This data has a lot of gaps
        (ChainId.polygon, "uniswap-v3", "WMATIC", "USDC", 0.0005),
    ]

    start_at = datetime.datetime(2024, 1, 1)
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
        liquidity_time_bucket=TimeBucket.d1,
        pairs=pair_ids,
        execution_context=unit_test_execution_context,
        liquidity=True,
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

    liquidity = strategy_universe.data_universe.liquidity
    pair = strategy_universe.get_pair_by_human_description(pair_ids[0])
    tvl= liquidity.get_samples_by_pair(pair.internal_id)

    df = pd.DataFrame({
        "close": tvl["close"]
    })

    # Resample 1d -> 1h
    # forward fill the data until a specific date
    end_timestamp = datetime.datetime(2024, 6, 1)
    df_ff = forward_fill(
        df,
        "1h",
        columns=("close",),
        forward_fill_until=end_timestamp,
    )
    series = df_ff["close"]

    # Check we forward-filled index
    assert series.index[-1] == pd.Timestamp("2024-06-01 00:00:00")
    assert series.index[-2] == pd.Timestamp("2024-05-31 23:00:00")
    assert series.index[0] == pd.Timestamp("2024-01-01 00:00:00")
    assert series.index[1] == pd.Timestamp("2024-01-01 01:00:00")

    # See forward_filled flag is correctly set
    assert df_ff.loc[pd.Timestamp("2024-06-01 00:00:00")]["forward_filled"] == True
    assert df_ff.loc[pd.Timestamp("2024-05-31 23:00:00")]["forward_filled"] == True
    assert df_ff.loc[pd.Timestamp("2024-01-01 00:00:00")]["forward_filled"] == False
    assert df_ff.loc[pd.Timestamp("2024-01-01 01:00:00")]["forward_filled"] == True
