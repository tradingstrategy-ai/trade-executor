import datetime
import os
import pytest
import pandas as pd
from pandas import Timestamp
from unittest.mock import patch

from tradeexecutor.utils.binance import create_binance_universe, fetch_binance_dataset
from tradeexecutor.strategy.trading_strategy_universe import Dataset
from tradingstrategy.binance.downloader import BinanceDownloader
from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.binance.constants import BINANCE_EXCHANGE_ID, BINANCE_CHAIN_ID
from tradingstrategy.utils.groupeduniverse import resample_candles

START_AT = datetime.datetime(2021, 1, 1)
END_AT = datetime.datetime(2021, 1, 2)
TIME_BUCKET = TimeBucket.d1
STOP_LOSS_TIME_BUCKET = TimeBucket.h4


@pytest.fixture()
def correct_df_candles() -> pd.DataFrame:
    """Return a correct dataframe for the candles."""
    data = {
        'timestamp': [
            '2021-01-01 00:00:00', '2021-01-01 04:00:00', '2021-01-01 08:00:00',
            '2021-01-01 12:00:00', '2021-01-01 16:00:00', '2021-01-01 20:00:00',
            '2021-01-02 00:00:00'
        ],
        'open': [736.42, 744.87, 737.37, 738.85, 735.39, 725.34, 728.91],
        'high': [749.0, 747.09, 741.76, 743.33, 737.73, 731.97, 734.4],
        'low': [729.33, 734.4, 725.1, 732.12, 714.29, 722.5, 714.91],
        'close': [744.82, 737.38, 738.85, 735.39, 725.34, 728.91, 730.39],
        'volume': [130893.19622, 72474.10311, 128108.21447, 121504.02184, 156457.71927, 65676.83838, 119184.4496],
        'pair_id': [1, 1, 1, 1, 1, 1, 1],
        'base_token_symbol': ['ETH', 'ETH', 'ETH', 'ETH', 'ETH', 'ETH', 'ETH'],
        'quote_token_symbol': ['USDT', 'USDT', 'USDT', 'USDT', 'USDT', 'USDT', 'USDT'],
        'exchange_slug': ['binance', 'binance', 'binance', 'binance', 'binance', 'binance', 'binance'],
        'chain_id': [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        'fee': [0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005],
        'buy_volume_all_time': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        'address': [
            '0xe82ac67166a910f4092c23f781cd39e46582ec9c', '0xe82ac67166a910f4092c23f781cd39e46582ec9c',
            '0xe82ac67166a910f4092c23f781cd39e46582ec9c', '0xe82ac67166a910f4092c23f781cd39e46582ec9c',
            '0xe82ac67166a910f4092c23f781cd39e46582ec9c', '0xe82ac67166a910f4092c23f781cd39e46582ec9c',
            '0xe82ac67166a910f4092c23f781cd39e46582ec9c'
        ],
        'exchange_id': [129875571.0, 129875571.0, 129875571.0, 129875571.0, 129875571.0, 129875571.0, 129875571.0],
        'token0_address': [
            '0x4b2d72c1cb89c0b2b320c43bb67ff79f562f5ff4', '0x4b2d72c1cb89c0b2b320c43bb67ff79f562f5ff4',
            '0x4b2d72c1cb89c0b2b320c43bb67ff79f562f5ff4', '0x4b2d72c1cb89c0b2b320c43bb67ff79f562f5ff4',
            '0x4b2d72c1cb89c0b2b320c43bb67ff79f562f5ff4', '0x4b2d72c1cb89c0b2b320c43bb67ff79f562f5ff4',
            '0x4b2d72c1cb89c0b2b320c43bb67ff79f562f5ff4'
        ],
        'token1_address': [
            '0x5b1a1833b16b6594f92daa9f6d9b7a6024bce9d0', '0x5b1a1833b16b6594f92daa9f6d9b7a6024bce9d0',
            '0x5b1a1833b16b6594f92daa9f6d9b7a6024bce9d0', '0x5b1a1833b16b6594f92daa9f6d9b7a6024bce9d0',
            '0x5b1a1833b16b6594f92daa9f6d9b7a6024bce9d0', '0x5b1a1833b16b6594f92daa9f6d9b7a6024bce9d0',
            '0x5b1a1833b16b6594f92daa9f6d9b7a6024bce9d0'
        ],
        'token0_symbol': ['ETH', 'ETH', 'ETH', 'ETH', 'ETH', 'ETH', 'ETH'],
        'token1_symbol': ['USDT', 'USDT', 'USDT', 'USDT', 'USDT', 'USDT', 'USDT'],
        'token0_decimals': [18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0],
        'token1_decimals': [18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0]
    }
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.index = pd.Int64Index([0, 1, 2, 3, 4, 5, 6], dtype='int64')

    if os.environ.get("GITHUB_ACTIONS", None) == True:
        # gets converted internally
        df['pair_id'] == ["ETHUSDT", "ETHUSDT", "ETHUSDT", "ETHUSDT", "ETHUSDT", "ETHUSDT", "ETHUSDT"]

    return df


@pytest.fixture()
def correct_df_lending():
    """Return a correct dataframe for the lending."""

    data = {
        'lending_rates': [0.000250, 0.000250, 0.001045, 0.001045],
        'pair_id': ['ETH', 'ETH', 'USDT', 'USDT']
    }
    df = pd.DataFrame(data, index=pd.to_datetime(['2020-12-31', '2021-01-01', '2020-12-31', '2021-01-01']))
    return df


@pytest.mark.skipif(os.environ.get("BINANCE_LENDING_DATA") == "false", reason="Binance lending API not available in the country")
def test_fetch_binance_dataset(correct_df_candles, correct_df_lending):
    """Test that the fetch_binance_dataset function works as expected."""
    if os.environ.get("GITHUB_ACTIONS", None) == "true":
        with patch(
            "tradingstrategy.binance.downloader.BinanceDownloader.fetch_candlestick_data"
        ) as mock_fetch_candlestick_data, patch(
            "tradingstrategy.binance.downloader.BinanceDownloader.fetch_lending_rates"
        ) as mock_fetch_lending_data:
            mock_fetch_candlestick_data.return_value = correct_df_candles
            mock_fetch_lending_data.return_value = correct_df_lending

            dataset = fetch_binance_dataset(
                ["ETHUSDT"],
                TIME_BUCKET,
                STOP_LOSS_TIME_BUCKET,
                START_AT,
                END_AT,
                include_lending=True,
            )
    else:
        dataset = fetch_binance_dataset(
            ["ETHUSDT"],
            TIME_BUCKET,
            STOP_LOSS_TIME_BUCKET,
            START_AT,
            END_AT,
            include_lending=True,
            force_download=True,
        )
    
    assert dataset.backtest_stop_loss_candles.equals(correct_df_candles)
    assert len(dataset.backtest_stop_loss_candles) == 7
    assert dataset.backtest_stop_loss_candles.isna().sum().sum() == 0

    assert len(dataset.candles) == 2
    assert dataset.candles.isna().sum().sum() == 0
    
    assert len(dataset.pairs) == 1
    assert dataset.time_bucket == TimeBucket.d1
    assert dataset.backtest_stop_loss_time_bucket == TimeBucket.h4
    assert dataset.exchanges.exchanges[BINANCE_EXCHANGE_ID]

    assert dataset.candles["timestamp"][0].to_pydatetime() == START_AT
    assert dataset.candles["timestamp"][-1].to_pydatetime() == END_AT
    
    assert dataset.backtest_stop_loss_candles["timestamp"].iloc[0].to_pydatetime() == START_AT
    assert dataset.backtest_stop_loss_candles["timestamp"].iloc[-1].to_pydatetime() == END_AT


@pytest.mark.skipif(os.environ.get("BINANCE_LENDING_DATA") == "false", reason="Binance lending API not available in the country")
def test_create_binance_universe(correct_df_candles, correct_df_lending):
    """Test that the create_binance_universe function works as expected."""
    if os.environ.get("GITHUB_ACTIONS", None) == "true":
        with patch(
            "tradingstrategy.binance.downloader.BinanceDownloader.fetch_candlestick_data"
        ) as mock_fetch_candlestick_data, patch(
            "tradingstrategy.binance.downloader.BinanceDownloader.fetch_lending_rates"
        ) as mock_fetch_lending_data:
            mock_fetch_candlestick_data.return_value = correct_df_candles
            mock_fetch_lending_data.return_value = correct_df_lending
            universe = create_binance_universe(
                ["ETHUSDT"],
                TIME_BUCKET,
                STOP_LOSS_TIME_BUCKET,
                START_AT,
                END_AT,
                include_lending=True,
            )
    else:
        universe = create_binance_universe(
            ["ETHUSDT"],
            TIME_BUCKET,
            STOP_LOSS_TIME_BUCKET,
            START_AT,
            END_AT,
            include_lending=True,
            force_download=True,
        )

    assert universe.backtest_stop_loss_time_bucket == TimeBucket.h4
    assert len(universe.backtest_stop_loss_candles.df) == 7
    assert universe.backtest_stop_loss_candles.df.isna().sum().sum() == 0

    data_universe = universe.data_universe
    assert data_universe.time_bucket == TimeBucket.d1
    assert len(data_universe.candles.df) == 2
    assert data_universe.candles.df.isna().sum().sum() == 0
    assert data_universe.candles.df.index[0].to_pydatetime() == START_AT
    assert data_universe.candles.df.index[-1].to_pydatetime() == END_AT

    assert len(data_universe.lending_reserves.reserves) == 2
    assert data_universe.chains == {BINANCE_CHAIN_ID}
    assert len(data_universe.pairs.df) == 1
    assert len(data_universe.pairs.df.columns) == 37
    # pairs df can have nans

    assert len(data_universe.lending_candles.variable_borrow_apr.df) == 4
    assert data_universe.lending_candles.variable_borrow_apr.df.isna().sum().sum() == 0
    assert len(data_universe.lending_candles.supply_apr.df) == 4
    assert data_universe.lending_candles.supply_apr.df.isna().sum().sum() == 0

    assert data_universe.lending_candles.variable_borrow_apr.df.index[0].to_pydatetime() == START_AT
    assert data_universe.lending_candles.variable_borrow_apr.df.index[-1].to_pydatetime() == END_AT



@pytest.mark.skipif(os.environ.get("BINANCE_LENDING_DATA") == "false", reason="Binance lending API not available in the country")
def test_create_binance_universe_multipair():
    """Test that the create_binance_universe function works as expected for multipair."""
    candles_data = {
        "open": [
            736.42, 744.87, 737.37, 738.85, 735.39, 725.34, 728.91, 730.39, 735.12, 729.7,
            768.45, 784.79, 28923.63, 29278.41, 29092.84, 29313.49, 29188.67, 29029.04, 29331.7, 29351.95,
            29751.47, 29754.99, 31691.09, 33027.2
        ],
        "high": [
            749, 747.09, 741.76, 743.33, 737.73, 731.97, 734.4, 740.49, 738.35, 772.8,
            787.69, 785.48, 29470, 29395, 29402.57, 29600, 29360, 29338.89, 29469, 29820.5,
            29899, 31800, 33300, 33061.37
        ],
        "low": [
            729.33, 734.4, 725.1, 732.12, 714.29, 722.5, 714.91, 726.26, 723.01, 728.25,
            764.5, 750.12, 28690.17, 28806.54, 28872.24, 29030.14, 28624.57, 28880.37, 28946.53, 29220,
            29473.91, 29741.39, 31616.42, 30300
        ],
        "close": [
            744.82, 737.38, 738.85, 735.39, 725.34, 728.91, 730.39, 735.12, 729.7, 768.43,
            784.79, 774.56, 29278.4, 29092.83, 29313.49, 29188.67, 29029.04, 29331.69, 29351.95, 29750,
            29755, 31691.29, 33027.2, 32178.33
        ],
        "volume": [
            130893.19622, 72474.10311, 128108.21447, 121504.02184, 156457.71927, 65676.83838, 119184.4496, 97938.27713, 120264.73109, 428448.45842,
            334396.9924, 252385.66804, 11560.456553, 7308.910274, 8283.705319, 11794.949515, 9850.965345, 5383.938005, 7393.028526, 9865.642845,
            9226.804608, 34028.973399, 34849.617798, 34629.806186
        ],
        "pair_id": [
            "ETHUSDT", "ETHUSDT", "ETHUSDT", "ETHUSDT", "ETHUSDT", "ETHUSDT", "ETHUSDT", "ETHUSDT", "ETHUSDT", "ETHUSDT",
            "ETHUSDT", "ETHUSDT", "BTCUSDT", "BTCUSDT", "BTCUSDT", "BTCUSDT", "BTCUSDT", "BTCUSDT", "BTCUSDT", "BTCUSDT",
            "BTCUSDT", "BTCUSDT", "BTCUSDT", "BTCUSDT"
        ]
    }
    candles_dates = [
        "2021-01-01T00:00:00.000", "2021-01-01T04:00:00.000", "2021-01-01T08:00:00.000", "2021-01-01T12:00:00.000", "2021-01-01T16:00:00.000",
        "2021-01-01T20:00:00.000", "2021-01-02T00:00:00.000", "2021-01-02T04:00:00.000", "2021-01-02T08:00:00.000", "2021-01-02T12:00:00.000",
        "2021-01-02T16:00:00.000", "2021-01-02T20:00:00.000", "2021-01-01T00:00:00.000", "2021-01-01T04:00:00.000", "2021-01-01T08:00:00.000",
        "2021-01-01T12:00:00.000", "2021-01-01T16:00:00.000", "2021-01-01T20:00:00.000", "2021-01-02T00:00:00.000", "2021-01-02T04:00:00.000",
        "2021-01-02T08:00:00.000", "2021-01-02T12:00:00.000", "2021-01-02T16:00:00.000", "2021-01-02T20:00:00.000"
    ]
    candles_df = pd.DataFrame(candles_data, index=pd.to_datetime(candles_dates))

    lending_data = {
        "lending_rates": [0.000250, 0.000250, 0.001045, 0.001045, 0.000300, 0.000300, 0.001045, 0.001045],
        "pair_id": ["ETH", "ETH", "USDT", "USDT", "BTC", "BTC", "USDT", "USDT"]
    }
    lending_dates = ["2020-12-31", "2021-01-01", "2020-12-31", "2021-01-01", "2020-12-31", "2021-01-01", "2020-12-31", "2021-01-01"]
    lending_rates_df = pd.DataFrame(lending_data, index=pd.to_datetime(lending_dates))

    if os.environ.get("GITHUB_ACTIONS", None) == "true":
        with patch(
            "tradingstrategy.binance.downloader.BinanceDownloader.fetch_candlestick_data"
        ) as mock_fetch_candlestick_data, patch(
            "tradingstrategy.binance.downloader.BinanceDownloader.fetch_lending_rates"
        ) as mock_fetch_lending_data:
            mock_fetch_candlestick_data.return_value = candles_df
            mock_fetch_lending_data.return_value = lending_rates_df

            universe = create_binance_universe(
                ["ETHUSDT", "BTCUSDT"],
                TIME_BUCKET,
                STOP_LOSS_TIME_BUCKET,
                START_AT,
                END_AT,
                include_lending=True,
            )
    else:
        universe = create_binance_universe(
            ["ETHUSDT", "BTCUSDT"],
            TIME_BUCKET,
            STOP_LOSS_TIME_BUCKET,
            START_AT,
            END_AT,
            include_lending=True,
        )


@pytest.mark.skipif(os.environ.get("GITHUB_ACTIONS", None) == "true" or os.environ.get("BINANCE_LENDING_DATA") == "false", reason="Github US servers are blocked by Binance with HTTP 451")
def test_binance_multi_pair():
    """Check multipair resampling works."""
    dataset = fetch_binance_dataset(
        ["ETHUSDT", "BTCUSDT"],
        candle_time_bucket=TimeBucket.d1,
        stop_loss_time_bucket=TimeBucket.h4,
        start_at=datetime.datetime(2020, 1, 1),
        end_at=datetime.datetime(2020, 2, 1),
        include_lending=False,
        force_download=True,
    )

    assert dataset.candles.iloc[0]["timestamp"].to_pydatetime() == datetime.datetime(2020, 1, 1)
    assert dataset.candles.iloc[-1]["timestamp"].to_pydatetime() == datetime.datetime(2020, 2, 1)

    eth_row = dataset.candles.iloc[0]
    assert eth_row["pair_id"] == 1
    assert eth_row.to_dict() == {'open': 129.16, 'high': 133.05, 'low': 128.68, 'close': 130.77, 'volume': 144770.52197, 'pair_id': 1, 'base_token_symbol': 'ETH', 'quote_token_symbol': 'USDT', 'exchange_slug': 'binance', 'chain_id': -1.0, 'fee': 0.0005, 'token0_address': '0x4b2d72c1cb89c0b2b320c43bb67ff79f562f5ff4', 'token1_address': '0x5b1a1833b16b6594f92daa9f6d9b7a6024bce9d0', 'token0_symbol': 'ETH', 'token1_symbol': 'USDT', 'token0_decimals': 18.0, 'token1_decimals': 18.0, 'timestamp': Timestamp('2020-01-01 00:00:00')}

    eth_row = dataset.candles.iloc[1]
    assert eth_row["pair_id"] == 1
    assert eth_row["open"] < 5000

    btc_row = dataset.candles.iloc[-1]
    assert btc_row["pair_id"] == 2
    assert btc_row.to_dict() == {'open': 9351.71, 'high': 9464.53, 'low': 9341.17, 'close': 9432.33, 'volume': 4624.992378, 'pair_id': 2, 'base_token_symbol': 'BTC', 'quote_token_symbol': 'USDT', 'exchange_slug': 'binance', 'chain_id': -1.0, 'fee': 0.0005, 'token0_address': '0x505e65d08c67660dc618072422e9c78053c261e9', 'token1_address': '0x5b1a1833b16b6594f92daa9f6d9b7a6024bce9d0', 'token0_symbol': 'BTC', 'token1_symbol': 'USDT', 'token0_decimals': 18.0, 'token1_decimals': 18.0, 'timestamp': Timestamp('2020-02-01 00:00:00')}


@pytest.mark.skipif(os.environ.get("GITHUB_ACTIONS", None) == "true", reason="Github US servers are blocked by Binance with HTTP 451")
def test_binance_timezone():
    """See timezone data is correct.

    - Check for UTC issues
    """

    strategy_universe = create_binance_universe(
        ["BTCUSDT", "ETHUSDT"],   # Binance internal tickers later mapped to Trading strategy DEXPair metadata class
        candle_time_bucket=TimeBucket.h1,
        stop_loss_time_bucket=None,
        start_at=datetime.datetime(2019, 1, 1),  # Backtest for 5 years data
        end_at=datetime.datetime(2019, 2, 1),
        include_lending=False
    )

    pair_desc = (ChainId.centralised_exchange, "binance", "BTC", "USDT")
    pair = strategy_universe.data_universe.pairs.get_pair_by_human_description(pair_desc)
    btc = strategy_universe.data_universe.candles.get_last_entries_by_pair_and_timestamp(pair, strategy_universe.end_at)
    assert btc.index[0] == pd.Timestamp("2019-01-01 00:00:00")


@pytest.mark.skipif(os.environ.get("GITHUB_ACTIONS", None) == "true", reason="Github US servers are blocked by Binance with HTTP 451")
def test_binance_resample_no_change():
    """The candle data should not change if the resample input and output frequencies are the same.

    - Load 1h stop loss base data

    - Resample to 1d

    - Resample to 1d again

    - The result should be the samme
    """

    strategy_universe = create_binance_universe(
        ["BTCUSDT", "ETHUSDT"],   # Binance internal tickers later mapped to Trading strategy DEXPair metadata class
        candle_time_bucket=TimeBucket.d1,
        stop_loss_time_bucket=TimeBucket.h1,
        start_at=datetime.datetime(2019, 1, 1),  # Backtest for 5 years data
        end_at=datetime.datetime(2019, 2, 1),
        include_lending=False
    )

    pair_desc = (ChainId.centralised_exchange, "binance", "BTC", "USDT")
    pair = strategy_universe.data_universe.pairs.get_pair_by_human_description(pair_desc)
    btc = strategy_universe.data_universe.candles.get_last_entries_by_pair_and_timestamp(pair, strategy_universe.end_at)

    assert btc.index[0] == pd.Timestamp('2019-01-01')
    assert btc.index[1] == pd.Timestamp('2019-01-02')
    btc_resampled_again = resample_candles(btc, TimeBucket.d1.to_pandas_timedelta())

    assert btc_resampled_again.index[0] == pd.Timestamp('2019-01-01')
    assert btc_resampled_again.index[1] == pd.Timestamp('2019-01-02')

    for i in range(0, 4):
        assert btc.iloc[i]["open"] == btc_resampled_again.iloc[i]["open"]
        assert btc.iloc[i]["close"] == btc_resampled_again.iloc[i]["close"]
        assert btc.iloc[i]["high"] == btc_resampled_again.iloc[i]["high"]
        assert btc.iloc[i]["low"] == btc_resampled_again.iloc[i]["low"]


@pytest.mark.skipif(os.environ.get("GITHUB_ACTIONS", None) == "true", reason="Github US servers are blocked by Binance with HTTP 451")
def test_binance_upsample_again():
    """Check that we can arrive to the same daily candles in two days

    - Load 1h, upsample 1d

    - Load 1h, upsample 8h, upsample 1d
    ."""

    start_at = datetime.datetime(2019, 1, 1)
    end_at = datetime.datetime(2019, 2, 1)

    strategy_universe_daily = create_binance_universe(
        ["BTCUSDT", "ETHUSDT"],   # Binance internal tickers later mapped to Trading strategy DEXPair metadata class
        candle_time_bucket=TimeBucket.d1,
        stop_loss_time_bucket=TimeBucket.h1,
        start_at=start_at,  # Backtest for 5 years data
        end_at=end_at,
        include_lending=False
    )

    strategy_universe_8h = create_binance_universe(
        ["ETHUSDT", "BTCUSDT"],
        candle_time_bucket=TimeBucket.h8,
        stop_loss_time_bucket=TimeBucket.h1,
        start_at=start_at,
        end_at=end_at,
        include_lending=False,
    )

    pair_desc = (ChainId.centralised_exchange, "binance", "BTC", "USDT")
    pair = strategy_universe_daily.data_universe.pairs.get_pair_by_human_description(pair_desc)
    btc_daily = strategy_universe_daily.data_universe.candles.get_last_entries_by_pair_and_timestamp(pair, end_at)

    pair_desc = (ChainId.centralised_exchange, "binance", "BTC", "USDT")
    pair = strategy_universe_8h.data_universe.pairs.get_pair_by_human_description(pair_desc)
    btc_8h = strategy_universe_8h.data_universe.candles.get_last_entries_by_pair_and_timestamp(pair, end_at)

    btc_resampled = resample_candles(btc_8h, pd.Timedelta(days=1))

    for i in btc_daily.index:
        assert btc_daily["open"][i] == btc_resampled["open"][i]
        assert btc_daily["close"][i] == btc_resampled["close"][i]
        assert btc_daily["high"][i] == btc_resampled["high"][i]
        assert btc_daily["low"][i] == btc_resampled["low"][i]
