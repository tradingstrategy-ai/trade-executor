from math import e, sqrt
from typing import List
import numpy as np
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy
from pandas_ta import ema, pvt, obv
from pandas.tseries.frequencies import to_offset
from pathlib import Path
import parallelbar
from tradingstrategy.timebucket import TimeBucket

from tradeexecutor.backtest.grid_search import perform_grid_search_parallelbar, prepare_grid_combinations

import tradeexecutor
import statsmodels as sm
from tradeexecutor.backtest.grid_search import GridCombination

valid_pairs_df = None


def calculate_ema_signal(df: pd.DataFrame):
    """Calculate different EMA, their diffs as a single signal

    The signal is based on the idea that when short moving average or price
    is moving faster than the long moving average of the price,
    the asset has momentu.

    - We calculate three different long EMA and short EMA differences
    - We normalise these difference over the price development of some duration
    - Each normalised difference is mapped to -1 ... 1
    - Signal is equally weighted sum of all SMA diffs

    Sources:

    - Momentum and trend following trading strategies for currencies and bitcoin
      by Janick Rohrbach, Silvan Suremann, Joerg Osterriede]

    - Dissecting Investment Strategies in the Cross Section and Time Series
      by Jamil Baza, Nick Grangerb, Campbell R. Harveyc, Nicolas Le Rouxd and Sandy Rattraye

    :return:
        DataFrame with added columns.

        Columns: signal, ema_signal_1, ema_long_1, ema_short_1, ema_diff_1...
        
    """

    # Did we manage to calculate all long/short EMA pairs for this pair
    # or is the data duration too short / not enough data
    enough_data = True

    for idx, ema_tuple in enumerate(short_long_ema_pairs, start=1):
        short_ema, long_ema = ema_tuple
        assert short_ema < long_ema
        df[f"ema_long_{idx}"] = ema(df["close"], length=long_ema) 
        df[f"ema_short_{idx}"] = ema(df["close"], length=short_ema)
        df[f"ema_diff_{idx}"] = df[f"ema_short_{idx}"] - df[f"ema_long_{idx}"]

        # Normalise EMA diff with 90 candles moving standard deviation        
        df[f"ema_diff_normalised_{idx}"] = df[f"ema_diff_{idx}"] / df[f"ema_diff_{idx}"].rolling(ema_diff_short_normalisation_period).std()
        # Normalise the normalised EMA diff with 280 candles moving standard deviation
        df[f"ema_diff_double_normalised_{idx}"] = df[f"ema_diff_normalised_{idx}"] / df[f"ema_diff_normalised_{idx}"].rolling(ema_diff_long_normalisation_period).std()

        # Apply response function to normalise signal on -1 ... +1 range
        if apply_response_function:
            # df[f"ema_diff_double_normalised_{idx}"] = df[f"ema_diff_normalised_{idx}"]
            # x exp(-x^2 / 4)
            # 0.858
            x = df[f"ema_diff_double_normalised_{idx}"]
            denominator = sqrt(2) * e**(-0.5)
            exponent = (x**2) / -4

            if not pd.isnull(exponent).all():
                exponented = np.exp(exponent)
                ranged_response = df[f"ema_signal_{idx}"] = df[f"ema_diff_double_normalised_{idx}"] * exponented / denominator
            else:
                # Could not calculate any of the exponents because all values in the series are NaN
                enough_data = False
                break

            assert ranged_response.max() < 1.1
            assert ranged_response.min() > -1.1

        else:
            # Pass normalised EMA diff as is
            # df[f"ema_signal_{idx}"] = df[f"ema_diff_double_normalised_{idx}"]
            df[f"ema_signal_{idx}"] = df[f"ema_diff_normalised_{idx}"]

    if enough_data:
        df["signal"] = 0
        # We could calculate partial results for all EMA pairs        
        for idx, ema_tuple in enumerate(short_long_ema_pairs, start=1):
            df["signal"] += df[f"ema_signal_{idx}"] 
        df["signal"] = df["signal"] / len(short_long_ema_pairs)
    else:
        df["signal"] = pd.NA

    # Trading day needs to use signal calculated from the previous day's data
    df["signal"] = df["signal"].shift(1)

    return df    


def calculate_signal_vs_profit(
    df: pd.DataFrame, 
    pair_id: str,
    short_long_ema_pairs: List[tuple], 
    profit_window: pd.Timedelta,
    time_frame: pd.Timedelta,        
) -> pd.DataFrame:
    """Calculate signals and profits for all incoming candles."""

    number_of_look_back_candles = lookback_window / time_frame
    number_of_look_forward_candles = profit_window / time_frame
    assert number_of_look_forward_candles > 0 and number_of_look_forward_candles.is_integer(), f"Could not calculate candle count that fits into profit window {profit_window} for data time frame {time_frame}"
    assert number_of_look_back_candles > 0 and number_of_look_forward_candles.is_integer(), f"Could not calculate candle count that fits into profit window {lookback_window} for data time frame {time_frame}"
    number_of_look_forward_candles = int(number_of_look_forward_candles)
    number_of_look_back_candles = int(number_of_look_back_candles)

    # Create entries for past price to be used for signal
    # and future price (used for the price correlation)
    momentum_offset = to_offset(lookback_window)
    profit_offset = to_offset(profit_window)

    # No data left after filtering
    if len(df.index) == 0:
        return pd.DataFrame()

    # Calculate trading pair age in a column
    start = df.index[0]

    # Remove first N days of trading history to filter out scam pump and dumps
    df = df.loc[df.index > start + min_age].copy()

    # No data left after filtering
    if len(df) < number_of_look_back_candles:
        return pd.DataFrame()
    
    df["age"] = df.index - start

    # Fix missing prices
    df["open"] = df["open"].replace(0, np.NaN)

    df["prev"] = df["open"].shift(number_of_look_back_candles)
    df["next"] = df["open"].shift(-number_of_look_forward_candles)

    # What is our predicted price
    df["price_diff"] = (df["next"] - df["open"]) / df["open"]  # Get the profit on the final day of profit window

    # Calculate signal from the past and price difference to the future
    df["momentum"] = (df["prev"] - df["open"]) / df["open"]

    #df["shifted_close"] = df["volume"].shift(1).rolling(obv_len).sum()
    #shifted_close = df.rolling(obv_len)
    #assert shifted_close["close"] is not None
    #assert shifted_close["volume"] is not None
    #import ipdb ; ipdb.set_trace()
    df["pvt"] = pvt(df["close"], df["volume"])
    df["obv"] = obv(df["close"], df["volume"])

    #df["obv"] = obv(shifted_close["close"], shifted_close["volume"])
    #shifted = shifted.iloc[-obv_len:-1]
    # df["obv"] = obv(shifted["close"], shifted["volume"])

    # Drop any momentum value that seems to be incorrect (more than 99% price movement)
    df["momentum"] = np.where(df["momentum"] > 0.99, 0, df["momentum"])
    df["momentum"] = np.where(df["momentum"] < -0.99, 0, df["momentum"])
    
    # df.loc[df["bullish"] & (df["momentum"] >= 0), "signal"] = df["momentum"]
    # df.loc[df["bearish"] & (df["momentum"] < 0), "signal"] = df["momentum"]    
    df["rolling_cum_volume"] = df["volume"].rolling(window=long_lookback_window).sum() 
    df["rolling_obv"] = df["obv"] - df["obv"].shift(periods=number_of_look_back_candles)
    df["rolling_pvt"] = df["pvt"] - df["pvt"].shift(periods=number_of_look_back_candles)
    # df["signal"] = df["rolling_pvt"].shift(1) / df["rolling_cum_volume"].shift(1)
    #df["signal"] = df["rolling_obv"].shift(1) / df["rolling_cum_volume"].shift(1)

    if signal_source == "weighted_ema":
        df = calculate_ema_signal(df)    
    elif signal_source == "momentum":
        df["signal"] = df["momentum"]
    else:
        raise RuntimeError(f"Figure out {signal_source}")
    
    # On negative signals, we go short.
    # On zero signal and lack of data set side to NA
    df["side"] = pd.NA
    
    df.loc[df["signal"] > zero_signal_cut_off, "side"] = "long"
    df.loc[df["signal"] < -zero_signal_cut_off, "side"] = "short"

    # Max and min price wihtin the profit window will determine the profit for longs and shorts respective
    df["max_future_price"] = df["close"].rolling(number_of_look_forward_candles).max().shift(-number_of_look_forward_candles) # Get the max profit on the profit window, assuming take profit %
    df["min_future_price"] = df["close"].rolling(number_of_look_forward_candles).min().shift(-number_of_look_forward_candles) # Get the max profit on the profit window, assuming take profit %    
    
    df["profit"] = df["price_diff"]
    df["profit_max"] = df["profit"]
    df["profit_abs"] = df["profit_max"].abs()
    # Calculate profit separately for longs and shorts
    # using Pandas Mask
    # https://stackoverflow.com/a/33770421/315168
    #
    # We calculate both profit after X time,
    # and also max take profit, assuming
    # we could do a perfect trailing stop loss
    #
    #longs = (df["side"] == "long")
    #shorts = (df["side"] == "short")
    #df.loc[longs, "profit"] = df["price_diff"]
    #df.loc[shorts, "profit"] = -df["price_diff"]
    #df.loc[longs, "profit_max"] = (df["max_future_price"] - df["open"]) / df["open"]  # Get the profit based on max price
    #df.loc[shorts, "profit_max"] = -(df["min_future_price"] - df["open"]) / df["open"]  # Get the profit based on max price

    #df.loc[longs, "desc"] = df.agg('{0[ticker]} long'.format, axis=1)
    #df.loc[shorts, "desc"] = df.agg('{0[ticker]} short'.format, axis=1)

    df["profit"] = df["profit"].fillna(0)
    df["profit_max"] = df["profit_max"].fillna(0)

    # On too low trading volume we zero out signal
    candle_volume_threshold = daily_volume_threshold * (time_frame / pd.Timedelta(days=1))
    volume_threshold_exceeded = df["volume"] >= candle_volume_threshold
    df["signal"] = np.where(volume_threshold_exceeded, df["signal"], np.NaN)
    df["profit"] = np.where(volume_threshold_exceeded, df["profit"], np.NaN)
    df["profit_max"] = np.where(volume_threshold_exceeded, df["profit_max"], np.NaN)
    
    return df


def calculate_signal_vs_price_for_pair(
    grouped_candles: DataFrameGroupBy,
    pair_id: str
) -> pd.DataFrame:
    """Calculate signal vs. profit ratio for an individual pair."""
    try:
        df = grouped_candles.get_group(pair_id).copy()
    except KeyError:
        # Scam pairs 
        return pd.DataFrame()
        
    df = calculate_signal_vs_profit(
        df,
        pair_id,
        lookback_window,
        profit_window,
        time_frame=time_bucket.to_pandas_timedelta(),
    )
    return df


def init_process(_valid_pairs):
    """Upload dataframe to each process when the grid search starts."""
    # Need to store dataframe somewhere
    global valid_pairs
    valid_pairs = _valid_pairs


def process_background_job(combination: GridCombination) -> tuple:
    # Create signal vs. price analysis for examined pairs and calculate correlation
    # Make a copy of DataFrame as it is mutated in-place

    global valid_pairs

    sma_short, sma_long, profit_window = combination.destructure()
            
    signal_vs_profit = [calculate_signal_vs_price_for_pair(df, sma_short, sma_long, profit_window) for df in valid_pairs]
    
    # Calculate linear regression for signal vs. profit 
    df = pd.concat(signal_vs_profit)
    df = df.dropna()
    df = df.loc[df["profit"] >= profit_threshold]
    longs = df.loc[df["side"] == "long"]
    shorts = df.loc[df["side"] == "short"]

    # https://stackoverflow.com/a/54685349/315168
    #regression = sm.OLS(df["profit_max"], df["signal"]).fit()
    long_regression = sm.OLS(longs["profit"], longs["signal"]).fit()
    short_regression = sm.OLS(shorts["profit"], shorts["signal"]).fit()
    return sma_short, sma_long, profit_window, long_regression, short_regression
        

def main():

    # Load preprocessed candle dataset
    # See fetch-binance-candles.py   
    time_bucket = TimeBucket.h1
    fpath = f"/tmp/binance-candles-{time_bucket.value}.parquet"
    all_candles_df = pd.read_parquet(fpath)

    interesting_pairs = {
        "ETHUSDT",
        "BTCUSDT",
        "LINKUSDT",
        "MATICUSDT",
        "AAVEUSDT",
        "COMPUSDT",
        "MKRUSDT",
        "BNBUSDT",
        "AVAXUSDT",
        "CAKEUSDT",
        "SNXUSDT",
        "CRVUSDT",
    }

    #all_candles_df = all_candles_df[all_candles_df["pair_id"].isin(interesting_pairs)]
    pair_ids = all_candles_df["pair_id"].unique()

    print(f"We are looking {len(pair_ids)} pairs")

    print("Preparing data")
    grouped_candles = all_candles_df.groupby("pair_id")

    valid_pairs = [grouped_candles.get_group(id) for id in grouped_candles.groups]

    print(f"Pairs with valid signal data {len(valid_pairs):,}")

    df = pd.concat(valid_pairs)    

    print(f"Total signal samples {len(df):,}")

    # This is the path where we keep the result files around
    storage_folder = Path(f"/tmp/binance-grid-search-{time_bucket.value}")

    parameters = {
        # "sma_short": [1, 2, 4, 8],
        # "long": [8, 16, 24, 32],
        #"profit_window": [2, 4, 8, 16],
        "sma_short": [4],
        "sma_slong": [24],
        "profit_window": [4],
    }

    combinations = prepare_grid_combinations(
        parameters,
        storage_folder,
    )

    grid_search_results = parallelbar.progress_imap(
        process_background_job,
        combinations,
        n_cpu=4,
        initializer=init_process,
        initargs=(valid_pairs,)
    )

    # What we have for our grid search result data
    columns = list(parameters.keys())
    columns += ["long_regression", "short_regression"]

    result_df = pd.DataFrame(grid_search_results, columns=columns)    


if __name__ == "__main__":
    main()