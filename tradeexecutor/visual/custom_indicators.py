"""Custom indicators for backtesting that are not found in pandas_ta."""

import pandas as pd

def calculate_on_balance_volume(close_prices: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculates the on balance volume from the close prices
    
    Logic for obv is as follows:
    - For the first data point, OBV = Volume.
    - For the rest of the data points 
        - if the current close price is higher than the previous close price, then OBV = previous OBV + Volume. 
        - If the current close price is lower than the previous close price, then OBV = previous OBV - Volume. 
        - If the current close price is equal to the previous close price, then OBV = previous OBV.
    """
    assert len(close_prices) == len(volume), "close prices and volume must have same length"
    assert all(
        item > 0 for item in volume
    ), "volume must be list of positive values representing total volume for each candle"

    obv = []
    for i in range(len(close_prices)):
        if i == 0:
            obv.append(volume[i])
            continue

        obv_latest = obv[-1]
        close_current = close_prices[i]
        close_prev = close_prices[i-1]

        volume_current = volume[i]

        if close_current > close_prev:
            obv.append(obv_latest + volume_current)
        elif close_current < close_prev:
            obv.append(obv_latest - volume_current)
        else:
            obv.append(obv_latest)

    return obv