import pandas as pd
import numpy as np
import plotly.graph_objects as go


def generate_bullish(dates: pd.DatetimeIndex, base_price=100, candle_fluctation = 2, drift_percentage=0.01):
    assert 0 < drift_percentage <= 1, 'drift must be greater than 0 for an uptrend'
    return _generate_trend(dates, base_price, candle_fluctation, drift_percentage)


def generate_bearish(dates: pd.DatetimeIndex, base_price=100, candle_fluctation = 2, drift_percentage=-0.01):
    assert -1 <= drift_percentage < 0, 'drift must be less than 0 for a downtrend'
    return _generate_trend(dates, base_price, candle_fluctation, drift_percentage)


def generate_sideways(dates: pd.DatetimeIndex, base_price=100, candle_fluctation = 1):
    return _generate_trend(dates, base_price, candle_fluctation, 0)


def _generate_trend(dates: pd.DatetimeIndex, base_price:float=100, candle_fluctation:float=2, drift_percentage:float=0.01):
    """Generate a random walk with a drift
    
    :param dates: 
        DatetimeIndex (list of dates) to generate the data for.
        Use pd.date_range() to generate the dates.
        See https://pandas.pydata.org/docs/reference/api/pandas.date_range.html"""

    # Set a seed for the random number generator for reproducibility
    np.random.seed(0)

    assert base_price > 0, 'base_price must be greater than 0'
    assert candle_fluctation > 0, 'candle_fluctation must be greater than 0'
    assert type(drift_percentage) in {float, int}, 'drift_percentage must be a float or an int'
    assert type(dates) == pd.DatetimeIndex, 'dates must be a DatetimeIndex. Use pd.date_range() to generate the dates. \nSee https://pandas.pydata.org/docs/reference/api/pandas.date_range.html'

    # Calculate the drift value
    drift = base_price * drift_percentage

    # Prepare lists to store the OHLC (Open, High, Low, Close) data
    opens = []
    highs = []
    lows = []
    closes = []

    # Generate the OHLC data
    for _ in dates:
        fluctuation_up = np.random.normal(0, candle_fluctation)  # Normally distributed random daily fluctuation
        fluctuation_down = np.random.normal(0, candle_fluctation)  # Normally distributed random daily fluctuation
        o = base_price
        h = o + abs(fluctuation_up)  # high is higher than open
        l = o - abs(fluctuation_down)  # low is lower than open
        c = o + fluctuation_up + drift  # close is open plus fluctuation and drift

        # Prevent negative prices
        c = max(c, 0)
        l = max(l, 0)

        opens.append(o)
        highs.append(h)
        lows.append(l)
        closes.append(c)

        base_price = c  # Update base price for the next day

    # Create a DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
    })

    # Set the timestamp as the index
    df.set_index('timestamp', inplace=True, drop=False)

    return df


def visualise_candles(df, title='Random Walk Candlestick Chart'):

    # Create a candlestick chart
    fig = go.Figure(data=[
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
        )
    ])

    # Set the title
    fig.update_layout(
        title=title,
        yaxis_title='Price',
        xaxis_title='Day',
    )

    fig.update_xaxes(rangeslider={"visible": False})

    return fig