"""Provides functions to detect crossovers.

- crossover between two series
- crossover between a series and a constant value
"""

import pandas as pd

def is_crossover_constant(series: pd.Series, value: float) -> bool:
    """Detect if a series has crossed over a constant value. To be used in decide_trades() 
    
    :param series:
        A pandas.Series object.
        
    :param value:
        A constant value to compare against.
        
    :returns:
        bool. True if the series has crossed over the value in the latest iteration, False otherwise.
        
    E.g. 
    
    decide_trades(...)
        ...
        if is_crossover_constant(fast_ema_series, 3300):
            visualisation.plot_indicator(timestamp, "Crossover 2", PlotKind.technical_indicator_overlay_on_detached, fast_ema_latest, colour="black", detached_overlay_name="Fast EMA", plot_shape=PlotShape.marker)
        ...
    """
    assert type(series) == pd.Series, "Series must be pandas.Series"
    assert type(value) in {int, float}, "Value must be int or float"
    
    latest = series.iloc[-1]
    
    if latest == value:
        return True
    elif len(series) == 1:
        return False
    
    previous = series.iloc[-2]
    
    return (
        latest == value or
        latest > value and previous < value
        or latest < value and previous > value
    )
    
def is_crossover(series1: pd.Series, series2: pd.Series) -> bool:
    """Detect if two series have cross over. To be used in decide_trades() 
    
    :param series1:
        A pandas.Series object.
        
    :param series2:
        A pandas.Series object.
        
    :returns:
        bool. True if the series has crossed over the other series in the latest iteration, False otherwise.
        
    E.g. 
    
    decide_trades(...)
        ...
        if is_crossover(fast_ema_series, slow_ema_series):
            visualisation.plot_indicator(timestamp, "Crossover 1", PlotKind.technical_indicator_overlay_on_detached, fast_ema_latest, colour="blue", detached_overlay_name="Fast EMA", plot_shape=PlotShape.marker)
        ...
    """
    assert type(series1) == type(series2) == pd.Series, "Series must be pandas.Series"
    
    s1_latest = series1.iloc[-1]
    s2_latest = series2.iloc[-1]
    
    if s1_latest == s2_latest:
        return True
    elif len(series1) == 1 or len(series2) == 1:
        return False
    
    s1_prev = series1.iloc[-2]
    s2_prev = series2.iloc[-2]

    return (
        (s1_latest > s2_latest
        and s1_prev < s2_prev)
        or (s1_latest < s2_latest
        and s1_prev > s2_prev)
    )