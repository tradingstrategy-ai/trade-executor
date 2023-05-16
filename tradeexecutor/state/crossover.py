"""Provides functions to detect crossovers.

- crossover between two series
- crossover between a series and a constant value
"""

import pandas as pd
    
    
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
    
    if len(series1) == 1 or len(series2) == 1:
        return False
    
    s1_prev = series1.iloc[-2]
    s2_prev = series2.iloc[-2]

    if s1_latest > s2_latest and s1_prev <= s2_prev:
        return True
    else:
        return False
    

def is_crossunder(series1: pd.Series, series2: pd.Series) -> bool:
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
    
    if len(series1) == 1 or len(series2) == 1:
        return False
    
    s1_prev = series1.iloc[-2]
    s2_prev = series2.iloc[-2]

    if s1_latest < s2_latest and s1_prev >= s2_prev:
        return True
    else:
        return False