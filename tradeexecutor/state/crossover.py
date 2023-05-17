"""Provides functions to detect crossovers.

- crossover between two series
- crossover between a series and a constant value
"""

import pandas as pd
    
    
def crossover(
        series1: pd.Series, 
        series2: pd.Series,
        lookback_period: int = 2,
        must_return_index: bool = False,
) -> bool:
    """Detect if two series have cross over. To be used in decide_trades() 
    
    :param series1:
        A pandas.Series object.
        
    :param series2:
        A pandas.Series object.

    :lookback_period:
        The number of periods to look back to detect a crossover.
    
    :param must_return_index:
        If True, also returns the index of the crossover.
        
    :returns:
        bool. True if the series has crossed over the other series in the latest iteration, False otherwise.
        
    """

    assert type(series1) == type(series2) == pd.Series, "Series must be pandas.Series"
    
    if len(series1) == 1 or len(series2) == 1:
        return False
    
    lookback1 = series1.iloc[-lookback_period:]
    lookback2 = series2.iloc[-lookback_period:]

    # get index of cross
    cross_index = None
    has_crossed = False
    locked = True
    for i, (x,y) in enumerate(zip(lookback1, lookback2)):
        
        if x > y and not locked:
            has_crossed = True
            cross_index = -(len(lookback1) - i)
            locked = True
            # don't break since we want to get the latest cross
        
        # x must be below y before we can cross
        if x < y:
            locked = False

    if must_return_index:
        return has_crossed, cross_index
    else:
        return has_crossed


def crossunder(
        series1: pd.Series, 
        series2: pd.Series,
        lookback_period: int = 2,
        must_return_index: bool = False,
) -> bool:
    """Detect if two series have cross over. To be used in decide_trades() 
    
    :param series1:
        A pandas.Series object.
        
    :param series2:
        A pandas.Series object.

    :lookback_period:
        The number of periods to look back to detect a crossover.
    
    :param must_return_index:
        If True, also returns the index of the crossover.
        
    :returns:
        bool. True if the series has crossed over the other series in the latest iteration, False otherwise.
        
    """

    assert type(series1) == type(series2) == pd.Series, "Series must be pandas.Series"
    
    assert len(series1) >= lookback_period, "Series must be longer than or equal to the lookback period"
    assert len(series2) >= lookback_period, "Series must be longer than or equal to the lookback period"
    
    lookback1 = series1.iloc[-lookback_period:]
    lookback2 = series2.iloc[-lookback_period:]

    # get index of cross
    cross_index = None
    has_crossed = False
    locked = True
    for i, (x,y) in enumerate(zip(lookback1, lookback2)):
        
        if x < y and not locked:
            has_crossed = True
            cross_index = -(len(lookback1) - i)
            locked = True
            # don't break since we want to get the latest cross
        
        # x must be below y before we can cross
        if x > y:
            locked = False

    if must_return_index:
        return has_crossed, cross_index
    else:
        return has_crossed