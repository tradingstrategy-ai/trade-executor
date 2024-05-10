"""Functions to determine whether a series is monotonically increasing or decreasing."""
import pandas as pd

def is_monotonically_increasing(series: pd.Series | list, index: int = 2) -> bool:
    """Check if a series is monotonically increasing.
    
    :param series:
        A pandas.Series object or a list.

    :param index:
        The number of latest elements to check for monotonically increasing series.
        
    :returns:
        bool. True if the series is monotonically increasing, False otherwise.
        
    """
    assert type(index) == int, "Index must be an integer."
    assert index > 0, "Index must be greater than 0."
    if len(series) < index:
        return False  # Not enough elements to determine
    return all(x <= y for x, y in zip(series[-index:-1], series[-index+1:]))

def is_monotonically_decreasing(series: pd.Series | list, index: int = 2) -> bool:
    """Check if a series is monotonically decreasing.
    
    :param series:
        A pandas.Series object or a list.

    :param index:
        The number of latest elements to check for monotonically decreasing series.
        
    :returns:
        bool. True if the series is monotonically decreasing, False otherwise.
        
    """
    assert type(index) == int, "Index must be an integer."
    assert index > 0, "Index must be greater than 0."
    if len(series) < index:
        return False  # Not enough elements to determine
    return all(x >= y for x, y in zip(series[-index:-1], series[-index+1:]))
