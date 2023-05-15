"""Provides functions to detect crossovers.

- crossover between two series
- crossover between a series and a constant value
"""

import pandas as pd
import numpy as np

epsilon = 1e-10  # This is just an example. You may need to adjust this value.


def has_crossover_occurred(
        series1: pd.Series, 
        series2_or_int: pd.Series | int, 
        min_values_above_cross: int | None = 1,
        min_values_below_cross: int | None = 1,
        min_percent_diff1: float | None = 0,
        min_percent_diff2: float | None = 0,
        min_gradient: float | None = None,
    ) -> bool:
    """Detect if the first series has crossed above the second series/int. Typically usage will be in decide_trades() 
    
    :param series1:
        A pandas.Series object.
        
    :param series2_or_int:
        A pandas.Series object or an int.
    
    :param min_values_above_cross:
        The number of latest values of series1 that must have crossed above series2_or_int.

    :param min_values_below_cross:
        The minimum number of latest values before the crossover that must be below the crossover.
        
    :returns:
        bool. True if the series has crossed over the other series in the latest iteration, False otherwise.
        
    """

    series1 = _convert_list_to_series(series1)
    series2_or_int = _convert_list_to_series(series2_or_int)

    _validate_args(
        series1, 
        series2_or_int, 
        min_values_above_cross, 
        min_values_below_cross,
        min_percent_diff1,
        min_percent_diff2,
    )


    # Check for min_percent_difference
    if min_percent_diff1:

        if isinstance(series2_or_int, int):
            series2 = [series2_or_int] * len(series1)
        else:
            series2 = series2_or_int

        if not any(va for val1, val2 in zip(series1, series2)):
            return False
    
    if min_percent_diff1:

        if series1.iloc[-1] == 0:
            _epsilon = epsilon
        else:
            _epsilon = 0
        
        cross_magnitude1 = (series1.iloc[-1] - series1.iloc[0]) / (series1.iloc[0] + epsilon)

        if abs(cross_magnitude1) < min_percent_diff1:
            return False


    if min_percent_diff2:

        if isinstance(series2_or_int, int):
            raise ValueError("min_percent_diff2 cannot be used when series2_or_int is an int")



        if series2_or_int.iloc[-1] == 0:
            _epsilon = epsilon
        else:
            _epsilon = 0

        cross_magnitude2 = (series2_or_int.iloc[-1] - series2_or_int.iloc[0]) / (series2_or_int.iloc[0] + epsilon)

        if abs(cross_magnitude2) < min_percent_diff2:
            return False


    during_lookback1 = series1.iloc[-min_values_above_cross:]

    if isinstance(series2_or_int, int):
        # values in the lookback period
        during_lookback2 = [series2_or_int] * len(during_lookback1)

        cross_value = series2_or_int
    else:
        # values in the lookback period
        during_lookback2 = series2_or_int.iloc[-min_values_above_cross:]

        # average of latest value before the lookback period and first value in the lookback period
        cross_value = (series2_or_int.iloc[-min_values_above_cross-1] + series2_or_int.iloc[-min_values_above_cross]) / 2

    # values before the lookback period
    before_lookback1 = series1[:-min_values_above_cross]

    # Validation for min_values_below_cross
    if min_values_below_cross > len(before_lookback1):
        return False

    satisfied = [x < cross_value for x in before_lookback1[-min_values_below_cross:]]

    # make sure not fake crossover
    # at least `min_values_below_cross` values before the lookback period must be smaller than the cross_value
    # the point right before crossover must obviously be smaller than the cross_value
    if (
        sum(satisfied) < min_values_below_cross
        or before_lookback1.iloc[-1] > cross_value #  the point of crossover (can be equal to)
    ): 
        return False
    
    # Validation for min_gradient
    # calculate gradient
    if min_gradient is not None:
        num_values = min_values_below_cross + min_values_above_cross
        x = np.arange(num_values)
        y = _standardize(series1.iloc[-num_values:].values)
        slope, _ = np.polyfit(x, y, 1)
        if slope < min_gradient:
            return False
    
    

    return all(val1 > val2 for val1, val2 in zip(during_lookback1, during_lookback2))


def has_crossunder_occurred(series1: pd.Series, series2_or_int: pd.Series | int, min_values_above_cross: int | None = 1) -> bool:
    """Detect if the first series has crossed above the second series/int. Typically usage will be in decide_trades() 
    
    :param series1:
        A pandas.Series object.
        
    :param series2_or_int:
        A pandas.Series object or an int.
    
    :param min_values_above_cross:
        The number of latest values of series1 that must have crossed above series2_or_int.
        
    :returns:
        bool. True if the series has crossed over the other series in the latest iteration, False otherwise.
        
    """

    series1 = _convert_list_to_series(series1)
    series2_or_int = _convert_list_to_series(series2_or_int)

    _validate_args(series1, series2_or_int, min_values_above_cross)

    during_lookback1 = series1.iloc[-min_values_above_cross:]

    if isinstance(series2_or_int, int):
        # values in the lookback period
        during_lookback2 = [series2_or_int] * len(during_lookback1)
        
        # latest value before the lookback period
        cross_value = series2_or_int
    else:
        # values in the lookback period
        during_lookback2 = series2_or_int.iloc[-min_values_above_cross:]

        # average of latest value before the lookback period and first value in the lookback period
        cross_value = (series2_or_int.iloc[-min_values_above_cross-1] + series2_or_int.iloc[-min_values_above_cross]) / 2

    # values before the lookback period
    before_lookback1 = series1[:-min_values_above_cross]

    # make sure not fake crossover
    # at least 1 value before the lookback period must be greater than the cross_value
    if not (any(before_lookback1 > cross_value) and before_lookback1.iloc[-1] >= cross_value): 
        return False

    return all(val1 < val2 for val1, val2 in zip(during_lookback1, during_lookback2))


def _convert_list_to_series(series1):
    if isinstance(series1, list):
        series1 = pd.Series(series1)
    return series1


def _validate_args(
    series1,
    series2_or_int, 
    min_values_above_cross, 
    min_values_below_cross,
    min_percent_diff1,
    min_percent_diff2,
):
    assert type(series1) == pd.Series, "series1 must be a pandas.Series object"
    assert type(series2_or_int) in [
        pd.Series,
        int,
    ], "series2_or_int must be a pandas.Series object or an int"
    assert type(min_values_above_cross) == int and min_values_above_cross > 0, "min_values_above_cross must be a positive int"
    assert type(min_values_below_cross) == int and min_values_below_cross > 0, "min_values_below_cross must be a positive int"

    assert type(min_percent_diff1) in {float, int} and 0 <= min_percent_diff1 <= 1, "min_percent_above must be a number between 0 and 1"

    assert type(min_percent_diff2) in {float, int} and 0 <= min_percent_diff2 <= 1, "min_percent_below must be a number between 0 and 1


def _standardize(series):
    mean = series.mean()
    std_dev = series.std()
    standardized_series = (series - mean) / std_dev
    return standardized_series