"""Provides functions to detect crossovers.

- crossover between two series
- crossover between a series and a constant value
"""

import pandas as pd
import numpy as np


epsilon = 1e-10  # This is just an example. You may need to adjust this value.


def has_crossover_occurred(
        series1: pd.Series, 
        series2: pd.Series | int, 
        min_values_above_cross: int | None = 1,
        min_values_below_cross: int | None = 1,
        buffer_above_percent: float | None = 0,
        buffer_below_percent: float | None = 0,
        min_gradient: float | None = None,
    ) -> bool:
    """Detect if the first series has crossed above the second series/int. Typically usage will be in decide_trades() 
    
    :param series1:
        A pandas.Series object.
        
    :param series2:
        A pandas.Series object.
    
    :param values_above_cross:
        The number of values of series1 directly after the cross that must be above the crossover.

    :param values_below_cross:
        The number of values of series1 directly before the cross that must be below the crossover.

    :param buffer_above_percent:
        The minimum percentage that the series must exceed the crossover value by for the crossover to be considered.
    
    :param buffer_below_percent:
        The minimum percentage difference the series must be below the crossover value by for the crossover to be considered.
        
    :returns:
        bool. True if the series has crossed over the other series in the latest iteration taking values_above_cross and values_below_cross into considertaion, False otherwise.
        
    """

    series1 = _convert_list_to_series(series1)
    series2 = _convert_list_to_series(series2)

    # trim series2 to the same length as series1 (using latest values)
    if len(series2) > len(series1):
        series2 = series2[-len(series1):]

    _validate_args(
        series1, 
        series2, 
        min_values_above_cross, 
        min_values_below_cross,
        buffer_above_percent,
        buffer_below_percent,
    )

    # find the latest index of crossover
    cross_index = None
    for i,x in enumerate(series1):
        if i == 0:
            continue

        if (series1[i-1] <= series2[i-1] and x > series2[i]):
            # cross_index will be first index after the crossover
            cross_index = i

            # cross value if value of second series directly after the crossover
            cross_value = series2[i]

            # to avoid division by zero
            if cross_value == 0:
                cross_value = epsilon

            # don't break here, we want the latest index of crossover

    if not cross_index:
        return False

    after_cross1 = series1[cross_index-1:]
    # min_values_above_cross
    if len(after_cross1) < min_values_above_cross:
        return False
    # buffer_above_percent
    if not (max(after_cross1) - cross_value)/cross_value >= buffer_above_percent:
        return False

    # min_values_above_cross
    after_cross1_cut = after_cross1[:min_values_above_cross]
    if any(x <= series2[cross_index] for x in after_cross1_cut):
        return False

    before_cross1 = series1[:cross_index-1]
    # make sure there are values below the cross
    if not any(x < cross_value for x in before_cross1):
        return False
    # min_values_below_cross
    if len(series1[:-cross_index]) < min_values_below_cross:
        return False
    # buffer_below_percent
    if not (cross_value - min(before_cross1))/cross_value >= buffer_below_percent:
        return False

    # min_values_below_cross
    before_cross1_cut = before_cross1[-min_values_below_cross:]
    if any(x >= series2[cross_index] for x in before_cross1_cut):
        return False

    # min_gradient
    if min_gradient is not None:
        x = np.arange(len(series1))
        y = _standardize(series1.values)
        slope, _ = np.polyfit(x, y, 1)
        if slope < min_gradient:
            return False

    # If we get here, then all conditions are met

    return True


def has_crossunder_occurred(
        series1: pd.Series, 
        series2: pd.Series | int, 
        min_values_above_cross: int | None = 1,
        min_values_below_cross: int | None = 1,
        buffer_above_percent: float | None = 0,
        buffer_below_percent: float | None = 0,
        min_gradient: float | None = None,
    ) -> bool:
    """Detect if the first series has crossed below the second series/int. 
    
    :param series1:
        A pandas.Series object.
        
    :param series2:
        A pandas.Series object.
    
    :param values_above_cross:
        The number of values of series1 directly before the cross that must be above the crossover.

    :param values_below_cross:
        The number of values of series1 directly after the cross that must be below the crossover.

    :param buffer_above_percent:
        The minimum percentage difference the series must be above the crossover value by for the crossover to be considered.
    
    :param buffer_below_percent:
        The minimum percentage that the series must fall below the crossover value by for the crossover to be considered.
        
    :returns:
        bool. True if the series has crossed under the other series in the latest iteration taking values_above_cross and values_below_cross into consideration, False otherwise.
    """
    
    series1 = _convert_list_to_series(series1)
    series2 = _convert_list_to_series(series2)

    # trim series2 to the same length as series1 (using latest values)
    if len(series2) > len(series1):
        series2 = series2[-len(series1):]

    _validate_args(
        series1, 
        series2, 
        min_values_above_cross, 
        min_values_below_cross,
        buffer_above_percent,
        buffer_below_percent,
    )

    # find the latest index of crossunder
    cross_index = None
    for i,x in enumerate(series1):
        if i == 0:
            continue

        if (series1[i-1] >= series2[i-1] and x < series2[i]):
            cross_index = i

            # cross value is value of second series directly after the crossunder
            cross_value = series2[i]

            # to avoid division by zero
            if cross_value == 0:
                cross_value = epsilon

    if not cross_index:
        return False

    after_cross1 = series1[cross_index-1:]
    if len(after_cross1) < min_values_below_cross:
        return False
    if not (cross_value - min(after_cross1))/cross_value >= buffer_below_percent:
        return False

    after_cross1_cut = after_cross1[:min_values_below_cross]
    if any(x >= series2[cross_index] for x in after_cross1_cut):
        return False

    before_cross1 = series1[:cross_index-1]
    # make sure there are values above the cross
    if not any(x > cross_value for x in before_cross1):
        return False
    if len(series1[:-cross_index]) < min_values_above_cross:
        return False
    if not (max(before_cross1) - cross_value)/cross_value >= buffer_above_percent:
        return False

    before_cross1_cut = before_cross1[-min_values_above_cross:]
    if any(x <= series2[cross_index] for x in before_cross1_cut):
        return False

    # min_gradient
    if min_gradient is not None:
        x = np.arange(len(series1))
        y = _standardize(series1.values)
        slope, _ = np.polyfit(x, y, 1)
        if slope > -min_gradient:
            return False

    # If we get here, then all conditions are met

    return True


def _convert_list_to_series(series):
    if isinstance(series, list):
        series = pd.Series(series)
    return series


def _validate_args(
    series1,
    series2, 
    min_values_above_cross, 
    min_values_below_cross,
    min_percent_diff1,
    min_percent_diff2,
):
    assert type(series1) == pd.Series, "series1 must be a pandas.Series object"
    assert type(series2) == pd.Series, "series2 must be a pandas.Series object"
    assert type(min_values_above_cross) == int and min_values_above_cross > 0, "min_values_above_cross must be a positive int"
    assert type(min_values_below_cross) == int and min_values_below_cross > 0, "min_values_below_cross must be a positive int"

    assert type(min_percent_diff1) in {float, int} and 0 <= min_percent_diff1 <= 1, "min_percent_above must be a number between 0 and 1"

    assert type(min_percent_diff2) in {float, int} and 0 <= min_percent_diff2 <= 1, "min_percent_below must be a number between 0 and 1"


def _standardize(series):
    mean = series.mean()
    std_dev = series.std()
    standardized_series = (series - mean) / std_dev
    return standardized_series
