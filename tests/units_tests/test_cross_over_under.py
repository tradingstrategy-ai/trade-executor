import pytest
import pandas as pd

from tradeexecutor.state.crossover import has_crossover_occurred, has_crossunder_occurred

def test_crossover_crossunder():
    # Test 1: Crossover and Crossunder occur.
    series1 = pd.Series([1, 2, 3, 4, 5, 4, 3, 2, 1])
    series2 = pd.Series([3, 3, 3, 3, 3, 3, 3, 3, 3])
    assert has_crossover_occurred(series1, series2) == True
    assert has_crossunder_occurred(series1, series2) == True

def test_no_crossover_crossunder():
    # Test 2: No crossover or crossunder occur.
    series1 = pd.Series([1, 2, 3, 4, 5])
    series2 = pd.Series([6, 7, 8, 9, 10])
    assert has_crossover_occurred(series1, series2) == False
    assert has_crossunder_occurred(series1, series2) == False

def test_not_enough_values_after():
    # Test 3: Crossover or crossunder occur but not enough values are above/below after the cross.
    series1 = pd.Series([1, 2, 3, 4, 5, 4, 3])
    series2 = pd.Series([4, 4, 4, 4, 4, 4, 4])
    assert has_crossover_occurred(series1, series2, min_values_above_cross=3) == False
    assert has_crossunder_occurred(series1, series2, min_values_below_cross=3) == False

def test_not_enough_buffer_after():
    # Test 4: Crossover or crossunder occur but the values are not sufficiently above/below after the cross.
    series1 = pd.Series([1, 2, 3, 4, 5, 4])
    series2 = pd.Series([4, 4, 4, 4, 4, 4])
    assert has_crossover_occurred(series1, series2, buffer_above_percent=0.5) == False
    assert has_crossunder_occurred(series1, series2, buffer_below_percent=0.5) == False

def test_not_enough_values_before():
    # Test 5: Crossover or crossunder occur but not enough values are below/above before the cross.
    series1 = pd.Series([2, 5, 4, 4, 2, 1])
    series2 = pd.Series([4, 4, 4, 4, 4, 4])
    assert has_crossover_occurred(series1, series2, min_values_below_cross=3) == False
    assert has_crossunder_occurred(series1, series2, min_values_above_cross=3) == False

def test_not_enough_buffer_before():
    # Test 6: Crossover or crossunder occur but the values are not sufficiently below/above before the cross.
    series1 = pd.Series([4, 5, 4, 3, 2, 1])
    series2 = pd.Series([4, 4, 4, 4, 4, 4])
    assert has_crossover_occurred(series1, series2, buffer_below_percent=0.5) == False
    assert has_crossunder_occurred(series1, series2, buffer_above_percent=0.5) == False

def test_not_steep_enough_gradient():
    # Test 7: Crossover or crossunder occur but the gradient is not steep enough.
    series1 = pd.Series([1, 2, 3, 4, 5, 4, 3, 2, 1])
    series2 = pd.Series([3, 3, 3, 3, 3, 3, 3, 3, 3])
    assert has_crossover_occurred(series1, series2, min_gradient=1) == False
    assert has_crossunder_occurred(series1, series2, min_gradient=1) == False

def test_fake_crossover_crossunder():
    # Test 8: Values dip/rise to be equal, but don't cross
    series1 = pd.Series([1,2,3,4,5,4,3,2,1])
    series2 = pd.Series([5,5,5,5,5,5,5,5,5])
    assert has_crossover_occurred(series1, series2) == False

    series1 = pd.Series([5,4,3,2,1,2,3,4,5])
    series2 = pd.Series([1,1,1,1,1,1,1,1,1])
    assert has_crossunder_occurred(series1, series2) == False
