import pandas as pd
from tradeexecutor.utils.increasing import is_monotonically_increasing, is_monotonically_decreasing
import pytest

@pytest.mark.parametrize("series, index, expected", [
    (pd.Series([1, 2, 3, 4, 5]), 3, True),
    (pd.Series([5, 4, 3, 2, 1]), 3, False),
    (pd.Series([1, 3, 2, 4, 3]), 3, False),
    (pd.Series([2, 2, 2]), 2, True),  # Testing constant values
    (pd.Series([]), 1, False),        # Testing empty series
    (pd.Series([1]), 1, True),         # Testing single element
    (pd.Series([1,2,1,3,4,5]), 5, False),

])
def test_is_monotonically_increasing(series, index, expected):
    assert is_monotonically_increasing(series, index) == expected

@pytest.mark.parametrize("series, index, expected", [
    (pd.Series([5, 4, 3, 2, 1]), 3, True),
    (pd.Series([1, 2, 3, 4, 5]), 3, False),
    (pd.Series([3, 2, 2, 1]), 3, True),
    (pd.Series([2, 2, 2]), 2, True),  # Testing constant values
    (pd.Series([]), 1, False),        # Testing empty series
    (pd.Series([1]), 1, True),         # Testing single element
    (pd.Series([5,4,5,3,2,1]), 5, False)
])
def test_is_monotonically_decreasing(series, index, expected):
    assert is_monotonically_decreasing(series, index) == expected
