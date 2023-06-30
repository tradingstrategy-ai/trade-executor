import pytest
import pandas as pd
from tradeexecutor.visual.custom_indicators import calculate_on_balance_volume  # Import the function from your module

def test_simple_case():
    close_prices = pd.Series([10, 20, 30, 20, 10])
    volume = pd.Series([100, 200, 300, 200, 100])
    expected = pd.Series([100, 300, 600, 400, 300])
    assert (calculate_on_balance_volume(close_prices, volume) == expected).all()

def test_volume_assertion():
    close_prices = pd.Series([10, 20, 30, 20, 10])
    volume = pd.Series([100, -200, 300, 200, 100])
    with pytest.raises(AssertionError):
        calculate_on_balance_volume(close_prices, volume)

def test_length_assertion():
    close_prices = pd.Series([10, 20, 30, 20])
    volume = pd.Series([100, 200, 300, 200, 100])
    with pytest.raises(AssertionError):
        calculate_on_balance_volume(close_prices, volume)

def test_same_close_price():
    close_prices = pd.Series([10, 10, 10, 10, 10])
    volume = pd.Series([100, 200, 300, 200, 100])
    expected = pd.Series([100, 100, 100, 100, 100])
    assert (calculate_on_balance_volume(close_prices, volume) == expected).all()
