import pytest
import pandas as pd
import numpy as np
from tradeexecutor.statistics.native_advanced_metrics import calculate_sharpe_ratio, calculate_sortino_ratio, calculate_profit_factor, _prepare_returns, _annualize_result



@pytest.fixture(scope="module")
def returns():
    np.random.seed(0)
    return pd.Series(np.random.randn(365) * 0.1)


def test_calculate_sharpe_ratio(returns):
    sharpe_ratio = calculate_sharpe_ratio(returns)
    assert sharpe_ratio == pytest.approx(-0.24084014183815203)


def test_calculate_sortino_ratio(returns):
    sortino_ratio = calculate_sortino_ratio(returns)
    assert sortino_ratio == pytest.approx(-0.3444346032029643)


def test_calculate_profit_factor(returns):
    profit_factor = calculate_profit_factor(returns)
    assert profit_factor == pytest.approx(0.9692944238568301)


def test__prepare_returns(returns):
    prepared_returns = _prepare_returns(returns)
    assert isinstance(prepared_returns, pd.Series), "Return type should be pd.Series."
    assert len(prepared_returns) == 365, "Return length should be the same as input."
    assert prepared_returns[0] == pytest.approx(0.1764052345967664)
    assert prepared_returns.iloc[-1] == pytest.approx(-0.001568211160255477)


def test__prepare_returns_with_rf(returns):
    risk_free_rate = 0.05
    prepared_returns_with_risk_free_rate = _prepare_returns(returns, risk_free_rate)
    rf = np.power(1 + risk_free_rate, 1 / 365) - 1
    assert prepared_returns_with_risk_free_rate.equals(returns - rf), "Returns should be corrected by risk free rate."


def test__prepare_returns_with_rf_and_periods(returns):
    risk_free_rate = 0.01
    periods = 252
    prepared_returns_with_risk_free_rate = _prepare_returns(returns, risk_free_rate, periods)
    rf = np.power(1 + risk_free_rate, 1 / periods) - 1
    assert prepared_returns_with_risk_free_rate.equals(returns - rf), "Returns should be corrected by risk free rate."


def test__annualize_result():
    result = np.random.rand()
    annualized = _annualize_result(result)
    assert isinstance(annualized, float), "Return type should be float."
    assert annualized == result * np.sqrt(1), "Annualization should not change the result when annualize=True."

    not_annualized = _annualize_result(result, 365)
    assert not_annualized == result * np.sqrt(365), "Result should be scaled by sqrt of periods when annualize=False."
