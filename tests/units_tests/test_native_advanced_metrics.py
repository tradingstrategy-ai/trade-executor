import pytest
import pandas as pd
import numpy as np
from tradeexecutor.statistics.native_advanced_metrics import _ratio_decorator, calculate_sharpe_ratio, calculate_sortino_ratio, calculate_profit_factor, _prepare_returns, _annualize_result


@pytest.fixture(autouse=True)
def set_seed():
    np.random.seed(0)

def test_ratio_decorator():
    def ratio_func(returns, risk_free_rate, periods, annualize):
        return returns.mean()

    decorated = _ratio_decorator(ratio_func)
    returns = pd.Series(np.random.randn(365) * 0.1)
    assert decorated(returns) == pytest.approx(-0.0012484491204292514)

    with pytest.raises(AssertionError):
        decorated(returns, -0.01)


def test_calculate_sharpe_ratio():
    returns = pd.Series(np.random.randn(365) * 0.1)
    sharpe_ratio = calculate_sharpe_ratio(returns)
    assert sharpe_ratio == pytest.approx(-0.012606149175810898)


def test_calculate_sortino_ratio():
    returns = pd.Series(np.random.randn(365) * 0.1)
    sortino_ratio = calculate_sortino_ratio(returns)
    assert sortino_ratio == pytest.approx(-0.01802853110842994)


def test_calculate_profit_factor():
    returns = pd.Series(np.random.randn(365) * 0.1)
    profit_factor = calculate_profit_factor(returns)
    assert profit_factor == pytest.approx(0.9692944238568301)


def test__prepare_returns():
    returns = pd.Series(np.random.randn(365) * 0.1)
    prepared_returns = _prepare_returns(returns)
    assert isinstance(prepared_returns, pd.Series), "Return type should be pd.Series."
    assert len(prepared_returns) == 365, "Return length should be the same as input."

    prepared_returns_with_risk_free_rate = _prepare_returns(returns, 0.01)
    assert all(prepared_returns_with_risk_free_rate == returns - 0.01), "Returns should be corrected by risk free rate."


def test__annualize_result():
    result = np.random.rand()
    annualized = _annualize_result(result, 365, True)
    assert isinstance(annualized, float), "Return type should be float."
    assert annualized == result * np.sqrt(1), "Annualization should not change the result when annualize=True."

    not_annualized = _annualize_result(result, 365, False)
    assert not_annualized == result * np.sqrt(365), "Result should be scaled by sqrt of periods when annualize=False."
