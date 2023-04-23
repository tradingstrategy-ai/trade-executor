"""Grid search tests."""
from pathlib import Path

import pandas as pd

from tradeexecutor.backtest.grid_search import prepare_grid_combinations


def test_prepare_grid_search_parameters():
    """Prepare grid search parameters."""

    parameters = {
        "stop_loss": [0.9, 0.95],
        "max_asset_amount": [3, 4],
        "momentum_lookback_days": ["7d", "14d", "21d"]
    }

    combinations = prepare_grid_combinations(parameters)
    assert len(combinations) == 2 * 2 * 3

    first = combinations[0]
    assert first.parameters[0].name == "max_asset_amount"
    assert first.parameters[0].value == 3

    assert first.get_state_path() == Path('max_asset_amount=3/momentum_lookback_days=7d/stop_loss=0.9')
