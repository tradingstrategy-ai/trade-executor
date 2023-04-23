



def test_prepare_grid_search_parameters():
    """Prepare grid search parameters."""

    parameters = {
        "stop_loss": [0.9, 0.95],
        "max_asset_amount": [3, 4],
        "momentum_lookback_days": [7, 14, 21],
    }


