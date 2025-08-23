""""Walk-forward model testing."""
import os

import pandas as pd
import pytest

from tradeexecutor.strategy.machine_learning.model import WalkForwardModel, CachedModelLoader


@pytest.fixture()
def walk_forward_model_loader() -> CachedModelLoader:
    """Fixture for WalkForwardModel."""
    path = os.path.dirname(__file__)
    return CachedModelLoader.load_folder(path)


def test_walk_forward_metadata(walk_forward_model_loader: CachedModelLoader):
    """Read WalkForwardModel metadata about how many folds we have."""
    walk_forward_model = walk_forward_model_loader.model
    assert isinstance(walk_forward_model, WalkForwardModel)
    assert len(walk_forward_model.folds) == 6
    assert isinstance(walk_forward_model.get_fold_metrics_table(), pd.DataFrame)
    assert isinstance(walk_forward_model.get_mean_fold_metrics(), pd.Series)


def test_walk_forward_load_fold(walk_forward_model_loader: CachedModelLoader):
    """Open one of the folds."""
    fold_0 = walk_forward_model_loader.load_model_by_fold(0)
    assert isinstance(fold_0.model, Model)
