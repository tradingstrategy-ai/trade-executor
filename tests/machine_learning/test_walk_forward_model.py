""""Walk-forward model testing."""
import os
from pathlib import Path

import pandas as pd
import pytest

from tensorflow.keras.models import Model

from tradeexecutor.strategy.machine_learning.model import WalkForwardModel, CachedModelLoader


@pytest.fixture()
def walk_forward_model_loader() -> CachedModelLoader:
    """Fixture for WalkForwardModel."""
    path = Path(os.path.dirname(__file__))
    return CachedModelLoader.load_folder(path)


def test_walk_forward_metadata(walk_forward_model_loader: CachedModelLoader):
    """Read WalkForwardModel metadata about how many folds we have."""
    walk_forward_model = walk_forward_model_loader.model
    assert isinstance(walk_forward_model, WalkForwardModel)
    assert len(walk_forward_model.folds) == 4
    assert isinstance(walk_forward_model.get_fold_metrics_table(), pd.DataFrame)
    assert isinstance(walk_forward_model.get_mean_fold_metrics(), pd.Series)


def test_walk_forward_read_fold(walk_forward_model_loader: CachedModelLoader):
    """Open one of the folds."""
    walk_forward_model = walk_forward_model_loader.model
    fold_0 = walk_forward_model.folds[0]
    assert fold_0.fold_id == 0
    assert fold_0.training_rows == 862
    assert fold_0.test_start_at == pd.Timestamp('2020-06-03 00:00:00')
    assert "accuracy" in fold_0.training_metrics
    assert "price_open_end" in fold_0.training_metrics


def test_walk_forward_load_model(walk_forward_model_loader: CachedModelLoader):
    """Load the model for a fold."""
    walk_forward_model = walk_forward_model_loader.model
    fold_0 = walk_forward_model.folds[0]
    model = walk_forward_model_loader.get_cached_model_by_fold(fold_0)
    assert isinstance(model, Model)
