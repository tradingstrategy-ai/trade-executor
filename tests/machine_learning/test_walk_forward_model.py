""""Walk-forward model testing."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tensorflow.keras.models import Model

from tradeexecutor.strategy.machine_learning.model import WalkForwardModel, CachedModelLoader, ModellingTooEarly, CachedPredictor, CachedPredictorOutput


@pytest.fixture()
def walk_forward_model_loader() -> CachedModelLoader:
    """Fixture for WalkForwardModel."""
    path = Path(os.path.dirname(__file__))
    return CachedModelLoader.load_folder(path)


@pytest.fixture()
def test_data() -> pd.DataFrame:
    """Load the Binance daily ETH/USDT data used in testing."""
    path = Path(os.path.dirname(__file__))
    df = pd.read_parquet(path / "binance-ethusdt-1d.parquet")
    # Because the model only works with a single pair,
    # don't confuse it with pair_id column
    del df["pair_id"]
    return df


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


def test_walk_forward_load_keras(walk_forward_model_loader: CachedModelLoader):
    """Load the Keras model for a fold."""
    walk_forward_model = walk_forward_model_loader.model
    fold_0 = walk_forward_model.folds[0]
    model = walk_forward_model_loader.get_cached_model_by_fold(fold_0)
    assert isinstance(model, Model)


def test_walk_forward_load_keras_by_timestamp(walk_forward_model_loader: CachedModelLoader):
    """Load the Keras model for a specific timestamp."""
    walk_forward_model = walk_forward_model_loader.model
    fold = walk_forward_model.get_active_fold_for_timestamp(pd.Timestamp('2020-06-04'))
    model = walk_forward_model_loader.get_cached_model_by_fold(fold)
    assert isinstance(model, Model)


def test_walk_forward_load_keras_by_timestamp_too_early(walk_forward_model_loader: CachedModelLoader):
    """Load the Keras model for a specific timestamp, but we do not have one yet."""
    walk_forward_model = walk_forward_model_loader.model
    with pytest.raises(ModellingTooEarly):
        walk_forward_model.get_active_fold_for_timestamp(pd.Timestamp('2000-01-01'))


def test_walk_forward_load_keras_by_timestamp_too_early(walk_forward_model_loader: CachedModelLoader):
    """Load the Keras model for a specific timestamp, but we do not have one yet."""
    walk_forward_model = walk_forward_model_loader.model
    with pytest.raises(ModellingTooEarly):
        walk_forward_model.get_active_fold_for_timestamp(pd.Timestamp('2000-01-01'))


def test_walk_forward_original_predictions(walk_forward_model_loader: CachedModelLoader):
    """Check we get datetime indexed predictions out from our training."""
    walk_forward_model = walk_forward_model_loader.model
    predictions = walk_forward_model.make_prediction_series_from_training()
    assert isinstance(predictions, pd.Series)
    assert isinstance(predictions.index, pd.DatetimeIndex)
    s = predictions
    continuous = s.index.is_monotonic_increasing and (s.index.diff().dropna() == pd.Timedelta(s.index.freq)).all()
    assert continuous


def test_extract_features_one_fold(
    walk_forward_model_loader: CachedModelLoader,
    test_data: pd.Series,
):
    """Extract features for a single fold.

    - Each fold has its own scaler, so we need to prepocess data by a fold
    """
    walk_forward_model = walk_forward_model_loader.model

    cached_predictor = CachedPredictor(walk_forward_model_loader)

    # Choose a range within fold 3
    range = pd.Timestamp('2023-06-06'), pd.Timestamp('2023-07-01')
    fold = walk_forward_model.get_active_fold_for_timestamp(range[0])

    features_df = walk_forward_model.prepare_input(test_data)

    assert "RSI_14" in features_df.columns
    assert "volume" in features_df.columns
    assert "volatility_10" in features_df.columns
    features_df_slice = features_df[range[0]:range[1]]

    model_input = cached_predictor.prepare_input_cached(
        fold=fold,
        features_df=features_df_slice,
    )

    assert model_input.fold_id == 3
    assert len(model_input.features_df) == len(features_df_slice)
    assert isinstance(model_input.sequences, np.ndarray)
    # nsamples_train, timesteps, n_features
    assert model_input.sequences.shape == (19, 7, 12)
    assert isinstance(model_input.x_scaled, np.ndarray)


def test_walk_forward_original_make_predictions_one_fold(
    walk_forward_model_loader: CachedModelLoader,
    test_data: pd.Series,
):
    """Create our own predictions based on historical data and compare them to the predictions made during training.

    - Do predictions within a single fold
    """

    walk_forward_model = walk_forward_model_loader.model
    original_predictions = walk_forward_model.make_prediction_series_from_training()

    cached_predictor = CachedPredictor(
        loader=walk_forward_model_loader,
    )

    # Choose a range that is within fold 3
    range = (pd.Timestamp('2023-06-10'), pd.Timestamp('2023-07-01'))

    features_df = walk_forward_model.prepare_input(test_data)
    features_df_slice = features_df[range[0]:range[1]]

    fold = walk_forward_model.get_active_fold_for_timestamp(range[0])

    model_input = cached_predictor.prepare_input_cached(
        fold=fold,
        features_df=features_df_slice,
    )

    model_output = cached_predictor.prepare_output_cached(
        fold,
        model_input,
    )
    assert isinstance(model_output, CachedPredictorOutput)

    # Check a single prediction first.
    # The first day in the range is not available
    # because you need to account for the LSTM buffer length.
    date = features_df_slice.index[8]
    original_prediction_in_range = original_predictions[date]
    our_prediction_in_range = model_output.predictions[date]

    assert original_prediction_in_range == our_prediction_in_range
    import ipdb ; ipdb.set_trace()
