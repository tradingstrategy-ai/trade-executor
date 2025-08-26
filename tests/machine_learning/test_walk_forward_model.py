""""Walk-forward model testing."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tensorflow.keras.models import Model

from tradeexecutor.strategy.machine_learning.model import WalkForwardModel, CachedModelLoader, ModellingTooEarly, CachedPredictor, CachedFoldOutput


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


def test_walk_forward_calculate_indicators(
    walk_forward_model_loader: CachedModelLoader,
    test_data: pd.Series,
):
    """Calculate indicators on test data.

    - Uses cloudpickled functions from notebook to calculate indicators
    """
    walk_forward_model = walk_forward_model_loader.model

    # Manually call the exposed function in ModelCalculation() class
    df = test_data
    df["adjusted_close"] = df["close"]
    df["RSI_calc_manual"] = walk_forward_model.model_input_calculation.compute_rsi(test_data)
    assert df.loc[pd.Timestamp('2023-09-01')]["RSI_calc_manual"] == pytest.approx(45.284224591163905)

    # Calculate all indicators
    model_input = walk_forward_model.prepare_input(df)
    features_df = model_input.features_df
    assert "RSI_14" in features_df.columns
    # Manually picked test entry
    assert features_df.loc[pd.Timestamp('2023-09-01')]["RSI_14"] == pytest.approx(45.284224591163905)


    # Momentum Indicators: RSI & MACD
    def _compute_rsi(df, column="adjusted_close", period=14):
        delta = df[column].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.convolve(gain, np.ones(period)/period, mode='same')
        avg_loss = np.convolve(loss, np.ones(period)/period, mode='same')
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _compute_rsi(df, column="adjusted_close", period=14):
        delta = df[column].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean()
        avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _compute_rsi_debug(df, column="adjusted_close", period=14):
        print("Input column:", df[column].head())
        delta = df[column].diff()
        print("Delta:", delta.head())
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        print("Gain:", gain[:5])
        print("Loss:", loss[:5])
        avg_gain = pd.Series(gain).rolling(window=period, min_periods=1).mean()
        avg_loss = pd.Series(loss).rolling(window=period, min_periods=1).mean()
        print("Avg Gain:", avg_gain.head())
        print("Avg Loss:", avg_loss.head())
        rs = avg_gain / avg_loss
        print("RS:", rs.head())
        rsi = 100 - (100 / (1 + rs))
        print("RSI:", rsi.head())
        return rsi

    def calculate_rsi_simple(df, column="adjusted_close", window=14):
        """RSI calculation using simple moving average."""
        prices = df[column]
        delta = prices.diff()
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)

        # Use simple moving average for the first calculation
        avg_gains = gains.rolling(window=window).mean()
        avg_losses = losses.rolling(window=window).mean()

        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))

        return rsi

    # Now do this with "simulated live data" which
    # only contains N rows that the model tells us we need
    # for the prediction
    end_at = pd.Timestamp('2023-09-01')
    start_at = end_at - pd.Timedelta(days=1) * walk_forward_model.minimum_input_rows
    new_df = test_data[start_at:end_at]
    # new_df["RSI_clipped"] = walk_forward_model.model_input_calculation.compute_rsi(new_df)
    new_df["RSI_clipped"] = calculate_rsi_simple(new_df)
    assert new_df.loc[pd.Timestamp('2023-09-01')]["RSI_clipped"] == pytest.approx(45.284224591163905)


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


def test_walk_forward_make_predictions_one_fold(
    walk_forward_model_loader: CachedModelLoader,
    test_data: pd.Series,
):
    """Create our own predictions based on historical data and compare them to the predictions made during training.

    - Do predictions within a single fold
    """

    walk_forward_model = walk_forward_model_loader.model
    original_predictions = walk_forward_model.make_prediction_series_from_training()

    model_input = walk_forward_model.prepare_input(test_data)

    cached_predictor = CachedPredictor(
        loader=walk_forward_model_loader,
    )

    # Choose a range that is within fold 3
    range = (pd.Timestamp('2023-05-27'), pd.Timestamp('2024-06-01'))

    features_df_slice = model_input.features_df[range[0]:range[1]]

    fold = walk_forward_model.get_active_fold_for_timestamp(range[0])

    # Check sequences at a specific date
    # loc = model_input.features_df.index.get_loc(pd.Timestamp('2023-06-03'))
    # assert loc == 1969
    # sequences = model_input.sequences[loc]
    # assert sequences[0][0] == pytest.approx(51.46211064621103)
    # assert features_df_slice.iloc[0]["RSI_14"] == pytest.approx(51.46211064621103)

    # Check model looks sane
    model = walk_forward_model_loader.get_cached_model_by_fold(fold)
    assert model.name == "LSTM_Attention_Model"

    # Do sequencing
    fold_input = cached_predictor.prepare_input_cached(
        fold=fold,
        model_input=model_input,
    )

    # Make predictions
    fold_output = cached_predictor.prepare_output_cached(
        fold,
        fold_input,
    )
    assert isinstance(fold_output, CachedFoldOutput)

    # Check a single prediction first.
    # The first day in the range is not available
    # because you need to account for the LSTM buffer length.
    date = features_df_slice.index[8]
    original_prediction_in_range = original_predictions[date]
    our_prediction_in_range = fold_output.predictions[date]

    # The prediction that was made on this specific date in the training notebook,
    # is the same which we get now when we run the prediction using a loaded model
    assert pytest.approx(original_prediction_in_range) == pytest.approx(our_prediction_in_range)


def test_walk_forward_original_predict_next(
    walk_forward_model_loader: CachedModelLoader,
    test_data: pd.Series,
):
    """Predict the next value."""
    walk_forward_model = walk_forward_model_loader.model

    # Do 3 predictions, so we have little bit more data to work with in this test.
    # Having 7+ days data allows us to spot any aligment issues
    predictions_wanted = pd.date_range(pd.Timestamp('2023-09-01'), pd.Timestamp('2023-09-14'))

    cached_predictor = CachedPredictor(
        loader=walk_forward_model_loader,
    )

    # Check what the inputs should be for the prediction,
    # from the model training time and then we can compare
    # them with the inputs we feed to the model during our test.
    fold = walk_forward_model.get_active_fold_for_timestamp(predictions_wanted[0])

    predictions_made = []

    for prediction_date in predictions_wanted:
        # Choose a date that is within fold 3
        # (Inclusive)
        end_at = prediction_date

        # Clip test data, so we cannot have forward-looking bias
        clipped_df = test_data[:end_at]

        # Discard extra data we are not going to use for the prediction
        start_at = end_at - pd.Timedelta(days=1) * walk_forward_model.minimum_input_rows
        clipped_df = clipped_df[start_at:]

        # Predict based on our input buffer
        next_prediction = cached_predictor.predict_next(clipped_df)

        # Check some of the inputs we calculated correctly,
        # values picked from manual validation
        model_input = next_prediction.model_input
        comp_df = model_input.features_df
        assert comp_df.loc[pd.Timestamp("2023-09-01")]["RSI_14"] == pytest.approx(42.272069)

        predictions_made.append(next_prediction)


    # Compare to the predicted value we calculated during the model training
    fold = walk_forward_model.get_active_fold_for_timestamp(next_prediction.timestamp)
    train_time_predictions = fold.get_prediction_series()

    train_time_predictions = train_time_predictions[predictions_wanted]

    df = pd.DataFrame({
        "train_time": train_time_predictions,
        "ours": pd.Series([p.predicted_value for p in predictions_made], index=predictions_wanted),
    })
    print(df)
    import ipdb ; ipdb.set_trace()

