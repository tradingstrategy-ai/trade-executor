""""Walk-forward model testing."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from cytoolz.itertoolz import first
from gmpy2.gmpy2 import next_above

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
    predictions = walk_forward_model.get_all_train_time_predictions()
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

    model_input = walk_forward_model.prepare_input(test_data)
    features_df = model_input.features_df

    assert "RSI_14" in features_df.columns
    assert "volume" in features_df.columns
    assert "volatility_10" in features_df.columns

    fold_input = cached_predictor.prepare_input_cached(
        fold=fold,
        model_input=model_input,
    )

    assert fold_input.fold_id == 3
    assert len(fold_input.features_df) == 364
    assert isinstance(fold_input.sequences, np.ndarray)
    # nsamples_train, timesteps, n_features
    assert fold_input.sequences.shape == (364, 7, 12)
    assert isinstance(fold_input.x_scaled, np.ndarray)


def test_walk_forward_make_predictions_one_fold(
    walk_forward_model_loader: CachedModelLoader,
    test_data: pd.Series,
):
    """Create our own predictions based on historical data and compare them to the predictions made during training.

    - Do all predictions within a single fold
    """

    walk_forward_model = walk_forward_model_loader.model
    original_predictions = walk_forward_model.get_all_train_time_predictions()

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

    # Check a single prediction
    date = features_df_slice.index[4]
    original_prediction_in_range = original_predictions[date]
    our_prediction_in_range = fold_output.predictions[date]

    # The prediction that was made on this specific date in the training notebook,
    # is the same which we get now when we run the prediction using a loaded model
    eps = 0.01
    pd.testing.assert_series_equal(
        pd.Series(original_prediction_in_range, index=features_df_slice.index),
        pd.Series(our_prediction_in_range, index=features_df_slice.index),
        check_exact=False,
        atol=eps,
        rtol=0,
        check_names=False,
    )


def test_walk_forward_predict_next(
    walk_forward_model_loader: CachedModelLoader,
    test_data: pd.Series,
):
    """Predict the next value.

    - Also checks for lookahead bias (unintended consequence)
    - https://github.com/FranQuant/AI-based-Trading-Strategies/issues/10
    """
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
        assert comp_df.loc[pd.Timestamp("2023-09-01")]["RSI_14"] == pytest.approx(45.284224591163905)

        predictions_made.append(next_prediction)


    # Compare to the predicted value we calculated during the model training
    fold = walk_forward_model.get_active_fold_for_timestamp(next_prediction.timestamp)
    train_time_predictions = fold.get_prediction_series()

    train_time_predictions = train_time_predictions[predictions_wanted]

    df = pd.DataFrame({
        "train_time": train_time_predictions,
        "ours": pd.Series([p.predicted_value for p in predictions_made], index=predictions_wanted),
    })

    check_date = pd.Timestamp("2023-09-01")
    assert df.loc[check_date]["train_time"] == pytest.approx(0.5078742508888245)
    assert df.loc[check_date]["ours"] == pytest.approx(0.5078742508888245)

    # Check our predictions with cut data are the same as within the original
    # notebook backtest
    eps = 0.01
    pd.testing.assert_series_equal(
        df["train_time"],
        df["ours"],
        check_exact=False,
        atol=eps,
        rtol=0,
        check_names=False,
    )


def test_walk_forward_predict_full_backtest(
    walk_forward_model_loader: CachedModelLoader,
    test_data: pd.Series,
):
    """Predict all values for the backtest duration

    - Cross fold prediction time series generation\
    """
    walk_forward_model = walk_forward_model_loader.model

    price_df = test_data

    cached_predictor = CachedPredictor(
        loader=walk_forward_model_loader,
    )

    train_time_predictions = walk_forward_model.get_all_train_time_predictions()

    # Do debug checks we have not accidentally missed a timestamp
    gaps = _find_datetime_index_gaps(train_time_predictions)
    assert len(gaps) == 0
    monotonic_index_issues = _find_monotonic_increase_violations(train_time_predictions.index)
    assert len(monotonic_index_issues) == 0
    assert train_time_predictions.index.is_monotonic_increasing

    # Check that we can assign on our input prices to fold timestamps correctly
    data = {f:s for f, s in cached_predictor.split_to_folds(price_df)}
    assert len(data) == 4
    fold_iter = iter(data.keys())
    first_fold = next(fold_iter)
    second_fold = next(fold_iter)
    assert first_fold.test_rows == 365
    assert first_fold.test_start_at == pd.Timestamp('2020-06-03 00:00:00')
    assert first_fold.test_end_at == pd.Timestamp('2021-06-02 00:00:00')
    assert first_fold.training_start_at == pd.Timestamp('2018-01-23 00:00:00')
    assert first_fold.training_end_at == pd.Timestamp('2020-06-02 00:00:00')
    assert second_fold.test_start_at == pd.Timestamp('2021-06-03 00:00:00')
    assert second_fold.test_end_at == pd.Timestamp('2022-06-02 00:00:00')

    # Check that we allocate the correct slide from model input data for this fold
    series_iter = iter(data.values())
    first_input_series = next(series_iter)
    assert first_input_series.index[0] == first_fold.test_start_at
    assert first_input_series.index[-1] == first_fold.test_end_at
    assert len(first_input_series) == first_fold.test_rows
    second_input_series = next(series_iter)
    assert second_input_series.index[0] == second_fold.test_start_at
    assert second_input_series.index[-1] == second_fold.test_end_at
    assert len(second_input_series) == 365

    # Predict for all folds
    predicted_labels_ours_series = cached_predictor.make_predictions(price_df)

    assert len(predicted_labels_ours_series) == len(train_time_predictions)

    df = pd.DataFrame({
        "train_time": train_time_predictions,
        "ours": predicted_labels_ours_series,
    })

    import ipdb ; ipdb.set_trace()

    # Check our predictions with cut data are the same as within the original
    # notebook backtest
    eps = 0.01
    pd.testing.assert_series_equal(
        df["train_time"],
        df["ours"],
        check_exact=False,
        atol=eps,
        rtol=0,
        check_names=False,
    )


def _find_datetime_index_gaps(s: pd.Series | pd.DataFrame, freq: str | None = None) -> pd.DataFrame:
    idx = pd.DatetimeIndex(s.index)
    if len(idx) < 2:
        return pd.DataFrame(columns=["pos", "timestamp", "prev_timestamp", "delta", "issue"])

    d = pd.Series(idx).diff()  # Timedelta between consecutive stamps
    issues = []

    # Duplicates
    dup_locs = np.flatnonzero(idx.duplicated())
    for i in dup_locs:
        issues.append((i, idx[i], idx[i - 1], d.iloc[i], "duplicate"))

    # Non‑increasing (<= 0 delta)
    non_inc_locs = np.flatnonzero(d <= pd.Timedelta(0))
    for i in non_inc_locs:
        if i == 0:
            continue
        issues.append((i, idx[i], idx[i - 1], d.iloc[i], "non-increasing"))

    # Determine expected step
    expected = None
    if freq is not None:
        expected = (pd.date_range("2000-01-01", periods=2, freq=freq)[1] - pd.Timestamp("2000-01-01"))
    else:
        inferred = idx.freq or idx.inferred_freq
        if inferred is not None:
            expected = (pd.date_range("2000-01-01", periods=2, freq=inferred)[1] - pd.Timestamp("2000-01-01"))
        else:
            pos = d[d > pd.Timedelta(0)]
            if not pos.empty:
                expected = pos.mode().iloc[0]

    # Gaps larger than expected step
    if expected is not None:
        gap_locs = np.flatnonzero(d > expected)
        for i in gap_locs:
            issues.append((i, idx[i], idx[i - 1], d.iloc[i], f"gap>(expected {expected})"))

    out = pd.DataFrame(issues, columns=["pos", "timestamp", "prev_timestamp", "delta", "issue"])
    return out.sort_values("pos").reset_index(drop=True)



def _find_monotonic_increase_violations(idx: pd.DatetimeIndex) -> pd.DataFrame:
    idx = pd.DatetimeIndex(idx)
    if len(idx) < 2:
        return pd.DataFrame(columns=["pos", "timestamp", "prev_timestamp", "delta"])

    deltas = pd.Series(idx).diff()
    bad = np.flatnonzero(deltas < pd.Timedelta(0))  # use <= for strictly increasing

    return pd.DataFrame({
        "pos": bad,
        "timestamp": idx[bad],
        "prev_timestamp": idx[bad - 1],
        "delta": deltas.iloc[bad].values,
    }).reset_index(drop=True)