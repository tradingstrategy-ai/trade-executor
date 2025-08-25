"""Machine learning model lifecycle management.

- Load walk-forward models with multiple folds and one trained model per each fold

Example LSTM sequencer function:

.. code-block:: python

    def create_sequences(X_df: pd.DataFrame, y_series: pd.Series | None, seq_length=7) -> tuple[NDArray, NDArray]:
        X_values = X_df.values

        if y_series is not None:
            y_values = y_series.values
        else:
            y_values = None

        # For each valid starting index i, create a sequence: [i : i + seq_length]
        X_seq = np.array([
            X_values[i : i + seq_length]
            for i in range(len(X_values) - seq_length)
        ])
        # The label for that sequence is y at index i + seq_length
        if y_series is not None:
            y_seq = y_values[seq_length:]
        else:
            y_seq = None

        return X_seq, y_seq

"""
import logging
import os
from dataclasses import dataclass, field
import datetime
from pathlib import Path
from typing import TypedDict, Protocol, Callable, TypeAlias, Iterable, Any

import cloudpickle
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from pandas.tseries.frequencies import to_offset


logger = logging.getLogger(__name__)


FoldId: TypeAlias = int


class ModellingTooEarly(Exception):
    """We tried to ask for a model for a data before first training finished"""



class PredictionSeriesMethod:
    """When we construct a series of predictions, which method we use."""

    #: Use the predictions we recorded during the training
    test_validation_during_training = "test_validation_during_training"

    #: Create
    test_validation_during_training = "test_validation_during_training"



class ModelInputCalculationFunction(Protocol):
    """Calculate inputs for the model based on price and other data.

    - The output DataFrame may contain excessive inputs which are not chosen to be used,
      and will be discareded, see :py:attr:`WalkForwardModel.feature_columns`
    """

    def __call__(
        self,
        price_df: pd.DataFrame,
    ) -> pd.DataFrame:
        pass


class Sequencer(Protocol):
    """Calculate inputs for the model based on price and other data.

    - The output DataFrame may contain excessive inputs which are not chosen to be used,
      and will be discareded, see :py:attr:`WalkForwardModel.feature_columns`
    """

    def __call__(
        self,
        X_df: pd.DataFrame,
        y_series: pd.Series | None,
    ) -> tuple[NDArray, NDArray | None]:
        pass


class TrainingMetrics(TypedDict):
    accuracy: float
    precision: float
    recall: float
    f1_score: float



@dataclass(slots=True)
class TrainingFold:
    """One training fold"""
    fold_id: int
    x_scaler: "sklearn.preprocessing.StandardScaler"
    training_start_at: datetime.datetime
    training_end_at: datetime.datetime
    training_rows: int
    test_start_at: datetime.datetime
    test_end_at: datetime.datetime
    test_rows: int
    training_metrics: TrainingMetrics

    #: True labels for test period, as used during the training.
    original_true_labels: NDArray

    #: Predicted labels for test period during the training.
    #:
    #: Original predictions made with the code during the training.
    #: These are stored for the internal assertion purposes.
    #: The model should always give the same predictions for the same input data.
    predicted_labels: NDArray

    #: The shape of X array used to train this fold originally
    x_shape: NDArray

    def __repr__(self):
        return f"<TrainingFold #{self.fold_id}, test range {self.test_start_at} - {self.test_end_at}>"

    def __hash__(self):
        return self.fold_id

    def __eq__(self, other):
        return self.fold_id == other.fold_id

    @property
    def model_filename(self) -> str:
        """Get the model filename for this fold."""
        return f"fold_{self.fold_id}_model.keras"

    def get_prediction_series(self) -> pd.Series:
        """Get the prediction series for testing we calculated during the model training."""
        test_period_timestamps = pd.date_range(start=self.test_start_at, end=self.test_end_at, freq='D')
        assert len(test_period_timestamps) == len(self.predicted_labels), f"Length mismatch, timestamps: {len(test_period_timestamps)} != predicted labels: {len(self.predicted_labels)}"
        return pd.Series(self.predicted_labels, index=test_period_timestamps)


@dataclass(slots=True)
class PreparedModelInput:
    """Features and sequences calculated for the whole input data.

    - Because sequencer loses N items by LSTM buffer size
      at the start of the each fold, we need to calculate sequences for overall data before giving it to the folds
    """

    features_df: pd.DataFrame
    sequences: NDArray

    def __post_init__(self):
        assert isinstance(self.features_df, pd.DataFrame), "features_df must be a DataFrame"
        assert isinstance(self.features_df.index, pd.DatetimeIndex), f"features_df must have a DatetimeIndex, got {type(self.features_df.index)}"
        assert isinstance(self.sequences, np.ndarray), f"sequences must be a NDArray, got {type(self.sequences)}"

    def clip(self, start_at: pd.Timestamp, end_at: pd.Timestamp) -> tuple[pd.DataFrame, NDArray]:
        """Clip the features DataFrame to the given range.

        - For the fold frange

        :param start_at:
            Inclusive start timestamp

        :param end_at:
            Exclusive end timestamp
        """
        index = self.features_df.index
        mask = (index >= start_at) & (index < end_at)
        clipped_df = self.features_df[mask]
        start_idx = index.get_loc(start_at)
        end_idx = index.get_loc(end_at)
        clipped_sequences = self.sequences[start_idx:end_idx]
        assert len(clipped_df) == len(clipped_sequences), f"Length mismatch, clipped_df: {len(clipped_df)} != clipped_sequences: {len(clipped_sequences)}"
        return clipped_df, clipped_sequences


@dataclass()
class WalkForwardModel:
    """Manage the machine learning model history.

    - Walk-forward trained model with several variants, identified by walk-forward fold

    .. note::

        Cannot use dataclass(slots) on this because it is imcompatible with cloudpickle
    """

    #: How many rows of dataframe we need to feed into :py:class:`ModelInputCalculationFunction` to ensure all technical indicators can be calculated
    minimum_input_rows: int

    #: Function that prepares the DataFrame
    model_input_calculation: ModelInputCalculationFunction

    #: How long LSTM sequences we make
    lstm_sequence_length: int

    #: create_sequence() function for LSMT
    sequencer: Callable

    #: Name of feature columns used.
    #:
    #: The order of this list is stable, because it will give the feature index in the model matrix.
    #:
    feature_columns: list[str]

    #: Metadata for the each fold.
    #:
    #: fold_id -> data
    folds: dict[int, TrainingFold] = field(default_factory=dict)

    def __post_init__(self):
        assert callable(self.model_input_calculation), f"Not callable: {self.model_input_calculation}"
        assert type(self.feature_columns) == list, f"feature_columns must be a list of strings, got {type(self.feature_columns)}: {self.feature_columns}"
        assert len(self.feature_columns) > 0

    def get_active_fold_for_timestamp(self, timestamp: datetime.datetime) -> TrainingFold:
        """Get the active training fold for a given timestamp.

        :raise ModellingTooEarly:
            We we ask for an too early
        """
        for fold in reversed(self.folds.values()):
            if fold.training_end_at <= timestamp:
                return fold

        raise ModellingTooEarly(f"No fold found for timestamp {timestamp}, first fold training ends {self.folds[0].training_end_at}")

    def save_fold(
        self,
        model_storage_path: Path,
        model: "tensorflow.keras.models.Model",
        fold: TrainingFold,
        verbose=True,
    ) -> None:
        """Save the training fold to the model storage path."""
        if not model_storage_path.exists():
            model_storage_path.mkdir(parents=True, exist_ok=True)

        fold_path = model_storage_path / fold.model_filename
        model.save(fold_path)

        self.folds[fold.fold_id] = fold
        metadata_path = model_storage_path / f"walk_forward_metadata.pickle"

        # Use Cloudpicle so we can save class instances and functions
        # created in notebook cells
        cloudpickle.dump(self, metadata_path.open("wb"))

        if verbose:
            model_size = fold_path.stat().st_size / (1024 * 1024)
            metadata_size = metadata_path.stat().st_size / (1024 * 1024)
            print(f"Saved fold {fold.fold_id} to {fold_path.resolve()}, {model_size:.2f} MB, metadata to {metadata_path.resolve()}, {metadata_size:.2f} MB")

    def prepare_input(self, price_df: pd.DataFrame) -> PreparedModelInput:
        """Prepare the price DataFrame for the model.

        - Uses the pickled `calculate_all()` feature extraction function from the original notebook
          that prepared the LSTM model
        - Calculates all indicators
        - Get columns we have marked as the model input features
        """
        assert len(price_df) >= self.minimum_input_rows, f"Not enough rows in price_df: {len(price_df)} < {self.minimum_input_rows}"
        all_feature_df = self.model_input_calculation(price_df)
        selected_features = all_feature_df[self.feature_columns]

        # We do not need to sequence y values,
        # because we are not training for a target
        sequences, _ = self.sequencer(
            selected_features,
            None,
            self.lstm_sequence_length,
        )

        return PreparedModelInput(
            features_df=selected_features,
            sequences=sequences,
        )

    def get_fold_metrics_table(self) -> pd.DataFrame:
        """Get the training progress and prediction metrics for all folds in human-readable table.

        - Used to diagnose the model trains and fits correctly
        """
        rows = []
        for fold_id, fold in self.folds.items():
            row = {
                "Fold ID": fold_id,
                "Training start": fold.training_start_at,
                "Training end": fold.training_end_at,
                "Training rows": fold.training_rows,
                "Test start": fold.test_start_at,
                "Test end": fold.test_end_at,
                "Test rows": fold.test_rows,
                "Test index start": fold.training_metrics["test_index_start"],
                "Accuracy": fold.training_metrics["accuracy"],
                "Precision": fold.training_metrics["precision"],
                "Recall": fold.training_metrics["recall"],
                "F1 score": fold.training_metrics["f1_score"],
                "Test seq start": ", ".join(f"{p:.3}" for p in fold.training_metrics["test_x_seq_start"]),
                "First predictions": ", ".join(f"{p:.3}" for p in fold.predicted_labels[0:3]),
                "Scaler means": ", ".join(f"{p:.3}" for p in fold.training_metrics["x_scaler_means"][0:3]),
            }
            rows.append(row)
        df = pd.DataFrame(rows)
        df = df.set_index("Fold ID")
        return df

    def get_mean_fold_metrics(self) -> pd.Series:
        """Get means across all folds."""
        table_df = self.get_fold_metrics_table()
        mean_series = table_df[["Accuracy", "Precision", "Recall", "F1 score"]].mean()
        return mean_series

    def make_prediction_series_from_training(self) -> pd.Series:
        """Compile the training-time predictions to easily accessible series.

        - Cross-validate original predictions with new predictions made by this Python code
        - Fast predictions for backtest access

        :return:
            DataFrame with DatetTimeIndex: prediction
        """
        return pd.concat([fold.get_prediction_series() for fold in self.folds.values()])


class CachedModelLoader:
    """A model loader.

    - A walk forward model is metadata file + associated Keras model per each fold
    - Can load multiple variations of the same model identified by walk-forward fold
    """

    def __init__(
        self,
        model: WalkForwardModel,
        model_storage_path: Path,
        too_old_threshold: datetime.timedelta = datetime.timedelta(days=365),
    ):
        self.model = model
        self.model_storage_path = model_storage_path

        #: fold-id -> model cahce
        self.cached_models: dict[int, "tensorflow.keras.models.Model"] = {}

    def get_cached_model_by_time(self, timestamp: datetime.datetime) -> "tensorflow.keras.models.Model":
        """Get live or backtesting model by a timestamp."""
        fold = self.model.get_active_fold_for_timestamp(timestamp)
        return self.get_cached_model_by_fold(fold)

    def get_cached_model_by_fold(self, fold: TrainingFold) -> "tensorflow.keras.models.Model":
        """Get live or backtesting model by a timestamp."""
        cached = self.cached_models.get(fold.fold_id)
        if not cached:
            cached = self.cached_models[fold.fold_id] = self.load_model_by_fold(fold.fold_id)
        return cached

    def load_model_by_time(
        self,
        timestamp: datetime.datetime,
    ) -> "tensorflow.keras.models.Model":
        """Get live or backtesting model by a timestamp."""
        fold = self.mdoel.get_active_fold_for_timestamp(timestamp)
        return self.load_model_by_fold(fold.fold_id)

    def load_model_by_fold(self, fold_id: int) -> "tensorflow.keras.models.Model":
        import tensorflow.keras.models
        fold = self.model.folds[fold_id]
        model_path = self.model_storage_path / fold.model_filename
        if not model_path.exists():
            raise FileNotFoundError(f"Model file {model_path.resolve()} does not exist.")

        model = tensorflow.keras.models.load_model(model_path, safe_mode=False)
        return model

    def predict(
        self,
        timestamp: datetime.datetime,
        price_df: pd.DataFrame,
    ) -> float:
        """Predict the next price.

        - Create
        """

        assert len(self.price_df) >= self.minimum_input_rows, f"Not enough rows in price_df: {len(price_df)} < {self.minimum_input_rows}"
        fold = self.get_cached_model_by_time(timestamp)
        model = fold.model
        features_df = self.model.prepare_input(price_df)

    @staticmethod
    def load_folder(path: Path) -> "CachedModelLoader":
        """Open a folder with fold files."""

        # Do whatever hardware hacks etc. we need to do
        # before loading any model
        _setup_keras()

        assert isinstance(path, Path), f"Not a Path: {path}"
        assert path.is_dir(), f"Not a directory: {path}"
        metadata_path = path / "walk_forward_metadata.pickle"
        assert metadata_path.exists(), f"Metadata file does not exist: {metadata_path}"
        model = cloudpickle.load(metadata_path.open("rb"))
        return CachedModelLoader(
            model=model,
            model_storage_path=path,
        )


@dataclass(slots=True, frozen=True)
class CachedFoldInput:
    """Cache input for a single fold."""

    #: Which walk-forward fold model this input is for
    fold_id: int

    #: Slice of features DataFrame used as an input
    #:
    #: This reference is copied across folds
    features_df: pd.DataFrame

    #: Sequenced LSTM features based on :py:attr:`scaled_features_df
    sequences: NDArray

    #: Scaled features, using the fold specific scaler.
    #:
    #: Reshaped to a suitable output for the model.
    x_scaled: NDArray

    def __repr__(self):
        return f"<CachedPredictorInput for fold {self.fold_id}, {self.features_df.index[0]} - {self.features_df.index[-1]}, {len(self.features_df)} rows>"

    def __post_init__(self):
        assert isinstance(self.features_df, pd.DataFrame), "features_df must be a DataFrame"
        # assert isinstance(self.scaled_features_df, pd.DataFrame), "scaled_featres_df must be a DataFrame"
        assert isinstance(self.features_df.index, pd.DatetimeIndex), f"features_df must have a DatetimeIndex, got {type(self.features_df.index)}"


@dataclass(slots=True)
class CachedFoldOutput:

    #: What was the LSTM shift used
    shift: int

    #: Raw predictions
    raw_predictions: NDArray

    #: Predictions keyed by the timestamp
    predictions: pd.Series


@dataclass(slots=True)
class NextPrediction:
    """Next prediction result."""
    timestamp: pd.Timestamp
    predicted_value: float
    fold: TrainingFold


class CachedPredictor:
    """Make a predictions based on the price input data.

    - Cache the generation of features and sequencing them for LSTM
    """

    def __init__(self, loader: CachedModelLoader):
        self.loader = loader
        self.cached_model_input: dict[Any, CachedFoldInput] = {}
        self.cached_model_output: dict[Any, CachedFoldOutput] = {}

    def calculate_features(
        self,
        price_df: pd.DataFrame,
    ) -> PreparedModelInput:
        """Calculate the features for the model."""
        assert not price_df.empty, f"price_df is empty: {price_df}"

        # Use the model input calculation function to get the features
        model_input = self.loader.model.prepare_input(price_df)
        return model_input

    def prepare_fold_input(
        self,
        fold: TrainingFold,
        model_input: PreparedModelInput,
        clip_to_fold: bool = True,
    ) -> CachedFoldInput:
        """Calculate the features for the model, caching the result.

        - Clip features to the fold range
        - Each fold has its own scaler, so we need to prepocess data by a fold
        - We *do not* check whether features_df is within the proper range of the fold

        :param clip_to_fold:
            Assume the input data is full backtest range, and we want to clip it to the fold range.

            If False, assume the input data is already clipped to some range that is not the fold,
            e.g. to be used in live prediction.
        """

        assert isinstance(model_input, PreparedModelInput), f"Not a PreparedModelInput: {type(model_input)}"
        model = self.loader.model
        features_df = model_input.features_df

        assert isinstance(model, WalkForwardModel), f"Not a WalkForwardModel: {type(model)}"
        assert isinstance(fold, TrainingFold), f"Not a TrainingFold: {type(fold)}"
        assert isinstance(features_df, pd.DataFrame), f"Not a DataFrame: {type(features_df)}"

        assert not features_df.empty, f"features_df is empty: {features_df}"

        fold_start_at = fold.test_start_at
        fold_end_at = fold.test_end_at

        if clip_to_fold:
            features_df, x_sequenced = model_input.clip(fold_start_at, fold_end_at)
        else:
            features_df = model_input.features_df
            x_sequenced = model_input.sequences

        our_features = list(features_df.columns)
        n_features = len(our_features)
        assert our_features == model.feature_columns, f"Feature columns do not match, expected: {model.feature_columns}, got: {our_features}"

        # Scale the output using the scaler from the training period
        scaler = fold.x_scaler
        x_reshaped = x_sequenced.reshape(-1, n_features)
        x_transformed = scaler.transform(x_reshaped)
        x_scaled = x_transformed.reshape(x_sequenced.shape)

        logger.info(
            "Prepared input sequence, features_df %s - %s, %d rows, x_scaled %s, x_sequenced %s, scaler means %s",
            features_df.index[0],
            features_df.index[-1],
            len(features_df),
            x_scaled.shape,
            x_sequenced.shape,
            scaler.mean_.tolist(),
        )

        # Cache the features DataFrame
        return CachedFoldInput(
            fold_id=fold.fold_id,
            features_df=features_df,
            x_scaled=x_scaled,
            sequences=x_sequenced,
        )

    def prepare_input_cached(
        self,
        fold: TrainingFold,
        model_input: PreparedModelInput,
        clip_to_fold: bool = True,
    ) -> CachedFoldInput:
        """Prepare the input for the model, caching the result.

        - Each fold has its own scaler, so we need to prepocess data by a fold
        """
        key = (id(model_input), fold)

        if key in self.cached_model_input:
            return self.cached_model_input[key]

        # If not cached, calculate the input
        input = self.prepare_fold_input(
            fold=fold,
            model_input=model_input,
            clip_to_fold=clip_to_fold,
        )
        self.cached_model_input[key] = input
        return input

    def prepare_output(
        self,
        fold: TrainingFold,
        model_input: CachedFoldInput,
        check_for_full_range: bool = True,
    ) -> CachedFoldOutput:
        """Make predictions for one input sequence.

        - Calculate LSTM output sequence for every input sequence

        :param check_for_full_range:
            Assume predictions and input data are for the full fold range.

            This is when the input data can be fully clipped to the fold range.
            We cannot do this during the live predictions.
        """

        walk_forward_model = self.loader.model
        model = self.loader.get_cached_model_by_fold(fold)

        raw_predictions = model.predict(model_input.x_scaled).flatten()

        input_index = model_input.features_df.index

        # Remap the timestamp of predictions with LSTM window shift
        shift = self.loader.model.lstm_sequence_length
        freq_str = pd.infer_freq(input_index)
        timedelta = to_offset(freq_str).base.delta
        start_at = input_index[0]
        end_at = input_index[-1]

        index = pd.date_range(start=start_at, end=end_at, freq=freq_str)

        if check_for_full_range:
            assert len(index) == len(raw_predictions), f"Length mismatch, index: {len(index)} != raw predictions: {len(raw_predictions)}\n" \
                f"index is {index[0]} -{index[-1]}, input index is {input_index[0]} - {input_index[-1]}, shift {shift}, freq {freq_str}, timedelta {timedelta}"

            predictions_series = pd.Series(
                raw_predictions,
                index=index,
            )
        else:

            # LSTM buffer ate some of our early rows
            assert len(model_input.x_scaled) == len(model_input.features_df) - walk_forward_model.lstm_sequence_length, f"Length mismatch, x_scaled: {len(model_input.x_scaled)} != features_df: {len(model_input.features_df)}"

            assert len(model_input.x_scaled)  == len(raw_predictions), f"Length mismatch, x_scaled: {len(model_input.x_scaled)} != raw predictions: {len(raw_predictions)}"
            shifted_index = index[shift:]
            predictions_series = pd.Series(
                raw_predictions,
                index=shifted_index,
            )

        output = CachedFoldOutput(
            shift=shift,
            raw_predictions=raw_predictions,
            predictions=predictions_series,
        )
        return output

    def prepare_output_cached(
        self,
        fold: TrainingFold,
        model_input: CachedFoldInput,
        check_for_full_range=True,
    ):
        key = fold, id(model_input)

        if key in self.cached_model_output:
            return self.cached_model_output[key]

        # If not cached, calculate the input
        output = self.prepare_output(
            fold=fold,
            model_input=model_input,
            check_for_full_range=check_for_full_range,
        )
        self.cached_model_output[key] = output
        return output

    def predict(
        self,
        price_df: pd.DataFrame,
        timestamp: datetime.datetime | pd.Timestamp,
    ):
        """Predict the future value using the model.

        - Make a prediction for a single value
        - Desigend to be used in live trading

        :param timestamp:
            The timestamp of the last price data row we have.

            Predict the next row.

            E.g. the timestamp of yesterday's close.
        """

        walk_forward_model = self.loader.model
        fold = walk_forward_model.get_active_fold_for_timestamp(timestamp)

        assert len(price_df) >= walk_forward_model.minimum_input_rows, f"Not enough rows in price_df: {len(price_df)} < {walk_forward_model.minimum_input_rows}"

        model = self.loader.get_cached_model_by_fold(fold)

        model_input = self.prepare_input_cached(
            model,
            fold,
            price_df,
        )

        model_output = self.prepare_output_cached(
            model,
            model_input,
        )

        return model_output.predictions[timestamp]

    def split_to_folds(
        self,
        price_df: pd.DataFrame,
    ) -> Iterable[tuple[TrainingFold, pd.DataFrame]]:
        """Take input data and split it to the corresponding fold we need to use to do predictions.

        - Take a slice of price data and match it to the fold that is able
          to make predictions at this time horizon

        :return:
            Tuple of corresponding fold and the slice of input price data we can use to make predictions within this fold.
        """
        assert isinstance(price_df.index, pd.DatetimeIndex)
        for current_fold, next_fold in _current_next(self.loader.model.folds.values()):

            if next_fold:
                slice = price_df.loc[current_fold.training_end_at:next_fold.training_end_at]
            else:
                # Final fold needs to predict any remaining data to
                # the heat death of the universe
                slice = price_df.loc[current_fold.training_end_at:]

            yield current_fold, slice

    def make_predictions(
        self,
        price_df: pd.DataFrame,
    ) -> pd.Series:
        """Make predictions for the whole price DataFrame.

        - Designed to be used in backtesting
        """

        prediction_series: pd.Series = []

        if price_df.empty:
            raise ValueError("price_df is empty")

        for fold, slice in self.split_to_folds(price_df):
            model = self.loader.get_cached_model_by_fold(fold)
            model_input = self.prepare_input_cached(
                model,
                fold,
                slice,
            )
            model_output = self.prepare_output_cached(
                model,
                model_input,
            )

            prediction_series.append(model_output.predictions)

        return pd.concat(prediction_series)

    def predict_next(self, price_df: pd.DataFrame) -> NextPrediction:
        """Predict the next value based on the given price DataFrame.

        - Designed to be used in live trading
        - Does a single prediction
        - Inefficient for doing multiple predictions

        :param price_df:
            Price DataFrame with DatetimeIndex, must contain at least minimum_input_rows rows

        :return:
            Predicted next value
        """
        if price_df.empty:
            raise ValueError("price_df is empty")

        assert isinstance(price_df.index, pd.DatetimeIndex), f"price_df must have a DatetimeIndex, got {type(price_df.index)}"
        index = price_df.index

        walk_forward_model = self.loader.model
        model_input = walk_forward_model.prepare_input(price_df)

        last_timestamp = index[-1]
        fold = walk_forward_model.get_active_fold_for_timestamp(last_timestamp)

        next_timestamp = pd.date_range(start=index[-1], periods=2, freq=index.freq)[-1]

        # Do sequencing
        fold_input = self.prepare_fold_input(
            fold=fold,
            model_input=model_input,
            clip_to_fold=False,
        )

        # Make predictions
        fold_output = self.prepare_output(
            fold,
            fold_input,
            check_for_full_range=False,
        )
        assert isinstance(fold_output, CachedFoldOutput)

        value = fold_output.predictions[last_timestamp]
        return NextPrediction(
            timestamp=last_timestamp,
            predicted_value=value,
            fold=fold,
        )


def _current_next(iterable: Iterable):
    """Allow us to iterate current and next values simultaneously."""
    it = iter(iterable)
    cur = next(it, None)
    if cur is None:
        return
    for nxt in it:
        yield cur, nxt
        cur = nxt
    yield cur, None  # last item has no next


_keras_setup_done = False

def _setup_keras():
    global _keras_setup_done
    if not _keras_setup_done:
        # TODO: Document what these do
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        os.environ['METAL_DEVICE_WRAPPER_SUPPRESS'] = '1'
        _keras_setup_done = True

