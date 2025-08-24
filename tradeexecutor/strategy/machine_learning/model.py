"""Machine learning model lifecycle management."""
import pickle
from dataclasses import dataclass, field
import datetime
from importlib.metadata import metadata
from pathlib import Path
from typing import TypedDict, Protocol, Callable, TypeAlias

import cloudpickle
from numpy.typing import NDArray
import pandas as pd


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

    @property
    def model_filename(self) -> str:
        """Get the model filename for this fold."""
        return f"fold_{self.fold_id}_model.keras"

    def get_prediction_series(self) -> pd.Series:
        test_period_timestamps = pd.date_range(start=self.test_start_at, end=self.test_end_at, freq='D')
        assert len(test_period_timestamps) == len(self.predicted_labels), f"Length mismatch, timestamps: {len(test_period_timestamps)} != predicted labels: {len(self.predicted_labels)}"
        return pd.Series(self.predicted_labels, index=test_period_timestamps)


@dataclass
class WalkForwardModel:
    """Manage the machine learning model history.

    - Walk-forward trained model with several variants, identified by walk-forward fold
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

    def prepare_input(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare the price DataFrame for the model.

        - Calculates all indicators
        """
        assert len(price_df) >= self.minimum_input_rows, f"Not enough rows in price_df: {len(price_df)} < {self.minimum_input_rows}"
        all_feature_df = self.model_input_calculation(price_df)
        return all_feature_df[self.feature_columns]

    def get_fold_metrics_table(self) -> pd.DataFrame:
        """Get the training progress and prediction metrics for all folds in human-readable table."""
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
                "Accuracy": fold.training_metrics["accuracy"],
                "Precision": fold.training_metrics["precision"],
                "Recall": fold.training_metrics["recall"],
                "F1 score": fold.training_metrics["f1_score"],
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
        assert isinstance(path, Path), f"Not a Path: {path}"
        assert path.is_dir(), f"Not a directory: {path}"
        metadata_path = path / "walk_forward_metadata.pickle"
        assert metadata_path.exists(), f"Metadata file does not exist: {metadata_path}"
        model = cloudpickle.load(metadata_path.open("rb"))
        return CachedModelLoader(
            model=model,
            model_storage_path=path,
        )


@dataclass(slots=True)
class CachedPredictorInput:
    features_df: pd.DataFrame
    scaled_features_df: pd.DataFrame
    sequences: NDArray

    def __post_init__(self):
        assert isinstance(self.features_df, pd.DataFrame), "features_df must be a DataFrame"
        assert isinstance(self.scaled_features_df, pd.DataFrame), "scaled_featres_df must be a DataFrame"
        assert isinstance(self.features_df.index, pd.DatetimeIndex), f"features_df must have a DatetimeIndex, got {type(self.features_df.index)}"


@dataclass(slots=True)
class CachedPredictorOutput:
    shift: int
    raw_predictions: NDArray
    predictions: pd.Series


class CachedPredictor:
    """Make a predictions based on the price input data.

    - Cache the generation of features and sequencing them for LSTM
    """

    def __init__(self, loader: CachedModelLoader):
        self.loader = loader

        self.cached_model_input = {FoldId, CachedPredictorInput}
        self.cached_model_output = {FoldId, CachedPredictorInput}

    def prepare_input(
        self,
        model: WalkForwardModel,
        fold: TrainingFold,
        price_df: pd.DataFrame,
    ) -> CachedPredictorInput:
        """Calculate the features for the model, caching the result."""
        if price_df.empty:
            raise ValueError("price_df is empty")

        # Use the model input calculation function to get the features
        features_df = self.loader.model.prepare_input(price_df)

        scaled_df = fold.x_scaler.transform(features_df)

        x_sequenced = model.sequence(
            features_df,
            None,
            self.loader.model.lstm_sequence_length,
        )

        # Cache the features DataFrame
        return CachedPredictorInput(
            features_df=features_df,
            scaled_featres_df=scaled_df,
            sequences=x_sequenced,
        )

    def prepare_input_cached(
        self,
        model: "tensorflow.keras.models.Model",
        fold: TrainingFold,
        price_df: pd.DataFrame,
    ) -> CachedPredictorInput:
        """Prepare the input for the model, caching the result."""
        key = tuple(id(price_df), fold)

        if key in self.cached_model_input:
            return self.cached_model_input[key]

        # If not cached, calculate the input
        input = self.prepare_input(
            model,
            fold,
            price_df
        )
        self.cached_model_input[key] = input
        return input

    def prepare_output_cached(
        self,
        model: "tensorflow.keras.models.Model",
        model_input: CachedPredictorInput,
    ) -> CachedPredictorOutput:
        cache_key = id(model_input)

        if cache_key in self.cached_model_output:
            return self.cached_model_output[cache_key]

        raw_predictions = model.predict(model_input.sequences).flatten()

        input_index = model_input.features_df.index
        shift = self.loader.model.lstm_sequence_length
        start_at = input_index[0] + shift
        end_at = input_index[-1] + shift
        predictions_series = pd.Series(
            raw_predictions,
            index=pd.date_range(start=start_at, end=end_at, freq=input_index.freq),
        )
        output = CachedPredictorOutput(
            shift=shift,
            raw_predictions=raw_predictions,
            predictions=predictions_series,
        )
        return output

    def predict(
        self,
        price_df: pd.DataFrame,
        timestamp: datetime.datetime | pd.Timestamp,
    ):
        """Predict the future value using the model.

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
