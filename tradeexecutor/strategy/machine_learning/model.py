"""Machine learning model lifecycle management."""
import pickle
from dataclasses import dataclass, field
import datetime
from pathlib import Path
from typing import TypedDict, Protocol

import pandas as pd

# We use CloudPickle to store functions needed to calculate inputs
from tradeexecutor.monkeypatch import cloudpickle_patch



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


class TrainingMetrics(TypedDict):
    accuracy: float
    f1_score: float


class ModelInputIndicators:

    def create_model_dataframe(
        self,
        price_df: pd.DataFrame) -> pd.DataFrame:
        pass



class TrainingFold:
    """One training fold"""
    fold_id: int
    training_start_at: datetime.datetime
    training_end_at: datetime.datetime
    test_start: datetime.datetime
    test_end: datetime.datetime
    training_metrics: TrainingMetrics

    @property
    def model_filename(self) -> str:
        """Get the model filename for this fold."""
        return f"fold_{self.fold_id}_model.keras"


@dataclass
class WalkForwardModel:
    """Manage the machine learning model history.

    - Walk-forward trained model with several variants, identified by walk-forward fold
    """

    #: Function that prepares the DataFrame
    model_input_calculation: ModelInputCalculationFunction

    #: Name of feature columns used.
    #:
    #: The order of this list is stable, because it will give the feature index in the model matrix.
    #:
    feature_columns: list[str]

    #: Metadata for the each fold.
    #:
    #: fold_id -> data
    folds: dict[int, TrainingFold] = field(default_factory=dict)

    def get_active_fold_for_timestamp(self, timestamp: datetime.datetime) -> TrainingFold | None:
        """Get the active training fold for a given timestamp."""
        for fold in reversed(self.folds):
            if fold.training_end_at >= timestamp:
                return fold
        return None

    def add_fold(
        self,
        model_storage_path: Path,
        model: "tensorflow.keras.models.Model",
        fold: TrainingFold,
    ) -> None:
        """Save the training fold to the model storage path."""
        if not model_storage_path.exists():
            model_storage_path.mkdir(parents=True, exist_ok=True)

        fold_path = model_storage_path / fold.model_filename
        model.save(fold_path)

        self.folds[fold.fold_id] = fold
        metadata_path = model_storage_path / f"walk_forward_metadata.pickle"

        pickle.dump(self, metadata_path.open("wb"))


class CachedModelLoader:
    """A model loader.

    - Can load multipe variations of the same model identified by walk-forward fold
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
        fold = self.folds[fold_id]
        model_path = self.model_storage_path / fold.model_filename
        if not model_path.exists():
            raise FileNotFoundError(f"Model file {model_path.resolve()} does not exist.")

        model = tensorflow.keras.models.keras.models.load_model(str(model_path))
        return model



