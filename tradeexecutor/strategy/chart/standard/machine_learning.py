"""Machine learning model metrics for trading strategies."""
from typing import Literal

import pandas as pd

from tradeexecutor.strategy.chart.definition import ChartInput

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def match_predictions(
    close: pd.Series,
    predictions: pd.Series,
    original_true_values: pd.Series,
    prediction_value_type: Literal["sigmoid_prediction"] = "sigmoid_prediction"
) -> pd.Series:
    """Check how well predictions match their original true value and and then our price true values."""

    match prediction_value_type:
        case "sigmoid_prediction":
            pred_binary = [1 if p >= 0.5 else 0 for p in predictions]
        case _:
            raise NotImplementedError(f"Unknown prediction value type: {prediction_value_type}")

    # Calculate accuracy
    assert len(original_true_values) == len(pred_binary), f"Length mismatch: {len(original_true_values)} != {len(pred_binary)}"
    original_accuracy_score = accuracy_score(original_true_values, pred_binary)

    metrics = {
        "Original accuracy": original_accuracy_score,
        "Our prices start at": close.index[0],
        "Our prices end at": close.index[-1],
        "Predictions start at": predictions.index[0],
        "Predictions end at": predictions.index[-1],
    }

    # Must convert everything to str, or serialisation for cached indicators fails
    metrics = {k: str(v) for k, v in metrics.items()}

    series = pd.Series(metrics)
    return series


def prediction_metrics_table(
    input: ChartInput,
    predictions_metrics_indicator="predictions_metrics",
) -> pd.Series:
    """Table of machine learning predictions.

    - Print out how accurate the original predictions where
    - Print out how different the original true values (price changes)
      are from our price series

    :return:
        Human readable table
    """
    return input.strategy_input_indicators.get_indicator_series(predictions_metrics_indicator)
