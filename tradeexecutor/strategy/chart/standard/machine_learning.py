"""Machine learning model metrics for trading strategies."""
from typing import Literal, Tuple

import numpy as np
import pandas as pd

from tradeexecutor.strategy.chart.definition import ChartInput

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def match_predictions(
    close: pd.Series,
    predictions: pd.Series,
    original_true_values: pd.Series,
    original_close_price: pd.Series,
    prediction_value_type: Literal["sigmoid_prediction"] = "sigmoid_prediction",
    range: Tuple[pd.Timestamp, pd.Timestamp] | None = None,
    price_match_tolerance=0.02,
) -> pd.Series:
    """Check how well predictions match their original true value and and then our price true values.

    :param price_match_tolerance:
        If our price and the orignal price is this % apart assume them matching.

    :param range:
        The backtest range of the data, and the range for metrics are calculated.

        If not given use our close price range.
    """

    if range is None:
        range = (close.index[0], close.index[-1])

    close = close.loc[range[0]:range[1]]
    predictions = predictions.loc[range[0]:range[1]]
    original_close_price = original_close_price.loc[range[0]:range[1]]
    original_true_values = original_true_values.loc[range[0]:range[1]]

    match prediction_value_type:
        case "sigmoid_prediction":
            pred_binary = [1 if p >= 0.5 else 0 for p in predictions]
        case _:
            raise NotImplementedError(f"Unknown prediction value type: {prediction_value_type}")

    # True values are calculated as following
    # # --- Define Target Variable (`d`) ---
    # df['d'] = np.where(df['log_returns'].shift(-1) > 0, 1, 0)
    our_log_returns = np.log(close / close.shift(1))
    our_true_value = np.where(our_log_returns.shift(-1) > 0, 1, 0)

    # Calculate accuracy
    assert len(original_true_values) == len(pred_binary), f"Length mismatch, original true values: {len(original_true_values)} != prediction: {len(pred_binary)}"
    assert len(our_true_value) == len(original_true_values), f"Length mismatch, our true values: {len(our_true_value)} != original true values: {len(original_true_values)}"

    label_matching_proportion = (our_true_value == original_true_values).mean()

    # Calculate the absolute percentage difference between our prices and original prices
    price_difference = abs(close - original_close_price) / original_close_price
    prices_matches = price_difference <= price_match_tolerance

    prices_matches_proportion = prices_matches.mean()

    original_accuracy_score = accuracy_score(original_true_values.values, pred_binary)

    metrics = {
        "Original accuracy": original_accuracy_score,
        "Our prices start at": close.index[0],
        "Our prices end at": close.index[-1],
        "Predictions start at": predictions.index[0],
        "Predictions end at": predictions.index[-1],
        "Returns true labels matching": label_matching_proportion,
        f"Prices matching with {price_match_tolerance:%} tolerance": prices_matches_proportion,
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
