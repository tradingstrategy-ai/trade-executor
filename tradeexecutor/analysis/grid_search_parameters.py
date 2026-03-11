"""Analyse grid search and optimiser parameter importance.

Provide visual and statistical analysis of how strategy parameters
affect backtest performance metrics (CAGR, Sharpe, drawdown, etc.).

Functions follow the standard analysis module pattern:

- Accept a ``printer`` callback for status messages
- Return objects (figures, dataframes) without rendering them directly
- The caller (typically a notebook cell) is responsible for displaying results

Example usage in a notebook:

.. code-block:: python

    from tradeexecutor.analysis.grid_search_parameters import (
        analyse_decision_tree,
        analyse_feature_importance,
        analyse_parameter_pair_heatmaps,
        analyse_parameter_clusters,
        analyse_parallel_coordinates,
    )

    fig, tree = analyse_decision_tree(df, analysis_metric="CAGR")
    fig.show()
"""

from typing import Callable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from matplotlib.figure import Figure as MatplotlibFigure
from sklearn.preprocessing import LabelEncoder


def _prepare_parameter_data(
    df: pd.DataFrame,
    analysis_metric: str = "CAGR",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, list[str]]:
    """Prepare grid search results for sklearn-based analysis.

    - Resets the multi-index to flat columns
    - Extracts parameter column names
    - Label-encodes any object (categorical) columns for sklearn compatibility
    - Splits into feature matrix X and target vector y

    :param df:
        Grid search results DataFrame with parameter multi-index.

    :param analysis_metric:
        Column name of the performance metric to analyse.

    :return:
        Tuple of (parameter_df, X, y, parameter_names) where
        parameter_df is the full reset-index DataFrame,
        X is the encoded feature matrix,
        y is the target series,
        parameter_names is the list of parameter column names.
    """
    parameter_names = list(df.index.names)
    parameter_df = df.reset_index()

    X = parameter_df[parameter_names].copy()
    y = parameter_df[analysis_metric]

    for col in X.columns:
        if X[col].dtype == "object":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

    return parameter_df, X, y, parameter_names


def analyse_decision_tree(
    df: pd.DataFrame,
    analysis_metric: str = "CAGR",
    max_depth: int = 4,
    figsize: tuple[int, int] = (20, 10),
    printer: Callable = print,
) -> tuple[MatplotlibFigure, "DecisionTreeRegressor"]:
    """Train a decision tree on grid search parameters and visualise it.

    Shows which parameter thresholds most strongly partition performance,
    making it easy to spot the dominant drivers at a glance.

    :param df:
        Grid search results DataFrame with parameter multi-index.

    :param analysis_metric:
        Column name of the performance metric to analyse (e.g. "CAGR", "Sharpe").

    :param max_depth:
        Maximum tree depth — deeper trees capture more interactions
        but are harder to read.

    :param figsize:
        Matplotlib figure size as (width, height) in inches.

    :param printer:
        Callback for status messages. Use ``print`` in notebooks,
        ``logger.info`` in production.

    :return:
        Tuple of (matplotlib Figure, fitted DecisionTreeRegressor).
    """
    import matplotlib.pyplot as plt
    from sklearn.tree import DecisionTreeRegressor, plot_tree

    parameter_df, X, y, parameter_names = _prepare_parameter_data(df, analysis_metric)

    printer(f"Analysing {analysis_metric} for parameters: {parameter_names}")

    tree = DecisionTreeRegressor(max_depth=max_depth)
    tree.fit(X, y)

    fig, ax = plt.subplots(figsize=figsize)
    plot_tree(tree, feature_names=X.columns, filled=True, fontsize=10, ax=ax)
    ax.set_title("Decision Tree for Parameter Importance")

    return fig, tree


def analyse_feature_importance(
    df: pd.DataFrame,
    analysis_metric: str = "CAGR",
    n_estimators: int = 100,
    figsize: tuple[int, int] = (10, 6),
    printer: Callable = print,
) -> tuple[MatplotlibFigure, pd.Series]:
    """Rank parameters by importance using a random forest model.

    Trains a ``RandomForestRegressor`` on the parameter grid and
    extracts Gini-based feature importances. Useful for identifying
    which parameters matter most when the search space is large.

    :param df:
        Grid search results DataFrame with parameter multi-index.

    :param analysis_metric:
        Column name of the performance metric to analyse.

    :param n_estimators:
        Number of trees in the random forest.

    :param figsize:
        Matplotlib figure size as (width, height) in inches.

    :param printer:
        Callback for status messages.

    :return:
        Tuple of (matplotlib Figure with horizontal bar chart,
        pd.Series of importances indexed by parameter name, sorted ascending).
    """
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestRegressor

    parameter_df, X, y, parameter_names = _prepare_parameter_data(df, analysis_metric)

    printer(f"Training random forest ({n_estimators} estimators) for {analysis_metric}")

    model = RandomForestRegressor(n_estimators=n_estimators)
    model.fit(X, y)

    importances = pd.Series(model.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(range(len(importances)), importances.values, align="center")
    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels(importances.index)
    ax.set_xlabel("Relative Importance")
    ax.set_title(f"Feature Importances for {analysis_metric}")

    return fig, importances


def analyse_parameter_pair_heatmaps(
    df: pd.DataFrame,
    analysis_metric: str = "CAGR",
    param_pairs: list[tuple[str, str]] | None = None,
    printer: Callable = print,
) -> list[go.Figure]:
    """Create heatmaps showing mean performance for each pair of parameters.

    Each heatmap shows the mean value of ``analysis_metric`` across all
    other parameter dimensions, for one specific pair of parameters.
    Useful for spotting interaction effects between two parameters.

    :param df:
        Grid search results DataFrame with parameter multi-index.

    :param analysis_metric:
        Column name of the performance metric to analyse.

    :param param_pairs:
        List of (row_param, col_param) tuples to plot.
        If ``None``, generates consecutive pairs from the parameter columns.

    :param printer:
        Callback for status messages.

    :return:
        List of Plotly Figure objects, one per parameter pair.
    """
    parameter_names = list(df.index.names)
    parameter_df = df.reset_index()

    if param_pairs is None:
        param_pairs = []
        for i in range(1, len(parameter_names)):
            param_pairs.append((parameter_names[i], parameter_names[i - 1]))

    printer(f"Creating {len(param_pairs)} heatmaps for {analysis_metric}")

    figures = []
    for param1, param2 in param_pairs:
        pivot = parameter_df.pivot_table(
            values=analysis_metric,
            index=param1,
            columns=param2,
            aggfunc="mean",
        )

        fig = go.Figure(
            data=go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale="Viridis",
                colorbar=dict(title=analysis_metric),
            )
        )

        fig.update_layout(
            title=f"Impact of {param1} and {param2} on {analysis_metric}",
            xaxis_title=param2,
            yaxis_title=param1,
        )

        figures.append(fig)

    return figures


def analyse_parameter_clusters(
    df: pd.DataFrame,
    analysis_metric: str = "CAGR",
    n_clusters: int = 5,
    printer: Callable = print,
) -> tuple[pd.DataFrame, go.Figure | None]:
    """Cluster parameter combinations and summarise performance per cluster.

    Applies KMeans clustering to the standardised parameter space, then
    reports aggregate performance statistics (mean, min, max) for each
    cluster. Optionally produces a 3-D PCA scatter plot if there are
    enough features.

    :param df:
        Grid search results DataFrame with parameter multi-index.

    :param analysis_metric:
        Column name of the performance metric used in the summary table.

    :param n_clusters:
        Number of KMeans clusters.

    :param printer:
        Callback for status messages.

    :return:
        Tuple of (cluster_perf DataFrame, Plotly 3-D scatter Figure or None
        if PCA fails due to insufficient features).
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    parameter_df, X, y, parameter_names = _prepare_parameter_data(df, analysis_metric)

    printer(f"Clustering {len(parameter_df)} combinations into {n_clusters} clusters")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    parameter_df["cluster"] = kmeans.fit_predict(X_scaled)

    cluster_perf = parameter_df.groupby("cluster").agg(
        {
            "CAGR": ["mean", "min", "max", "count"],
            "Sharpe": ["mean", "min", "max"],
            "Max DD": ["mean", "min", "max"],
        }
    ).reset_index()

    fig = None
    try:
        pca = PCA(n_components=3)
        components = pca.fit_transform(X_scaled)

        fig = px.scatter_3d(
            x=components[:, 0],
            y=components[:, 1],
            z=components[:, 2],
            color=parameter_df["cluster"],
            hover_data={
                "Sharpe": parameter_df["Sharpe"],
                "CAGR": parameter_df["CAGR"],
            },
            title="Clustering of Parameter Combinations",
        )
    except ValueError as e:
        printer(f"PCA visualisation skipped: {e}")

    return cluster_perf, fig


def analyse_parallel_coordinates(
    df: pd.DataFrame,
    analysis_metric: str = "CAGR",
    printer: Callable = print,
) -> go.Figure:
    """Create a parallel coordinates plot of parameter combinations.

    Each line represents one grid search combination, coloured by
    ``analysis_metric``. Useful for spotting which parameter ranges
    consistently produce good or bad results.

    :param df:
        Grid search results DataFrame with parameter multi-index.

    :param analysis_metric:
        Column name of the performance metric used for colouring.

    :param printer:
        Callback for status messages.

    :return:
        Plotly Figure with parallel coordinates chart.
    """
    parameter_names = list(df.index.names)
    parameter_df = df.reset_index()

    dimensions = parameter_names + [analysis_metric]

    printer(f"Creating parallel coordinates plot for {len(dimensions)} dimensions")

    fig = px.parallel_coordinates(
        parameter_df,
        dimensions=dimensions,
        color=analysis_metric,
        color_continuous_scale=px.colors.sequential.Peach,
        title="Parameter Impact on Strategy Performance",
    )

    fig.update_layout(
        font=dict(size=14),
        title=dict(
            text="Parameter Impact on Strategy Performance",
            font=dict(size=20),
        ),
        coloraxis_colorbar=dict(
            title=dict(
                text=analysis_metric,
                font=dict(size=16),
            ),
            tickfont=dict(size=14),
        ),
    )

    return fig
