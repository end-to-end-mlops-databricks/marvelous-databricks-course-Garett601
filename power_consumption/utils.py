"""Utility functions for the power consumption project."""

import matplotlib.pyplot as plt
import numpy as np
from pyspark.sql import SparkSession


def visualise_results(y_test, y_pred, target_names):
    """
    Visualise actual vs predicted power consumption for each target/zone as a time series.

    Parameters
    ----------
    y_test : pd.DataFrame
        Actual values with datetime index.
    y_pred : array-like
        Predicted values.
    target_names : list
        Names of the target variables/zones.
    """
    n_targets = len(target_names)
    fig, axes = plt.subplots(n_targets, 1, figsize=(15, 6 * n_targets), squeeze=False)

    for i, target in enumerate(target_names):
        ax = axes[i, 0]
        ax.plot(y_test.index, y_test.iloc[:, i], label="Actual", alpha=0.7)
        ax.plot(y_test.index, y_pred[:, i], label="Predicted", alpha=0.7)
        ax.set_title(f"{target} - Time Series")
        ax.set_xlabel("Date")
        ax.set_ylabel("Power Consumption")
        ax.legend()
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()


def plot_actual_vs_predicted(y_test, y_pred, target_names):
    """
    Plot actual vs predicted values for each target/zone.

    Parameters
    ----------
    y_test : array-like
        Actual values.
    y_pred : array-like
        Predicted values.
    target_names : list
        Names of the target variables/zones.
    """
    n_targets = len(target_names)
    fig, axes = plt.subplots(1, n_targets, figsize=(6 * n_targets, 5), squeeze=False)

    for i, target in enumerate(target_names):
        ax = axes[0, i]
        ax.scatter(y_test[:, i], y_pred[:, i], alpha=0.5)
        min_val = min(y_test[:, i].min(), y_pred[:, i].min())
        max_val = max(y_test[:, i].max(), y_pred[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
        ax.set_xlabel("Actual Consumption")
        ax.set_ylabel("Predicted Consumption")
        ax.set_title(f"{target} - Actual vs Predicted")

    plt.tight_layout()
    plt.show()


def plot_feature_importance(feature_importance, feature_names, top_n=10):
    """
    Plot feature importance.

    Parameters
    ----------
    feature_importance : array-like
        Feature importance scores.
    feature_names : array-like
        Names of the features.
    top_n : int, optional
        Number of top features to display, by default 10.
    """
    plt.figure(figsize=(10, 6))
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx[-top_n:].shape[0]) + 0.5
    top_features = feature_importance[sorted_idx[-top_n:]]
    top_names = [feature_names[i] for i in sorted_idx[-top_n:]]
    plt.barh(pos, top_features)
    plt.yticks(pos, top_names)
    plt.title(f"Top {top_n} Feature Importance")
    plt.tight_layout()
    plt.show()


def get_dbutils(spark: SparkSession) -> object:
    """
    Get the dbutils object based on the SparkSession.

    Parameters:
        spark (pyspark.sql.SparkSession): The SparkSession object.

    Returns:
        object: The dbutils object.

    """
    dbutils = None
    if spark.conf.get("spark.databricks.service.client.enabled") == "true":
        from pyspark.dbutils import DBUtils  # type: ignore

        dbutils = DBUtils(spark)

    else:
        import IPython

        dbutils = IPython.get_ipython().user_ns["dbutils"]

    return dbutils
