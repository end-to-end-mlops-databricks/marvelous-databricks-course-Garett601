"""Tests for utility functions in the power consumption project."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from pyspark.sql import SparkSession

from power_consumption.utils import (
    visualise_results,
    plot_actual_vs_predicted,
    plot_feature_importance,
    get_dbutils,
)

@pytest.fixture
def mock_spark(mocker):
    spark = mocker.Mock(spec=SparkSession)
    spark.conf = mocker.Mock()
    return spark

@pytest.mark.parametrize("n_targets", [1, 3])
def test_visualise_results(n_targets, mocker):
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    y_test = pd.DataFrame(np.random.rand(100, n_targets), index=dates)
    y_pred = np.random.rand(100, n_targets)
    target_names = [f"Target_{i}" for i in range(n_targets)]

    mock_show = mocker.patch.object(plt, "show")
    visualise_results(y_test, y_pred, target_names)
    mock_show.assert_called_once()


@pytest.mark.parametrize("n_targets", [1, 3])
def test_plot_actual_vs_predicted(n_targets, mocker):
    """
    Test the plot_actual_vs_predicted function.

    Parameters
    ----------
    n_targets : int
        Number of targets to test with.
    mocker : pytest.MockFixture
        Pytest mocker fixture.
    """
    y_test = np.random.rand(100, n_targets)
    y_pred = np.random.rand(100, n_targets)
    target_names = [f"Target_{i}" for i in range(n_targets)]

    mock_show = mocker.patch.object(plt, "show")
    plot_actual_vs_predicted(y_test, y_pred, target_names)
    mock_show.assert_called_once()


@pytest.mark.parametrize("n_features, top_n", [(5, 3), (10, 5)])
def test_plot_feature_importance(n_features, top_n, mocker):
    """
    Test the plot_feature_importance function.

    Parameters
    ----------
    n_features : int
        Number of features to test with.
    top_n : int
        Number of top features to display.
    mocker : pytest.MockFixture
        Pytest mocker fixture.
    """
    feature_importance = np.random.rand(n_features)
    feature_names = np.array([f"Feature_{i}" for i in range(n_features)])

    mock_show = mocker.patch.object(plt, "show")
    plot_feature_importance(feature_importance, feature_names, top_n)
    mock_show.assert_called_once()


def test_get_dbutils_databricks_environment(mock_spark, mocker):
    mock_spark.conf.get.return_value = "true"

    mock_dbutils = mocker.patch("pyspark.dbutils.DBUtils")
    mock_dbutils.return_value = "mock_databricks_dbutils"

    result = get_dbutils(mock_spark)

    assert result == "mock_databricks_dbutils"
    mock_dbutils.assert_called_once_with(mock_spark)


def test_get_dbutils_local_environment(mock_spark, mocker):
    mock_spark.conf.get.return_value = "false"

    mock_ipython = mocker.Mock()
    mock_ipython.user_ns = {"dbutils": "mock_ipython_dbutils"}
    mocker.patch("IPython.get_ipython", return_value=mock_ipython)

    result = get_dbutils(mock_spark)

    assert result == "mock_ipython_dbutils"
