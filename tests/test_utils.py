"""Tests for utility functions in the power consumption project."""

import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import yaml

from power_consumption.utils import (
    load_config,
    visualise_results,
    plot_actual_vs_predicted,
    plot_feature_importance,
)


def test_load_config():
    config = {"key1": "value1", "key2": 2}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as temp_file:
        yaml.dump(config, temp_file)
        temp_file_path = temp_file.name

    try:
        loaded_config = load_config(temp_file_path)
        assert loaded_config == config
    finally:
        os.unlink(temp_file_path)


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
