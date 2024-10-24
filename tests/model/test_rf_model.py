import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from power_consumption.model import ConsumptionModel
from power_consumption.config import Config
from power_consumption.preprocessing.data_preprocessor import DataProcessor
from loguru import logger

@pytest.fixture
def sample_data():
    np.random.seed(42)
    date_range = pd.date_range(start="2023-01-01", end="2023-01-10", freq="h")
    data = {
        "DateTime": date_range,
        "Temperature": np.random.uniform(0, 30, len(date_range)),
        "Humidity": np.random.uniform(30, 90, len(date_range)),
        "Wind_Speed": np.random.uniform(0, 20, len(date_range)),
        "general_diffuse_flows": np.random.uniform(0, 100, len(date_range)),
        "diffuse_flows": np.random.uniform(0, 100, len(date_range)),
        "Zone_1_Power_Consumption": np.random.uniform(100, 500, len(date_range)),
        "Zone_2_Power_Consumption": np.random.uniform(100, 500, len(date_range)),
        "Zone_3_Power_Consumption": np.random.uniform(100, 500, len(date_range)),
    }
    return pd.DataFrame(data)

@pytest.fixture
def data_processor(sample_data, project_config):
    processor = DataProcessor(project_config, sample_data)
    processor.preprocess_data()
    return processor

@pytest.fixture
def train_test_split(data_processor):
    return data_processor.split_data(test_size=0.2, random_state=42)

def test_model_initialisation(project_config):
    model = ConsumptionModel(project_config)

    assert isinstance(model.config, Config)
    assert model.model.estimator.n_estimators == project_config.hyperparameters.n_estimators
    assert model.model.estimator.max_depth == project_config.hyperparameters.max_depth

def test_model_train_and_predict(train_test_split, project_config):
    train_data, test_data = train_test_split
    X_train = train_data.drop(project_config.target.target, axis=1)
    y_train = train_data[project_config.target.target]
    X_test = test_data.drop(project_config.target.target, axis=1)

    model = ConsumptionModel(project_config)

    trained_model = model.train(X_train, y_train)
    assert trained_model is model

    y_pred = model.predict(X_test)
    assert y_pred.shape == (len(X_test), len(project_config.target.target))

def test_model_evaluate(train_test_split, project_config):
    train_data, test_data = train_test_split
    X_train = train_data.drop(project_config.target.target, axis=1)
    y_train = train_data[project_config.target.target]
    X_test = test_data.drop(project_config.target.target, axis=1)
    y_test = test_data[project_config.target.target]

    model = ConsumptionModel(project_config)
    model.train(X_train, y_train)

    mse, r2 = model.evaluate(X_test, y_test)
    assert mse.shape == (len(project_config.target.target),)
    assert r2.shape == (len(project_config.target.target),)
    assert np.all(mse >= 0)
    assert np.all(r2 <= 1) and np.all(r2 >= -1)

def test_get_feature_importance(train_test_split, project_config):
    train_data, _ = train_test_split
    X_train = train_data.drop(project_config.target.target, axis=1)
    y_train = train_data[project_config.target.target]

    model = ConsumptionModel(project_config)
    model.train(X_train, y_train)

    feature_importance, feature_names = model.get_feature_importance()
    assert feature_importance.shape == (X_train.shape[1],)
    assert len(feature_names) == X_train.shape[1]
    assert np.all(feature_importance >= 0) and np.all(feature_importance <= 1)
    assert np.isclose(np.sum(feature_importance), 1, atol=1e-6)
