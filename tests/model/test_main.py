import numpy as np
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from power_consumption.model.main import ConsumptionModel

@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = np.random.rand(100, 5)
    y = np.random.rand(100, 2)
    return X[:80], y[:80], X[80:], y[80:]

@pytest.fixture
def model_config():
    return {
        "parameters": {
            "n_estimators": 100,
            "max_depth": 5
        }
    }

@pytest.fixture
def preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), list(range(5)))
        ]
    )

def test_model_initialisation(preprocessor, model_config):
    model = ConsumptionModel(preprocessor, model_config)
    assert model.config == model_config
    assert isinstance(model.model, Pipeline)

def test_model_train_and_predict(sample_data, preprocessor, model_config):
    X_train, y_train, X_test, _ = sample_data
    model = ConsumptionModel(preprocessor, model_config)

    trained_model = model.train(X_train, y_train)
    assert trained_model is model

    y_pred = model.predict(X_test)
    assert y_pred.shape == (20, 2)

def test_model_evaluate(sample_data, preprocessor, model_config):
    X_train, y_train, X_test, y_test = sample_data
    model = ConsumptionModel(preprocessor, model_config)
    model.train(X_train, y_train)

    mse, r2 = model.evaluate(X_test, y_test)
    assert mse.shape == (2,)
    assert r2.shape == (2,)
    assert np.all(mse >= 0)
    assert np.all(r2 <= 1) and np.all(r2 >= -1)

def test_get_feature_importance(sample_data, preprocessor, model_config):
    X_train, y_train, _, _ = sample_data
    model = ConsumptionModel(preprocessor, model_config)
    model.train(X_train, y_train)

    feature_importance, feature_names = model.get_feature_importance()
    assert feature_importance.shape == (5,)
    assert len(feature_names) == 5
    assert np.all(feature_importance >= 0) and np.all(feature_importance <= 1)
    assert np.isclose(np.sum(feature_importance), 1, atol=1e-6)
