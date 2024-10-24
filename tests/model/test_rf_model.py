import numpy as np
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from power_consumption.model import ConsumptionModel
from power_consumption.config import Config, Hyperparameters, Features, Target, Dataset

@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = np.random.rand(100, 5)
    y = np.random.rand(100, 2)
    return X[:80], y[:80], X[80:], y[80:]

@pytest.fixture
def model_config():
    return Config(
        catalog_name="test_catalog",
        schema_name="test_schema",
        hyperparameters=Hyperparameters(
            learning_rate=0.01,
            n_estimators=100,
            max_depth=5
        ),
        features=Features(
            num_features=["feature1", "feature2", "feature3", "feature4", "feature5"],
            cat_features=[]
        ),
        target=Target(target=["target1", "target2"]),
        dataset=Dataset(id=1)
    )

@pytest.fixture
def preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), list(range(5)))
        ]
    )

def test_model_initialisation(preprocessor, model_config):
    model = ConsumptionModel(preprocessor, model_config)
    assert isinstance(model.config, Config)
    assert isinstance(model.model, Pipeline)
    assert model.model.named_steps["regressor"].estimator.n_estimators == model_config.hyperparameters.n_estimators
    assert model.model.named_steps["regressor"].estimator.max_depth == model_config.hyperparameters.max_depth

def test_model_train_and_predict(sample_data, preprocessor, model_config):
    X_train, y_train, X_test, _ = sample_data
    model = ConsumptionModel(preprocessor, model_config)

    trained_model = model.train(X_train, y_train)
    assert trained_model is model

    y_pred = model.predict(X_test)
    assert y_pred.shape == (20, len(model_config.target.target))

def test_model_evaluate(sample_data, preprocessor, model_config):
    X_train, y_train, X_test, y_test = sample_data
    model = ConsumptionModel(preprocessor, model_config)
    model.train(X_train, y_train)

    mse, r2 = model.evaluate(X_test, y_test)
    assert mse.shape == (len(model_config.target.target),)
    assert r2.shape == (len(model_config.target.target),)
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
