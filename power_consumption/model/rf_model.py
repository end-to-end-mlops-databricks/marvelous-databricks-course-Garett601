"""
Main module for the power consumption model.
"""

from typing import Dict, Self, Tuple

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline


class ConsumptionModel:
    """
    A model for predicting power consumption using a multivariate, multi-output random forest regressor.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        The preprocessor pipeline for feature transformation.
    config : Dict
        Configuration dictionary containing model parameters.

    Attributes
    ----------
    config : Dict
        Configuration dictionary.
    model : Pipeline
        The complete model pipeline including preprocessing and regression steps.
    """

    def __init__(self, preprocessor: ColumnTransformer, config: Dict) -> None:
        """
        Initialise the ConsumptionModel.
        """
        self.config = config
        self.model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "regressor",
                    MultiOutputRegressor(
                        RandomForestRegressor(
                            n_estimators=config["parameters"]["n_estimators"],
                            max_depth=config["parameters"]["max_depth"],
                            random_state=42,
                        )
                    ),
                ),
            ]
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Self:
        """
        Train the model on the given data.

        Parameters
        ----------
        X_train : np.ndarray
            Training feature matrix.
        y_train : np.ndarray
            Training target matrix.

        Returns
        -------
        Self
            The trained model.
        """
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix for prediction.

        Returns
        -------
        np.ndarray
            Predicted values.
        """
        return self.model.predict(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate the model performance.

        Parameters
        ----------
        X_test : np.ndarray
            Test feature matrix.
        y_test : np.ndarray
            True target values.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Mean squared error and R-squared score for each target variable.
        """
        y_pred = self.predict(X_test)
        mse = mean_squared_error(y_test, y_pred, multioutput="raw_values")
        r2 = r2_score(y_test, y_pred, multioutput="raw_values")
        return mse, r2

    def get_feature_importance(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get feature importances from the trained model.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Feature importances and corresponding feature names.
        """
        feature_importance = np.mean(
            [estimator.feature_importances_ for estimator in self.model.named_steps["regressor"].estimators_], axis=0
        )
        feature_names = self.model.named_steps["preprocessor"].get_feature_names_out()
        return feature_importance, feature_names
