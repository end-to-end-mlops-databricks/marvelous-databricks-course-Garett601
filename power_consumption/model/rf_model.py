"""
Main module for the power consumption model.
"""

from typing import List, Self, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

from power_consumption.config import Config


class ConsumptionModel:
    """
    A model for predicting power consumption using a multivariate, multi-output random forest regressor.

    Parameters
    ----------
    config : Config
        Configuration object containing model parameters.

    Attributes
    ----------
    config : Config
        Configuration object.
    model : MultiOutputRegressor
        The complete model pipeline including regression steps.
    """

    def __init__(self, config: Config) -> None:
        """
        Initialise the ConsumptionModel.
        """
        self.config = config
        self.model = MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=config.hyperparameters.n_estimators,
                max_depth=config.hyperparameters.max_depth,
                random_state=42,
            )
        )

    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> Self:
        """
        Train the model on the given data.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature matrix.
        y_train : pd.DataFrame
            Training target matrix.

        Returns
        -------
        Self
            The trained model.
        """
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix for prediction.

        Returns
        -------
        np.ndarray
            Predicted values.
        """
        return self.model.predict(X)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate the model performance.

        Parameters
        ----------
        X_test : pd.DataFrame
            Test feature matrix.
        y_test : pd.DataFrame
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

    def get_feature_importance(self) -> Tuple[np.ndarray, List[str]]:
        """
        Get feature importances from the trained model.

        Returns
        -------
        Tuple[np.ndarray, List[str]]
            Feature importances and corresponding feature names.
        """
        feature_importance = np.mean([estimator.feature_importances_ for estimator in self.model.estimators_], axis=0)
        feature_names = self.config.processed_features.num_features + self.config.processed_features.cat_features
        return feature_importance, feature_names
