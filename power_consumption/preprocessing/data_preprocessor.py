"""Data preprocessing module for the power consumption dataset."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from ucimlrepo import fetch_ucirepo


class DataProcessor:
    def __init__(self, dataset_id: int, config: Dict[str, Any]):
        """
        Initialise the DataProcessor.
        """
        self.data: Optional[pd.DataFrame] = None
        self.X: Optional[pd.DataFrame] = None
        self.y: Optional[pd.DataFrame] = None
        self.config: Dict[str, Any] = config
        self.preprocessor: Optional[ColumnTransformer] = None
        self.load_data(dataset_id)

    def load_data(self, dataset_id: int) -> None:
        """
        Load the dataset from UCI ML Repository.

        Parameters
        ----------
        dataset_id : int
            The ID of the dataset to fetch.
        """
        dataset = fetch_ucirepo(id=dataset_id)
        logger.info("Loading dataset:\n" f"  ID: {dataset_id}\n" f"  Name: {dataset.metadata.name}")
        self.data = dataset.data.original
        self._clean_column_names()

    def _clean_column_names(self) -> None:
        """Clean column names by joining split words."""
        self.data.columns = [" ".join(col.split()) for col in self.data.columns]

    def create_preprocessor(self) -> ColumnTransformer:
        """
        Create a preprocessing pipeline using configuration.

        Returns
        -------
        ColumnTransformer
            The preprocessing pipeline.
        """
        numeric_features = self.config["num_features"]
        categorical_features = self.config["cat_features"]

        logger.info(f"Numeric features: {numeric_features}")
        logger.info(f"Categorical features: {categorical_features}")

        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
                ("onehot", OneHotEncoder(drop="first")),
            ]
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        return self.preprocessor

    def preprocess_data(self) -> None:
        """
        Preprocess the data by extracting time-based features and creating the preprocessor.
        """
        logger.info("Creating time-based features")

        self.data["DateTime"] = pd.to_datetime(self.data["DateTime"])

        # Set DateTime as the index
        self.data.set_index("DateTime", inplace=True)

        self.data["Hour"] = self.data.index.hour
        self.data["Day"] = self.data.index.day
        self.data["Month"] = self.data.index.month
        self.data["DayOfWeek"] = self.data.index.dayofweek
        self.data["IsWeekend"] = self.data["DayOfWeek"].isin([5, 6]).astype(int)

        self.data = self.data.sort_index()

        self.create_preprocessor()

    def split_data(self, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split the data into features and target, then into training and test sets.

        Parameters
        ----------
        test_size : float, optional
            The proportion of the dataset to include in the test split, by default 0.2.

        Returns
        -------
        tuple of pd.DataFrame
            X_train, X_test, y_train, y_test
        """
        target_columns = self.config["target"]
        feature_columns = self.config["num_features"] + self.config["cat_features"]

        X = self.data[feature_columns]
        y = self.data[target_columns]

        logger.info(f"X shape: {X.shape}")
        logger.info(f"y shape: {y.shape}")

        # Use index for splitting to maintain time order
        split_index = int(len(self.data) * (1 - test_size))

        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

        logger.info(f"Train set date range: {X_train.index.min()} to {X_train.index.max()}")
        logger.info(f"Test set date range: {X_test.index.min()} to {X_test.index.max()}")

        return X_train, X_test, y_train, y_test

    def fit_transform_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit the preprocessor on the training data and transform both training and test data.

        Parameters
        ----------
        X_train : pd.DataFrame
            The training feature matrix.
        X_test : pd.DataFrame
            The test feature matrix.

        Returns
        -------
        tuple of np.ndarray
            Transformed X_train and X_test.
        """
        X_train_transformed = self.preprocessor.fit_transform(X_train)
        X_test_transformed = self.preprocessor.transform(X_test)

        logger.info(f"X_train_transformed shape: {X_train_transformed.shape}")
        logger.info(f"X_test_transformed shape: {X_test_transformed.shape}")

        return X_train_transformed, X_test_transformed