"""Data preprocessing module for the power consumption dataset."""

from __future__ import annotations

from typing import Tuple

import pandas as pd
import pyspark.sql.functions as F
from loguru import logger
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from power_consumption.config import Config
from power_consumption.schemas.processed_data import PowerConsumptionSchema, ProcessedPowerConsumptionSchema


class DataProcessor:
    def __init__(self, config: Config, data: pd.DataFrame):
        """
        Initialise the DataProcessor.
        """
        self.data: pd.DataFrame = data
        self.config: Config = config

    def _clean_column_names(self) -> None:
        """Clean column names by joining split words."""
        self.data.columns = ["_".join(col.split()) for col in self.data.columns]

    def _create_preprocessor(self) -> ColumnTransformer:
        """
        Create a preprocessing pipeline using configuration.

        Returns
        -------
        ColumnTransformer
            The preprocessing pipeline.
        """
        numeric_features = self.config.dataset.num_features
        categorical_features = self.config.dataset.cat_features

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

        return ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

    def preprocess_data(self) -> None:
        """
        Preprocess the data by extracting time-based features and creating the preprocessor.
        """
        self._clean_column_names()

        logger.info("Creating time-based features")

        # Convert DateTime to datetime and set as index
        self.data["DateTime"] = pd.to_datetime(self.data["DateTime"])
        self.data.set_index("DateTime", inplace=True)

        # Create time-based features
        self.data["Hour"] = self.data.index.hour
        self.data["Day"] = self.data.index.day
        self.data["Month"] = self.data.index.month
        self.data["DayOfWeek"] = self.data.index.dayofweek
        self.data["IsWeekend"] = self.data["DayOfWeek"].isin([5, 6]).astype(int)

        self.data = PowerConsumptionSchema.validate(self.data)

        self.data = self.data.sort_index()

        # Create and apply preprocessor
        preprocessor = self._create_preprocessor()
        feature_columns = self.config.dataset.num_features + self.config.dataset.cat_features
        preprocessed_features = preprocessor.fit_transform(self.data[feature_columns])

        # Convert preprocessed features back to DataFrame
        feature_names = (
            preprocessor.named_transformers_["num"].get_feature_names_out().tolist()
            + preprocessor.named_transformers_["cat"].get_feature_names_out().tolist()
        )
        preprocessed_df = pd.DataFrame(preprocessed_features, columns=feature_names, index=self.data.index)

        # Combine preprocessed features with target variable
        target_columns = self.config.target.target
        result = pd.concat([preprocessed_df, self.data[target_columns]], axis=1)

        self.data = ProcessedPowerConsumptionSchema.validate(result)

    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the DataFrame (self.df) into training and test sets, maintaining temporal order.

        Parameters
        ----------
        test_size : float, optional
            The proportion of the dataset to include in the test split, by default 0.2.
        random_state : int, optional
            Random state for reproducibility, by default 42. Note that this doesn't affect the split
            when shuffle is False, but it's kept for consistency with sklearn's API.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            train_set, test_set
        """
        train_set, test_set = train_test_split(self.data, test_size=test_size, shuffle=False, random_state=random_state)

        logger.info(f"Train set shape: {train_set.shape}")
        logger.info(f"Test set shape: {test_set.shape}")
        logger.info(f"Train set date range: {train_set.index.min()} to {train_set.index.max()}")
        logger.info(f"Test set date range: {test_set.index.min()} to {test_set.index.max()}")

        return train_set, test_set

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame, spark: SparkSession) -> None:
        """
        Save the training and test sets to the catalog with a timestamp and enable change data feed.

        Parameters
        ----------
        train_set : pd.DataFrame
            The training set to be saved.
        test_set : pd.DataFrame
            The test set to be saved.
        spark : SparkSession
            The SparkSession to use for saving the data.

        Returns
        -------
        None
        """
        timestamp = F.to_utc_timestamp(F.current_timestamp(), "UTC")
        train_set_with_timestamp = spark.createDataFrame(train_set).withColumn("update_timestamp_utc", timestamp)

        test_set_with_timestamp = spark.createDataFrame(test_set).withColumn("update_timestamp_utc", timestamp)

        train_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_set"
        )

        test_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.test_set"
        )

        spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )
