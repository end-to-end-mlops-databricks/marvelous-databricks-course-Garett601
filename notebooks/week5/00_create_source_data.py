# Databricks notebook source
import pandas as pd
import numpy as np

from pyspark.sql import SparkSession

from power_consumption.config import Config
from power_consumption.preprocessing.data_preprocessor import DataProcessor

spark = SparkSession.builder.getOrCreate()
# COMMAND ----------
config = Config.from_yaml("../../configs/project_configs.yml")

catalog_name = config.catalog_name
schema_name = config.schema_name
raw_data_table = config.dataset.raw_data_table
# COMMAND ----------
data = spark.table(f"{catalog_name}.{schema_name}.{raw_data_table}").toPandas().drop(columns=["_rescued_data"])
data["DateTime"] = pd.to_datetime(data["DateTime"])
# COMMAND ----------
def generate_synthetic_data(df, num_rows=1008):
    """
    Generate synthetic data matching the structure of the given dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The original raw dataset used as a template.
    num_rows : int
        Number of synthetic rows to generate. Default is 1008 (1 week for DateTime with 10-minute intervals).

    Returns
    -------
    pd.DataFrame
        A synthetic dataset matching the structure of the raw data.
    """
    synthetic_data = pd.DataFrame()

    last_datetime = pd.to_datetime(df['DateTime'].max())
    synthetic_datetimes = [last_datetime + pd.Timedelta(minutes=10 * i) for i in range(1, num_rows + 1)]
    synthetic_data['DateTime'] = synthetic_datetimes

    for column in df.columns:
        if column not in ['DateTime']:
            mean, std = df[column].mean(), df[column].std()
            synthetic_data[column] = np.random.normal(mean, std, num_rows)

    synthetic_data = synthetic_data[df.columns]

    return synthetic_data
# COMMAND ----------
synthetic_data = generate_synthetic_data(data, num_rows=1008)
# COMMAND ----------
data_processor = DataProcessor(config, synthetic_data)
data_processor.preprocess_data()
# COMMAND ----------
data_processor.data.reset_index(inplace=True)
data_processor.data["id"] = data_processor.data["DateTime"].astype("int64") // 10**9
data_processor.data["id"] = data_processor.data["id"].astype(str)
data_processor.data["update_timestamp_utc"] = pd.Timestamp.utcnow()

# COMMAND ----------
existing_schema = spark.table(f"{catalog_name}.{schema_name}.train_set").schema

source_data_with_timestamp = spark.createDataFrame(data_processor.data)

# COMMAND ----------
source_data_with_timestamp.write.mode("append").saveAsTable(
    f"{catalog_name}.{schema_name}.source_data"
)

spark.sql(
    f"ALTER TABLE {catalog_name}.{schema_name}.source_data "
    "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
)
