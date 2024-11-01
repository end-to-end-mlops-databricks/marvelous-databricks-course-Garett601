# Databricks notebook source
from power_consumption.preprocessing.data_preprocessor import DataProcessor
from power_consumption.config import Config
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()


# COMMAND ----------

config = Config.from_yaml("../../configs/project_configs.yml")

# COMMAND ----------
catalog_name = config.catalog_name
schema_name = config.schema_name
raw_data_table = config.dataset.raw_data_table
# COMMAND ----------
data_spark = spark.table(f"{catalog_name}.{schema_name}.{raw_data_table}")
# COMMAND ----------
data_pandas = data_spark.toPandas()
# COMMAND ----------
data_processor = DataProcessor(config, data_pandas)
# COMMAND ----------
data_processor.preprocess_data()
# COMMAND ----------
train_set, test_set = data_processor.split_data()
# COMMAND ----------
train_set.reset_index(inplace=True)
test_set.reset_index(inplace=True)
# Convert DateTime to Unix timestamp (milliseconds) and create ID column
train_set['id'] = train_set['DateTime'].astype('int64') // 10**9
test_set['id'] = test_set['DateTime'].astype('int64') // 10**9

# Convert to string
train_set['id'] = train_set['id'].astype(str)
test_set['id'] = test_set['id'].astype(str)
# COMMAND ----------
data_processor.save_to_catalog(train_set=train_set, test_set=test_set, spark=spark)
# COMMAND ----------
