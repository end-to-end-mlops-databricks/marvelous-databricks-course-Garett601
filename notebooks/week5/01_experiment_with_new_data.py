# Databricks notebook source
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

from power_consumption.config import Config

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------
config = Config.from_yaml("../../configs/project_configs.yml")

catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------
source_df = spark.table(f"{catalog_name}.{schema_name}.source_data")
train_df = spark.table(f"{catalog_name}.{schema_name}.train_set")
test_df = spark.table(f"{catalog_name}.{schema_name}.test_set")

# COMMAND ----------
max_train_date = train_df.select(F.max("DateTime")).collect()[0][0]
max_test_date = test_df.select(F.max("DateTime")).collect()[0][0]
# COMMAND ----------
newer_data = source_df.filter(F.col("DateTime") > max_test_date)
# COMMAND ----------
if newer_data.count() > 0:
    updated_train = train_df.unionAll(test_df)

    new_test = newer_data

    # Print some validation info
    print(f"Original train set count: {train_df.count()}")
    print(f"Current test set count: {test_df.count()}")
    print(f"Updated train set count: {updated_train.count()}")
    print(f"New test set count: {new_test.count()}")

    print("\nSample from updated train set:")
    updated_train.select("DateTime").orderBy(F.desc("DateTime")).show(5)
    print("\nSample from new test set:")
    new_test.select("DateTime").orderBy(F.asc("DateTime")).show(5)
else:
    print("No newer data found in source_data")
# COMMAND ----------
feature_table = spark.table(f"{catalog_name}.{schema_name}.power_consumption_features")
feature_table.createOrReplaceTempView("feature_table")
# COMMAND ----------
new_test.createOrReplaceTempView("new_test")
# COMMAND ----------
check = spark.sql(f"""
INSERT INTO feature_table
SELECT
    id,
    DateTime,
    Temperature,
    Humidity,
    Wind_Speed,
    general_diffuse_flows,
    diffuse_flows,
    DayOfWeek_1,
    DayOfWeek_2,
    DayOfWeek_3,
    DayOfWeek_4,
    DayOfWeek_5,
    DayOfWeek_6,
    IsWeekend_1
FROM new_test
""")
# COMMAND ----------
