"""Preprocess the source data and update the train and test sets."""

import argparse
import time

import pyspark.sql.functions as F
from databricks.sdk import WorkspaceClient
from loguru import logger
from pyspark.sql import SparkSession

from power_consumption.config import Config
from power_consumption.utils import get_dbutils

workspace = WorkspaceClient()
spark = SparkSession.builder.getOrCreate()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_path",
    action="store",
    default=None,
    type=str,
    required=True,
)

args = parser.parse_args()
config_path = args.config_path

config = Config.from_yaml(config_path)

catalog_name = config.catalog_name
schema_name = config.schema_name
pipeline_id = config.pipeline_id

source_df = spark.table(f"{catalog_name}.{schema_name}.source_data")
train_df = spark.table(f"{catalog_name}.{schema_name}.train_set")
test_df = spark.table(f"{catalog_name}.{schema_name}.test_set")

max_train_date = train_df.select(F.max("DateTime")).collect()[0][0]
max_test_date = test_df.select(F.max("DateTime")).collect()[0][0]
newer_data = source_df.filter(F.col("DateTime") > max_test_date)
if newer_data.count() > 0:
    # Update train set with test set to expand the training set
    updated_train = train_df.unionAll(test_df)
    # Create new test set with the newer data
    new_test = newer_data

    logger.info(f"Original train set count: {train_df.count()}")
    logger.info(f"Updated train set count: {updated_train.count()}")

    logger.info(f"Original test set count: {test_df.count()}")
    logger.info(f"New test set count: {new_test.count()}")

    logger.info("Sample from updated train set:")
    train_sample = (
        updated_train.select("DateTime").orderBy(F.desc("DateTime")).limit(5).toPandas().iloc[::-1].to_string()
    )
    logger.info(f"\n{train_sample}")

    logger.info("Sample from new test set:")
    test_sample = new_test.select("DateTime").orderBy(F.asc("DateTime")).limit(5).toPandas().to_string()
    logger.info(f"\n{test_sample}")

    test_df.write.mode("append").saveAsTable(f"{catalog_name}.{schema_name}.train_set")
    new_test.write.mode("overwrite").saveAsTable(f"{catalog_name}.{schema_name}.test_set")

    # Create temp view for new_test to use in SQL
    new_test.createOrReplaceTempView("new_test_data")

    # Update feature table with only the new_test data so the feature table has all the records
    spark.sql(f"""
        INSERT INTO {catalog_name}.{schema_name}.power_consumption_features
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
        FROM new_test_data
    """)

    refreshed = 1

    update_response = workspace.pipelines.start_update(
        pipeline_id=pipeline_id,
        full_refresh=False,
    )
    while True:
        update_info = workspace.pipelines.get_update(
            pipeline_id=pipeline_id,
            update_id=update_response.update_id,
        )
        state = update_info.update.state.value
        if state == "COMPLETED":
            break
        elif state in ["FAILED", "CANCELED"]:
            logger.error(
                f"Pipeline update failed. State: {state}, Pipeline ID: {pipeline_id}, Update ID: {update_response.update_id}"
            )
            raise SystemError("Online table failed to update.")
        elif state == "WAITING_FOR_RESOURCES":
            logger.info("Pipeline is waiting for resources.")
        else:
            logger.info(f"Pipeline is in {state} state.")
        time.sleep(30)
else:
    refreshed = 0
    logger.info("No newer data found in source data")

dbutils = get_dbutils(spark=spark)
dbutils.jobs.taskValues.set(key="refreshed", value=refreshed)
