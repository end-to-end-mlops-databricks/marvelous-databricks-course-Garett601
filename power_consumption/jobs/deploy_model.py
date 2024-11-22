"""Deploy the new model to the serving endpoint."""

import argparse

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ServedEntityInput
from pyspark.sql import SparkSession

from power_consumption.config import Config
from power_consumption.utils import get_dbutils

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

dbutils = get_dbutils(spark)


config = Config.from_yaml(config_path)
catalog_name = config.catalog_name
schema_name = config.schema_name

model_version = dbutils.jobs.taskValues.get(taskKey="evaluate_model", key="model_version")

workspace = WorkspaceClient()

workspace.serving_endpoints.update_config_and_wait(
    name="power-consumption-model-serving-fe",
    served_entities=[
        ServedEntityInput(
            entity_name=f"{catalog_name}.{schema_name}.power_consumption_model_fe",
            scale_to_zero_enabled=True,
            workload_size="Small",
            entity_version=model_version,
        )
    ],
)
