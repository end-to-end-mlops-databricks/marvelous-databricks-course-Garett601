# Databricks notebook source
# MAGIC %md
# MAGIC # Setup

# COMMAND ----------

# MAGIC %pip install ../power_consumption-0.0.1-py3-none-any.whl

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,imports
import time

import requests
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    OnlineTableSpec,
    OnlineTableSpecTriggeredSchedulingPolicy,
)
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from pyspark.sql import SparkSession

from power_consumption.config import Config

# COMMAND ----------

# DBTITLE 1,initialisations
workspace = WorkspaceClient()
spark = SparkSession.builder.getOrCreate()

config = Config.from_yaml("../../configs/project_configs.yml")

catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------

# DBTITLE 1,create online table
online_table_name = f"{catalog_name}.{schema_name}.power_consumption_online"
spec = OnlineTableSpec(
    primary_key_columns=["id"],
    source_table_full_name=f"{catalog_name}.{schema_name}.power_consumption_features",
    run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict({"triggered": "true"}),
    perform_full_copy=False,
)

online_table_pipeline = workspace.online_tables.create(name=online_table_name, spec=spec)

# COMMAND ----------

# MAGIC %md
# MAGIC # Create Endpoint

# COMMAND ----------

# DBTITLE 1,create endpoint
workspace.serving_endpoints.create(
    name="power-consumption-model-serving-fe",
    config=EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name=f"{catalog_name}.{schema_name}.power_consumption_model_fe",
                scale_to_zero_enabled=True,
                workload_size="Small",
                entity_version=3,
            )
        ]
    ),
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Call the Endpoint

# COMMAND ----------

# DBTITLE 1,set token and host
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------

# DBTITLE 1,create request body
columns = config.processed_features.num_features + config.processed_features.cat_features
exclude_columns = ["general_diffuse_flows", "diffuse_flows"]

required_columns = [col for col in columns if col not in exclude_columns]


train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()

sampled_records = train_set[required_columns].sample(n=1000, replace=True).to_dict(orient="records")

dataframe_records = [[record] for record in sampled_records]

# COMMAND ----------

start_time = time.time()

model_serving_endpoint = f"https://{host}/serving-endpoints/power-consumption-model-serving-fe/invocations"

response = requests.post(
    f"{model_serving_endpoint}",
    headers={"Authorization": f"Bearer {token}"},
    json={"dataframe_records": dataframe_records[0]},
)

end_time = time.time()
execution_time = end_time - start_time

print("Response status:", response.status_code)
print("Reponse text:", response.text)
print("Execution time:", execution_time, "seconds")

# COMMAND ----------
