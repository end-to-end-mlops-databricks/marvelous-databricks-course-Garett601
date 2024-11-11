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
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
    TrafficConfig,
    Route,
)
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

# DBTITLE 1,load data
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC # Create endpoint

# COMMAND ----------

workspace.serving_endpoints.create(
    name="power-consumption-model-serving",
    config=EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name=f"{catalog_name}.{schema_name}.power_consumption_model",
                scale_to_zero_enabled=True,
                workload_size="Small",
                entity_version=2,
            )
        ],
    traffic_config=TrafficConfig(
        routes=[
            Route(
                served_model_name="power_consumption_model-2",
                traffic_percentage=100,
            )
        ]
    ),
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

# DBTITLE 1,create sample request body
required_columns = config.processed_features.num_features + config.processed_features.cat_features

sampled_records = train_set[required_columns].sample(n=1000, replace=True).to_dict(orient="records")

dataframe_records = [[record] for record in sampled_records]

# COMMAND ----------

# MAGIC %md
# MAGIC ### example of body
# MAGIC
# MAGIC Each body should be a list of json with columns
# MAGIC
# MAGIC ```python
# MAGIC [{'Temperature': 0.9732703198032969,
# MAGIC   'Humidity': -0.27133371777652626,
# MAGIC   'Wind_Speed': -0.8048607918423459,
# MAGIC   'Hour': 0,
# MAGIC   'Day': 0,
# MAGIC   'Month': 0,
# MAGIC   'general_diffuse_flows': 2.372568431682211,
# MAGIC   'diffuse_flows': 0.12697856844758687,
# MAGIC   'DayOfWeek_1': 1.0,
# MAGIC   'DayOfWeek_2': 0.0,
# MAGIC   'DayOfWeek_3': 0.0,
# MAGIC   'DayOfWeek_4': 0.0,
# MAGIC   'DayOfWeek_5': 0.0,
# MAGIC   'DayOfWeek_6': 0.0,
# MAGIC   'IsWeekend_1': 0.0}]
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,call endpoint
start_time = time.time()

model_serving_endpoint = (
    f"https://{host}/serving-endpoints/power-consumption-model-serving/invocations"
)

headers = {"Authorization": f"Bearer {token}"}

response = requests.post(
    f"{model_serving_endpoint}",
    headers=headers,
    json={"dataframe_records": dataframe_records[0]},
)

end_time = time.time()
execution_time = end_time - start_time

print("Response status:", response.status_code)
print("Reponse text:", response.text)
print("Execution time:", execution_time, "seconds")

# COMMAND ----------

# MAGIC %md
# MAGIC # Load Test

# COMMAND ----------

# DBTITLE 1,initialise variables and request function
num_requests = 1000

def send_request():
    random_record = random.choice(dataframe_records)
    start_time = time.time()
    response = requests.post(
        model_serving_endpoint,
        headers=headers,
        json={"dataframe_records": random_record},
    )
    end_time = time.time()
    latency = end_time - start_time
    return response.status_code, latency

# COMMAND ----------

# DBTITLE 1,send concurrent requests
total_start_time = time.time()
latencies = []

with ThreadPoolExecutor(max_workers=100) as executor:
    futures = [executor.submit(send_request) for _ in range(num_requests)]

    for future in as_completed(futures):
        status_code, latency = future.result()
        latencies.append(latency)

total_end_time = time.time()
total_execution_time = total_end_time - total_start_time

# Calculate the average latency
average_latency = sum(latencies) / len(latencies)

print("\nTotal execution time:", total_execution_time, "seconds")
print("Average latency per request:", average_latency, "seconds")

# COMMAND ----------
