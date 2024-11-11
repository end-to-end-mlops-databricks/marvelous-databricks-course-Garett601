# Databricks notebook source
# MAGIC %md
# MAGIC # Setup

# COMMAND ----------

# MAGIC %pip install ../power_consumption-0.0.1-py3-none-any.whl

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC * Create feature table in unity catalog, it will be a delta table
# MAGIC * Create online table which uses the feature delta table created in the previous step
# MAGIC * Create a feature spec. When you create a feature spec, you specify the source Delta table.
# MAGIC   * This allows the feature spec to be used in both offline and online scenarios.
# MAGIC * For online lookups, the serving endpoint automatically uses the online table to perform low-latency feature lookups.
# MAGIC * The source Delta table and the online table must use the same primary key.

# COMMAND ----------

# DBTITLE 1,Imports
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import mlflow
import pandas as pd
import requests
from databricks import feature_engineering
from databricks.feature_engineering import FeatureLookup
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    OnlineTableSpec,
    OnlineTableSpecTriggeredSchedulingPolicy,
)
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from pyspark.sql import SparkSession
from power_consumption.config import Config

# COMMAND ----------

# DBTITLE 1,Initialise Databricks Clients
workspace = WorkspaceClient()
fe = feature_engineering.FeatureEngineeringClient()

# COMMAND ----------

# DBTITLE 1,Set MLFlow registry URI
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Loading

# COMMAND ----------

# DBTITLE 1,Load config and tables
config = Config.from_yaml("../../configs/project_configs.yml")

num_features = config.processed_features.num_features
cat_features = config.processed_features.cat_features
target = config.target.target
parameters = config.hyperparameters.__dict__

catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------

# DBTITLE 1,Define feature tables (offline and online)
feature_table_name = f"{catalog_name}.{schema_name}.power_consumption_preds"
online_table_name = f"{catalog_name}.{schema_name}.power_consumption_preds_online"

# COMMAND ----------

# DBTITLE 1,Load training and test sets from UC
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()

df = pd.concat([train_set, test_set])

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Loading and Pipelines

# COMMAND ----------

# DBTITLE 1,Load MLFlow model
pipeline = mlflow.sklearn.load_model(f"models:/{catalog_name}.{schema_name}.power_consumption_model/2")

# COMMAND ----------

# DBTITLE 1,Prepare DF for feature table
preds_df = df[["id", "Temperature", "Humidity", "Wind_Speed"]]

# COMMAND ----------

predictions = pipeline.predict(df[cat_features + num_features])

# COMMAND ----------

preds_df.loc[:, "Predicted_PowerConsumption_Zone_1"] = predictions[:, 0]
preds_df.loc[:, "Predicted_PowerConsumption_Zone_2"] = predictions[:, 1]
preds_df.loc[:, "Predicted_PowerConsumption_Zone_3"] = predictions[:, 2]

# COMMAND ----------

preds_df = spark.createDataFrame(preds_df)

# COMMAND ----------

# MAGIC %md
# MAGIC # Create Feature Tables

# COMMAND ----------

# DBTITLE 1,Feature table
fe.create_table(
  name=feature_table_name,
  primary_keys=["id"],
  df=preds_df,
  description="Power consumption feature table",

)

# COMMAND ----------

# Enable Change Data Feed
spark.sql(f"""
    ALTER TABLE {feature_table_name}
    SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

# COMMAND ----------

# DBTITLE 1,Online table using feature table
spec = OnlineTableSpec(
    primary_key_columns=["id"],
    source_table_full_name=feature_table_name,
    run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict({"triggered":"true"}),
    perform_full_copy=False,
)

# COMMAND ----------

# DBTITLE 1,Create online table in Databricks
online_table_pipeline = workspace.online_tables.create(name=online_table_name, spec=spec)

# COMMAND ----------

# MAGIC %md
# MAGIC # Create FeatureLookUp and FeatureSpecTable Feature Table

# COMMAND ----------

# DBTITLE 1,Define features to look up from the feature table
features = [
    FeatureLookup(
        table_name=feature_table_name,
        lookup_key="id",
        feature_names=[
            "Temperature",
            "Humidity",
            "Wind_Speed",
            "Predicted_PowerConsumption_Zone_1",
            "Predicted_PowerConsumption_Zone_2",
            "Predicted_PowerConsumption_Zone_3",
        ],
    )
]

# COMMAND ----------

# DBTITLE 1,Feature spec for  serving
feature_spec_name = f"{catalog_name}.{schema_name}.return_predictions_2"
fe.create_feature_spec(
    name=feature_spec_name,
    features=features,
    exclude_columns=None,
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Deploy Feature Serving Endpoint

# COMMAND ----------

endpoint_name = "power-consumption-feature-serving"

# COMMAND ----------

# DBTITLE 1,Create endpoint using feature spec
workspace.serving_endpoints.create(
  name=endpoint_name,
  config=EndpointCoreConfigInput(
    served_entities=[
      ServedEntityInput(
      entity_name=feature_spec_name,
      scale_to_zero_enabled=True,
      workload_size="Small",
    )
    ]
  ),
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Call The Endpoint

# COMMAND ----------

# DBTITLE 1,Get token and host from Databricks Session
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------

# Convert the Spark DataFrame column to a list
id_list = preds_df.select("id").rdd.flatMap(lambda x: x).collect()

# COMMAND ----------

random_id = random.choice(id_list)

# COMMAND ----------

# DBTITLE 1,Call endpoint [dataframe_records]
start_time = time.time()

serving_endpoint = f"https://{host}/serving-endpoints/{endpoint_name}/invocations"

response = requests.post(
    f"{serving_endpoint}",
    headers={"Authorization": f"Bearer {token}"},
    json={"dataframe_records": [{"id":random_id}]},
)

end_time = time.time()
execution_time = end_time - start_time

print("Response status:", response.status_code)
print("Response text:", response.text)
print("Execution time:", execution_time, "seconds")

# COMMAND ----------

# DBTITLE 1,Call endpoint [dataframe_split]
start_time = time.time()

response = requests.post(
    f"{serving_endpoint}",
    headers={"Authorization": f"Bearer {token}"},
    json={"dataframe_split": {"columns": ["id"], "data": [[random_id]]}},
)

end_time = time.time()
execution_time = end_time - start_time

print("Response status:", response.status_code)
print("Response text:", response.text)
print("Execution time:", execution_time, "seconds")

# COMMAND ----------

# MAGIC %md
# MAGIC # Load Test

# COMMAND ----------

headers = {"Authorization": f"Bearer {token}"}
num_requests = 10

# COMMAND ----------

# DBTITLE 1,Function to make requests and record latency
def send_request():
    random_id = random.choice(id_list)
    start_time = time.time()
    response = requests.post(
        serving_endpoint,
        headers=headers,
        json={"dataframe_records": [{"id": random_id}]},
    )
    end_time = time.time()
    latency = end_time - start_time
    return response.status_code, latency

# COMMAND ----------

# DBTITLE 1,Send multiple requests and measure latency
# Measure total execution time
total_start_time = time.time()
latencies = []

# Send requests concurrently
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
